#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate==1.12.0",
#   "av==16.1.0",
#   "numpy",
#   "pillow==12.0.0",
#   "pyyaml",
#   "scipy",
#   "shapely",
#   "transformers==4.57.3",
#   "torch==2.9.0",
#   "torchvision",
#   "torchcodec==0.9.1; platform_machine != 'aarch64'",
# ]
# ///

"""Multiview 3D bounding-box inference with Cosmos-Reason2 (V2).

Two inference modes:

  per_camera (default)
      Process each camera view independently.  The VLM produces 2D
      bounding boxes (its strength), which are then lifted to 3D via
      the polynomial camera model and ground-plane intersection.
      Better detection accuracy; 7× more model calls per frame.

  joint
      Feed all camera views simultaneously; the VLM outputs 2D boxes
      with an image_id, then the same geometric lifting is applied.
      Faster, but the model must juggle all views at once.

Key differences from V1:
  - Asks for 2D bounding boxes instead of 3D camera-frame coordinates.
  - Uses the full polynomial camera model + ground-plane intersection
    for precise 3D positioning.
  - Defaults to the 8B model for better detection quality.
"""

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import transformers
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 60

# ---------------------------------------------------------------------------
# Camera / sensor mapping
# ---------------------------------------------------------------------------

VIDEO_TO_SENSOR = {
    "auto_multiview_front_wide": "FRONT_CENTER",
    "auto_multiview_front_tele": "FRONT_CENTER_NARROW",
    "auto_multiview_cross_left": "FRONT_LEFT",
    "auto_multiview_cross_right": "FRONT_RIGHT",
    "auto_multiview_rear": "REAR_CENTER",
    "auto_multiview_rear_left": "REAR_LEFT",
    "auto_multiview_rear_right": "REAR_RIGHT",
}

CAMERA_ORDER = [
    "auto_multiview_front_wide",
    "auto_multiview_front_tele",
    "auto_multiview_cross_left",
    "auto_multiview_cross_right",
    "auto_multiview_rear_left",
    "auto_multiview_rear_right",
    "auto_multiview_rear",
]

PIXELS_PER_TOKEN = 32**2
TIMESTAMP_STEP_US = 100_000

# Default 3D sizes [length, width, height] in metres per category.
TYPICAL_SIZES: dict[str, list[float]] = {
    "car": [4.5, 1.85, 1.6],
    "truck": [6.0, 2.2, 2.2],
    "motorcycle": [2.1, 0.85, 1.5],
}

# Rotation from OpenCV camera frame (x-right, y-down, z-forward) to the
# FLU body frame (x-forward, y-left, z-up).
_R_CAM2FLU = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
], dtype=float)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_calibration(calib_path: Path) -> dict[str, dict]:
    """Load camera calibration with polynomial model and extrinsics.

    Each sensor dict contains:
        poly_coeffs, cx, cy, width, height,
        R_s2r (3×3), t_s2r (3,), rpy_deg (list[float])
    """
    with open(calib_path) as f:
        raw = json.load(f)
    rig = json.loads(raw["data"][0]["calibration_estimate"]["rig_json"])
    sensors: dict[str, dict] = {}
    for s in rig["rig"]["sensors"]:
        props = s["properties"]
        pose = s["nominalSensor2Rig_FLU"]
        rpy_deg = pose["roll-pitch-yaw"]
        R_rpy = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
        sensors[s["name"]] = {
            "poly_coeffs": [float(c) for c in props["polynomial"].split()],
            "cx": float(props["cx"]),
            "cy": float(props["cy"]),
            "width": int(props["width"]),
            "height": int(props["height"]),
            "R_s2r": R_rpy @ _R_CAM2FLU,
            "t_s2r": np.array(pose["t"], dtype=float),
            "rpy_deg": rpy_deg,
        }
    return sensors


def load_ego_motion(ego_path: Path) -> dict[int, dict]:
    """Load per-timestamp ego poses (translation + Rotation)."""
    with open(ego_path) as f:
        raw = json.load(f)
    poses: dict[int, dict] = {}
    for item in raw["data"]:
        ts = item["key"]["timestamp_micros"]
        loc = item["egomotion_estimate"]["location"]
        ori = item["egomotion_estimate"]["orientation"]
        poses[ts] = {
            "t": np.array([loc["x"], loc["y"], loc["z"]]),
            "R": Rotation.from_quat([ori["x"], ori["y"], ori["z"], ori["w"]]),
        }
    return poses


def get_ego_pose(ego_poses: dict[int, dict], timestamp_us: int) -> dict:
    """Return the ego pose for *timestamp_us*, falling back to nearest."""
    pose = ego_poses.get(timestamp_us)
    if pose is not None:
        return pose
    nearest = min(ego_poses, key=lambda t: abs(t - timestamp_us))
    return ego_poses[nearest]


def extract_frames(video_path: Path) -> list:
    """Return all frames from *video_path* as PIL Images."""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
    return frames


# ---------------------------------------------------------------------------
# Geometry: 2D pixel ↔ 3D world
# ---------------------------------------------------------------------------

def pixel_to_ray(u: float, v: float, cam: dict) -> np.ndarray:
    """Convert pixel (u, v) to a unit ray in the camera (OpenCV) frame.

    Uses the polynomial distortion model: θ = Σ cᵢ rⁱ
    where r = pixel distance from the principal point.
    """
    dx = u - cam["cx"]
    dy = v - cam["cy"]
    r = np.sqrt(dx * dx + dy * dy)
    if r < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    theta = sum(c * r**i for i, c in enumerate(cam["poly_coeffs"]))
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    ray = np.array([sin_t * dx / r, sin_t * dy / r, cos_t])
    return ray / np.linalg.norm(ray)


def _cam_origin_world(cam: dict, ego_pose: dict) -> np.ndarray:
    """Camera origin in the world frame."""
    return ego_pose["R"].apply(cam["t_s2r"]) + ego_pose["t"]


def _ray_to_world(ray_cam: np.ndarray, cam: dict, ego_pose: dict) -> np.ndarray:
    """Transform a direction from camera frame to world frame."""
    ray_flu = cam["R_s2r"] @ ray_cam
    return ego_pose["R"].apply(ray_flu)


MIN_GROUND_DEPTH_M = 2.0
MAX_GROUND_DEPTH_M = 200.0


def ground_plane_intersect(
    u: float,
    v: float,
    cam: dict,
    ego_pose: dict,
    ground_z: float = 0.0,
) -> tuple[np.ndarray | None, float]:
    """Find where the ray through pixel (u, v) hits the ground plane.

    Returns (world_point, depth_from_camera) or (None, -1) on failure.
    """
    ray_cam = pixel_to_ray(u, v, cam)
    ray_w = _ray_to_world(ray_cam, cam, ego_pose)
    origin_w = _cam_origin_world(cam, ego_pose)

    if abs(ray_w[2]) < 1e-8:
        return None, -1.0
    t_param = (ground_z - origin_w[2]) / ray_w[2]
    if t_param <= 0:
        return None, -1.0

    point = origin_w + t_param * ray_w
    depth = float(np.linalg.norm(point - origin_w))
    return point, depth


def bbox_to_world_box(
    bbox: list[float],
    category: str,
    depth_hint: float,
    heading_deg: float,
    cam: dict,
    ego_pose: dict,
) -> dict | None:
    """Lift a 2D bounding box to a 3D world-frame box.

    Depth is estimated by intersecting the ray through the bbox bottom-
    centre with the ground plane (z ≈ 0).  Falls back to the VLM's
    *depth_hint* along the ray through the bbox centre when that fails.
    """
    x_min, y_min, x_max, y_max = bbox
    u_mid = (x_min + x_max) / 2.0

    # Primary: ground-plane intersection through bottom-centre of bbox.
    ground_pt, depth = ground_plane_intersect(u_mid, y_max, cam, ego_pose)

    if ground_pt is not None and MIN_GROUND_DEPTH_M <= depth <= MAX_GROUND_DEPTH_M:
        center_world = ground_pt.copy()
    else:
        # Fallback: place along the ray through bbox centre at depth_hint.
        v_mid = (y_min + y_max) / 2.0
        ray_cam = pixel_to_ray(u_mid, v_mid, cam)
        pt_flu = cam["t_s2r"] + cam["R_s2r"] @ ray_cam * max(depth_hint, MIN_GROUND_DEPTH_M)
        center_world = ego_pose["R"].apply(pt_flu) + ego_pose["t"]

    size = np.array(TYPICAL_SIZES.get(category, TYPICAL_SIZES["car"]), dtype=float)
    center_world[2] = size[2] / 2.0

    # World-frame heading = ego yaw + camera mounting yaw + VLM heading.
    ego_yaw = ego_pose["R"].as_euler("ZYX", degrees=True)[0]
    cam_yaw = cam["rpy_deg"][2]
    world_yaw = ego_yaw + cam_yaw + heading_deg

    return {
        "category": category,
        "center": center_world,
        "size": size,
        "orientation": Rotation.from_euler("z", world_yaw, degrees=True),
    }


def facing_label(yaw_deg: float) -> str:
    """Human-readable facing direction from mounting yaw (degrees)."""
    if abs(yaw_deg) < 10:
        return "forward"
    if abs(yaw_deg - 60) < 20:
        return "forward-left"
    if abs(yaw_deg + 60) < 20:
        return "forward-right"
    if abs(yaw_deg - 120) < 20:
        return "rear-left"
    if abs(yaw_deg + 120) < 20:
        return "rear-right"
    if abs(abs(yaw_deg) - 180) < 15:
        return "rearward"
    return f"yaw {yaw_deg:.0f}°"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_per_camera_messages(
    cam_name: str,
    cam: dict,
    image,
    prompts: dict,
) -> list[dict]:
    """Chat conversation for a single camera view."""
    short = cam_name.replace("auto_multiview_", "")
    t = cam["t_s2r"]
    user_text = prompts["per_camera"]["user_prompt_template"].format(
        camera_name=short,
        facing=facing_label(cam["rpy_deg"][2]),
        tx=f"{t[0]:+.2f}",
        ty=f"{t[1]:+.2f}",
        tz=f"{t[2]:+.2f}",
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": prompts["per_camera"]["system_prompt"]}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]


def build_joint_messages(
    cameras: list[str],
    calibration: dict[str, dict],
    images: list,
    prompts: dict,
) -> list[dict]:
    """Chat conversation for all cameras at once."""
    lines = []
    for idx, cam_name in enumerate(cameras):
        cam = calibration[VIDEO_TO_SENSOR[cam_name]]
        short = cam_name.replace("auto_multiview_", "")
        t = cam["t_s2r"]
        lines.append(
            f"  Image {idx + 1} – {short}: "
            f"pos [{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}], "
            f"facing {facing_label(cam['rpy_deg'][2])}"
        )
    system_text = prompts["joint"]["system_prompt"].format(num_cameras=len(cameras))
    user_text = prompts["joint"]["user_prompt_template"].format(
        num_cameras=len(cameras),
        camera_layout="\n".join(lines),
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": user_text}],
        },
    ]


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    min_vision_tokens: int,
    max_vision_tokens: int,
) -> tuple:
    """Load a Cosmos-Reason2 model and its processor."""
    transformers.set_seed(0)
    print(f"Loading model: {model_name}")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }
    return model, processor


def run_model(model, processor, conversation: list[dict], max_new_tokens: int) -> str:
    """Run a chat conversation through the model, return generated text."""
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        add_vision_ids=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids, strict=False)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from model output, stripping <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        print("  WARNING: could not parse JSON from model response")
        return []


def validate_2d_detections(
    detections: list[dict],
    img_w: int = 1280,
    img_h: int = 720,
) -> list[dict]:
    """Keep only detections with plausible 2D bounding boxes."""
    valid = []
    for det in detections:
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        x0, x1 = max(0.0, min(x0, img_w)), max(0.0, min(x1, img_w))
        y0, y1 = max(0.0, min(y0, img_h)), max(0.0, min(y1, img_h))
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            continue
        out = dict(det)
        out["bbox"] = [x0, y0, x1, y1]
        out.setdefault("category", "car")
        out.setdefault("distance", 20.0)
        out.setdefault("heading", 0)
        valid.append(out)
    return valid


# ---------------------------------------------------------------------------
# Post-processing: cross-camera NMS
# ---------------------------------------------------------------------------

def _as_xyz(c) -> np.ndarray:
    """Coerce a center (ndarray or dict) to a (3,) float array."""
    if isinstance(c, np.ndarray):
        return c
    return np.array([c["x"], c["y"], c["z"]], dtype=float)


def _volume(s) -> float:
    """Coerce a size (ndarray or dict) to a scalar volume."""
    if isinstance(s, np.ndarray):
        return float(np.prod(s))
    return float(s["x"] * s["y"] * s["z"])


def center_distance_nms(boxes: list[dict], dist_threshold: float = 3.0) -> list[dict]:
    """Suppress duplicates whose centres are within *dist_threshold* metres.

    Works with both internal format (center/size as ndarrays) and output
    format (center/size as x/y/z dicts).
    """
    if len(boxes) <= 1:
        return boxes
    centers = np.array([_as_xyz(b["center"]) for b in boxes])
    volumes = np.array([_volume(b["size"]) for b in boxes])
    order = np.argsort(volumes)[::-1]

    keep: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        ii = int(i)
        if ii in suppressed:
            continue
        keep.append(ii)
        for j in order:
            jj = int(j)
            if jj in suppressed or jj == ii:
                continue
            if np.linalg.norm(centers[ii] - centers[jj]) < dist_threshold:
                suppressed.add(jj)
    return [boxes[k] for k in sorted(keep)]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def to_output_entry(det: dict, timestamp_us: int) -> dict:
    """Convert an internal detection to the ground-truth JSON schema."""
    q = det["orientation"].as_quat()  # scipy order: [x, y, z, w]
    return {
        "obstacle": {
            "category": det["category"],
            "center": {"x": float(det["center"][0]), "y": float(det["center"][1]), "z": float(det["center"][2])},
            "orientation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
            "size": {"x": float(det["size"][0]), "y": float(det["size"][1]), "z": float(det["size"][2])},
        },
        "key": {"timestamp_micros": timestamp_us},
    }


# ---------------------------------------------------------------------------
# 3-D oriented IoU (evaluation)
# ---------------------------------------------------------------------------

def _footprint_polygon(center: dict, size: dict, orientation: dict) -> Polygon:
    """Bird's-eye-view footprint of an oriented 3-D box."""
    rot = Rotation.from_quat([orientation["x"], orientation["y"], orientation["z"], orientation["w"]])
    hx, hy = size["x"] / 2, size["y"] / 2
    corners = rot.apply(np.array([
        [hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0], [hx, -hy, 0],
    ]))[:, :2] + np.array([center["x"], center["y"]])
    return Polygon(corners)


def compute_3d_iou(box_a: dict, box_b: dict) -> float:
    """Oriented 3-D IoU via BEV polygon intersection × vertical overlap."""
    ca, sa, oa = box_a["center"], box_a["size"], box_a["orientation"]
    cb, sb, ob = box_b["center"], box_b["size"], box_b["orientation"]
    pa, pb = _footprint_polygon(ca, sa, oa), _footprint_polygon(cb, sb, ob)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter_2d = pa.intersection(pb).area
    z_lo = max(ca["z"] - sa["z"] / 2, cb["z"] - sb["z"] / 2)
    z_hi = min(ca["z"] + sa["z"] / 2, cb["z"] + sb["z"] / 2)
    inter_3d = inter_2d * max(0.0, z_hi - z_lo)
    vol_a = sa["x"] * sa["y"] * sa["z"]
    vol_b = sb["x"] * sb["y"] * sb["z"]
    union = vol_a + vol_b - inter_3d
    return inter_3d / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

QUAT_NORM_THRESHOLD = 0.5
_IDENTITY_QUAT = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def _normalize_quat(obs: dict) -> dict:
    """Return *obs* with a normalised quaternion (identity if degenerate)."""
    o = obs["orientation"]
    qn = np.sqrt(o["w"]**2 + o["x"]**2 + o["y"]**2 + o["z"]**2)
    if qn > 0 and abs(qn - 1.0) < QUAT_NORM_THRESHOLD:
        return {**obs, "orientation": {k: v / qn for k, v in o.items()}}
    return {**obs, "orientation": _IDENTITY_QUAT.copy()}


def evaluate(
    pred_path: Path,
    gt_path: Path,
    *,
    apply_nms: bool = True,
    nms_dist: float = 3.0,
) -> dict:
    """3-D IoU evaluation with Hungarian matching per timestamp."""
    with open(pred_path) as f:
        preds = json.load(f)["data"]
    with open(gt_path) as f:
        gt_data = json.load(f)["data"]

    pred_by_ts: dict[int, list] = defaultdict(list)
    gt_by_ts: dict[int, list] = defaultdict(list)
    for p in preds:
        pred_by_ts[p["key"]["timestamp_micros"]].append(_normalize_quat(p["obstacle"]))
    for g in gt_data:
        gt_by_ts[g["key"]["timestamp_micros"]].append(g["obstacle"])

    # Deduplicate GT by trackline_id.
    for ts in gt_by_ts:
        seen: set[int] = set()
        deduped = []
        for obs in gt_by_ts[ts]:
            tid = obs.get("trackline_id")
            if tid is not None:
                if tid in seen:
                    continue
                seen.add(tid)
            deduped.append(obs)
        gt_by_ts[ts] = deduped

    if apply_nms:
        before = sum(len(v) for v in pred_by_ts.values())
        for ts in pred_by_ts:
            pred_by_ts[ts] = center_distance_nms(pred_by_ts[ts], dist_threshold=nms_dist)
        after = sum(len(v) for v in pred_by_ts.values())
        print(f"  NMS ({nms_dist}m): {before} → {after} predictions")

    timestamps = sorted(set(pred_by_ts) | set(gt_by_ts))
    all_matched_ious: list[float] = []
    total_gt = total_pred = total_matched = 0
    per_ts: list[dict] = []

    for ts in timestamps:
        pb = pred_by_ts.get(ts, [])
        gb = gt_by_ts.get(ts, [])
        total_gt += len(gb)
        total_pred += len(pb)

        if not pb or not gb:
            per_ts.append({"timestamp_micros": ts, "num_pred": len(pb), "num_gt": len(gb), "matched_ious": [], "mean_iou": 0.0})
            continue

        iou_mat = np.zeros((len(pb), len(gb)))
        for i, p in enumerate(pb):
            for j, g in enumerate(gb):
                iou_mat[i, j] = compute_3d_iou(p, g)

        row_idx, col_idx = linear_sum_assignment(-iou_mat)
        matched = iou_mat[row_idx, col_idx].tolist()
        all_matched_ious.extend(matched)
        total_matched += len(matched)

        mean = float(np.mean(matched)) if matched else 0.0
        per_ts.append({
            "timestamp_micros": ts, "num_pred": len(pb), "num_gt": len(gb),
            "matched_ious": [round(v, 4) for v in matched], "mean_iou": round(mean, 4),
        })
        print(
            f"  t={ts:>8d} us | pred={len(pb):>3d} | gt={len(gb):>3d} | "
            f"matched={len(matched):>3d} | mean IoU={mean:.4f}"
        )

    overall = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    print(SEPARATOR)
    print(f"Total GT objects:      {total_gt}")
    print(f"Total predicted:       {total_pred}")
    print(f"Total matched pairs:   {total_matched}")
    print(f"Overall mean 3D IoU:   {overall:.4f}")
    print(SEPARATOR)

    return {
        "overall_mean_3d_iou": round(overall, 4),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_matched": total_matched,
        "per_timestamp": per_ts,
    }


# ---------------------------------------------------------------------------
# Inference orchestration
# ---------------------------------------------------------------------------

def _infer_per_camera(
    cameras: list[str],
    camera_frames: dict[str, list],
    frame_idx: int,
    calibration: dict[str, dict],
    ego_pose: dict,
    prompts: dict,
    model,
    processor,
    args: argparse.Namespace,
) -> list[dict]:
    """Run per-camera inference for one frame and return 3D detections."""
    detections_3d: list[dict] = []
    for cam_name in cameras:
        cam = calibration[VIDEO_TO_SENSOR[cam_name]]
        image = camera_frames[cam_name][frame_idx]

        conversation = build_per_camera_messages(cam_name, cam, image, prompts)
        response = run_model(model, processor, conversation, args.max_new_tokens)

        raw = parse_json_response(response)
        dets = validate_2d_detections(raw, cam["width"], cam["height"])
        short = cam_name.replace("auto_multiview_", "")
        print(f"    {short}: {len(dets)} detections (raw: {len(raw)})")
        if args.verbose and response:
            print(f"      response: {response[:200]}…")

        for det in dets:
            box = bbox_to_world_box(
                bbox=det["bbox"],
                category=det["category"],
                depth_hint=float(det.get("distance", 20.0)),
                heading_deg=float(det.get("heading", 0)),
                cam=cam,
                ego_pose=ego_pose,
            )
            if box is not None:
                detections_3d.append(box)
    return detections_3d


def _infer_joint(
    cameras: list[str],
    camera_frames: dict[str, list],
    frame_idx: int,
    calibration: dict[str, dict],
    ego_pose: dict,
    prompts: dict,
    model,
    processor,
    args: argparse.Namespace,
) -> list[dict]:
    """Run joint (all-cameras) inference for one frame."""
    images = [camera_frames[cam][frame_idx] for cam in cameras]
    conversation = build_joint_messages(cameras, calibration, images, prompts)
    response = run_model(model, processor, conversation, args.max_new_tokens)
    print(f"  Response preview: {response[:300]}…")

    raw = parse_json_response(response)
    detections_3d: list[dict] = []
    for det in raw:
        image_id = int(det.get("image_id", 1))
        idx = image_id - 1
        if not 0 <= idx < len(cameras):
            continue
        cam = calibration[VIDEO_TO_SENSOR[cameras[idx]]]
        validated = validate_2d_detections([det], cam["width"], cam["height"])
        if not validated:
            continue
        d = validated[0]
        box = bbox_to_world_box(
            bbox=d["bbox"],
            category=d["category"],
            depth_hint=float(d.get("distance", 20.0)),
            heading_deg=float(d.get("heading", 0)),
            cam=cam,
            ego_pose=ego_pose,
        )
        if box is not None:
            detections_3d.append(box)
    print(f"  Parsed {len(detections_3d)} valid detections from {len(raw)} raw")
    return detections_3d


def run_inference(args: argparse.Namespace, cameras: list[str]) -> None:
    """Main inference loop: load data, run model, save predictions."""
    with open(args.prompt_file) as f:
        prompts = yaml.safe_load(f)

    calibration = load_calibration(args.calibration)
    print(f"Loaded calibration for {len(calibration)} cameras")

    ego_poses = load_ego_motion(args.ego_motion)
    print(f"Loaded ego motion for {len(ego_poses)} timestamps")

    model, processor = load_model(args.model, args.min_vision_tokens, args.max_vision_tokens)

    print("Extracting frames …")
    camera_frames: dict[str, list] = {}
    for cam_name in cameras:
        path = args.scene_dir / f"{cam_name}.mp4"
        camera_frames[cam_name] = extract_frames(path)
        print(f"  {cam_name}: {len(camera_frames[cam_name])} frames")

    num_frames = min(len(v) for v in camera_frames.values())
    frame_indices = list(range(0, num_frames, args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]

    infer_fn = _infer_per_camera if args.mode == "per_camera" else _infer_joint
    all_predictions: list[dict] = []

    for step, fi in enumerate(frame_indices):
        ts = fi * TIMESTAMP_STEP_US
        ego = get_ego_pose(ego_poses, ts)
        print(f"\n[{step + 1}/{len(frame_indices)}] frame {fi}  t={ts} us  (mode={args.mode})")

        dets_3d = infer_fn(cameras, camera_frames, fi, calibration, ego, prompts, model, processor, args)

        before = len(dets_3d)
        if not args.no_nms:
            dets_3d = center_distance_nms(dets_3d, dist_threshold=args.nms_dist)
        print(f"  Total: {before} detections → {len(dets_3d)} after NMS")

        for det in dets_3d:
            all_predictions.append(to_output_entry(det, ts))

    output = {
        "metadata": {
            "source": f"cosmos-reason2-v2-{args.mode} ({args.model})",
            "num_rows": len(all_predictions),
            "num_columns": 2,
            "columns": ["obstacle", "key"],
            "mode": args.mode,
            "cameras": [c.replace("auto_multiview_", "") for c in cameras],
        },
        "data": all_predictions,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(all_predictions)} predictions to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_cameras(camera_args: list[str] | None) -> list[str]:
    """Normalise short camera names to full names."""
    if camera_args is None:
        return list(CAMERA_ORDER)
    result = []
    for c in camera_args:
        full = c if c.startswith("auto_multiview_") else f"auto_multiview_{c}"
        if full not in VIDEO_TO_SENSOR:
            raise ValueError(f"Unknown camera: {c}  (valid: {', '.join(VIDEO_TO_SENSOR)})")
        result.append(full)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiview 3D bounding-box inference with Cosmos-Reason2 (V2)",
    )
    # To use the smaller 2B model instead, pass:  --model nvidia/Cosmos-Reason2-2B
    p.add_argument("--model", default="nvidia/Cosmos-Reason2-8B")
    p.add_argument(
        "--mode", choices=["per_camera", "joint"], default="per_camera",
        help="'per_camera' processes each view independently (better accuracy, 7× calls); "
             "'joint' feeds all views at once (faster)",
    )
    p.add_argument("--scene-dir", type=Path, default=ROOT / "assets" / "pre_trained_golden_hour" / "scene_001")
    p.add_argument("--prompt-file", type=Path, default=ROOT / "multiview_bbox_prompt_v2.yaml")
    p.add_argument("--calibration", type=Path, default=ROOT / "camera_calibration.json")
    p.add_argument("--ego-motion", type=Path, default=ROOT / "ego_motion.json")
    p.add_argument("--output", type=Path, default=ROOT / "bounding_box_predictions_v2.json")
    p.add_argument("--gt-file", type=Path, default=ROOT / "bounding_box_ground_truth.json")
    p.add_argument("--evaluate", action="store_true", help="Run 3D IoU evaluation after inference")
    p.add_argument("--evaluate-only", action="store_true", help="Skip inference; evaluate existing predictions")
    p.add_argument("--no-nms", action="store_true", help="Disable cross-view NMS deduplication")
    p.add_argument("--nms-dist", type=float, default=3.0, help="Centre-distance NMS threshold in metres")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="Cap the number of frames processed")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--min-vision-tokens", type=int, default=256)
    p.add_argument("--max-vision-tokens", type=int, default=1024)
    p.add_argument(
        "--cameras", nargs="*", default=None,
        help="Subset of cameras (short names OK, e.g. front_wide cross_left)",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Print full model responses")
    return p.parse_args()


def main():
    args = parse_args()
    cameras = resolve_cameras(args.cameras)

    eval_kwargs = dict(apply_nms=not args.no_nms, nms_dist=args.nms_dist)

    if args.evaluate_only:
        print(SEPARATOR)
        print("3D IoU Evaluation")
        print(SEPARATOR)
        summary = evaluate(args.output, args.gt_file, **eval_kwargs)
        eval_path = args.output.with_suffix(".eval.json")
        with open(eval_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {eval_path}")
        return

    run_inference(args, cameras)

    if args.evaluate:
        print(f"\n{SEPARATOR}")
        print("3D IoU Evaluation")
        print(SEPARATOR)
        summary = evaluate(args.output, args.gt_file, **eval_kwargs)
        eval_path = args.output.with_suffix(".eval.json")
        with open(eval_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {eval_path}")


if __name__ == "__main__":
    main()
