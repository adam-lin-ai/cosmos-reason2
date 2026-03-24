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

"""Multiview 3D bounding box inference and evaluation with Cosmos-Reason2.

Processes synchronized multiview camera videos, prompts the model to detect
vehicles with 3D bounding boxes, and optionally evaluates against ground truth
using oriented 3D IoU with Hungarian matching.
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


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

_R_CAM2FLU = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
], dtype=float)


def load_calibration(calib_path: Path) -> dict[str, dict]:
    """Load camera calibration and return a dict keyed by sensor name.

    Each value contains ``t`` (FLU position), ``rpy`` (roll-pitch-yaw in
    degrees), ``R_s2r`` (3×3 sensor-to-rig rotation), and ``t_s2r``
    (sensor-to-rig translation) from the rig JSON.
    """
    with open(calib_path) as f:
        raw = json.load(f)
    rig = json.loads(raw["data"][0]["calibration_estimate"]["rig_json"])
    sensors: dict[str, dict] = {}
    for s in rig["rig"]["sensors"]:
        pose = s["nominalSensor2Rig_FLU"]
        rpy_deg = pose["roll-pitch-yaw"]
        R_rpy = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
        R_s2r = R_rpy @ _R_CAM2FLU
        t_s2r = np.array(pose["t"], dtype=float)
        sensors[s["name"]] = {
            "t": pose["t"],
            "rpy": rpy_deg,
            "R_s2r": R_s2r,
            "t_s2r": t_s2r,
        }
    return sensors


def _facing_label(yaw: float) -> str:
    """Human-readable facing direction from yaw (degrees)."""
    if abs(yaw) < 10:
        return "forward"
    if abs(yaw - 60) < 20:
        return "forward-left"
    if abs(yaw + 60) < 20:
        return "forward-right"
    if abs(yaw - 120) < 20:
        return "rear-left"
    if abs(yaw + 120) < 20:
        return "rear-right"
    if abs(abs(yaw) - 180) < 15:
        return "rearward"
    return f"yaw {yaw:.0f}°"


def load_ego_motion(ego_path: Path) -> dict[int, dict]:
    """Load ego motion and return a dict keyed by timestamp_micros.

    Each value has ``t`` (translation np.array) and ``R``
    (scipy ``Rotation``) describing the ego pose in the world frame
    (origin = ego's initial position).
    """
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


def cam_to_world(
    cam_x: float,
    cam_y: float,
    cam_z: float,
    ego_pose: dict,
    cam_extrinsics: dict | None = None,
) -> np.ndarray:
    """Transform a point from a camera frame to the world frame.

    *cam_extrinsics* should contain ``R_s2r`` (3×3 rotation) and ``t_s2r``
    (translation) from the calibration.  When *None*, falls back to the
    FRONT_CENTER extrinsics (RPY = 0, identity axis swap + offset).
    """
    cam_pt = np.array([cam_x, cam_y, cam_z])
    if cam_extrinsics is not None:
        flu = cam_extrinsics["R_s2r"] @ cam_pt + cam_extrinsics["t_s2r"]
    else:
        flu = np.array([cam_z, -cam_x, -cam_y]) + np.array([0.42, -0.05, 1.67])
    return ego_pose["R"].apply(flu) + ego_pose["t"]


def build_camera_block(calibration: dict[str, dict]) -> str:
    """Build a text block describing each camera's pose for the prompt."""
    lines = ["Camera layout (positions in FLU ego-frame, metres):"]
    for idx, vid_name in enumerate(CAMERA_ORDER):
        sensor_name = VIDEO_TO_SENSOR[vid_name]
        cam = calibration[sensor_name]
        t = cam["t"]
        facing = _facing_label(cam["rpy"][2])
        short = vid_name.replace("auto_multiview_", "")
        lines.append(
            f"  Image {idx + 1} – {short}: "
            f"pos [{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}], facing {facing}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Video / frame helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path) -> list:
    """Extract all frames from a video as PIL Images."""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
    return frames


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array of detections from model output text."""
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


MIN_DEPTH_M = 2.0
MAX_DEPTH_M = 120.0
MIN_SIZE_M = 0.5
MAX_SIZE_M = 15.0
MAX_ABS_LATERAL_M = 60.0


def filter_detections(detections: list[dict]) -> list[dict]:
    """Remove implausible detections before coordinate conversion.

    Filters based on camera-frame constraints: minimum/maximum depth,
    reasonable bounding box dimensions, and lateral offset bounds.
    """
    kept = []
    for det in detections:
        center = det.get("center", [0, 0, 0])
        size = det.get("size", [0, 0, 0])
        if isinstance(center, dict):
            cz = center.get("z", 0)
            cx = center.get("x", 0)
        elif isinstance(center, (list, tuple)) and len(center) >= 3:
            cx, _, cz = center[0], center[1], center[2]
        else:
            continue

        if cz < MIN_DEPTH_M or cz > MAX_DEPTH_M:
            continue

        if abs(cx) > MAX_ABS_LATERAL_M:
            continue

        if isinstance(size, dict):
            dims = [size.get("x", 0), size.get("y", 0), size.get("z", 0)]
        elif isinstance(size, (list, tuple)):
            dims = list(size[:3])
        else:
            continue

        if any(d < MIN_SIZE_M or d > MAX_SIZE_M for d in dims):
            continue

        kept.append(det)
    return kept


def detections_to_gt_format(
    detections: list[dict],
    timestamp_micros: int,
    ego_pose: dict | None = None,
    calibration: dict[str, dict] | None = None,
) -> list[dict]:
    """Convert raw model detections into the ground-truth world-frame schema.

    The model outputs positions in a camera-centric frame (x=right, y=down,
    z=forward).  When *ego_pose* is provided, positions are transformed:

        camera → ego FLU (via per-camera extrinsics) → world (via ego pose)

    If a detection includes ``image_id`` (1–7 matching CAMERA_ORDER), the
    corresponding camera's extrinsics are used; otherwise FRONT_CENTER is
    assumed.  Orientation quaternions are rotated into the world frame via
    the sensor-to-rig and ego pose rotations.
    """
    def _to_list(val, default, n=3):
        """Coerce a list, dict, or scalar into a fixed-length float list."""
        if isinstance(val, dict):
            if "x" in val:
                keys = ["x", "y", "z", "w"][:n] if n <= 3 else ["w", "x", "y", "z"]
                return [float(val.get(k, 0)) for k in keys[:n]]
            return [float(v) for v in list(val.values())[:n]]
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val[:n]]
        return list(default)

    def _get_cam_extrinsics(det: dict) -> dict | None:
        if calibration is None:
            return None
        image_id = det.get("image_id")
        if image_id is not None:
            idx = int(image_id) - 1
            if 0 <= idx < len(CAMERA_ORDER):
                sensor = VIDEO_TO_SENSOR[CAMERA_ORDER[idx]]
                return calibration.get(sensor)
        return calibration.get("FRONT_CENTER")

    results = []
    for det in detections:
        cam = _to_list(det.get("center", [0, 0, 0]), [0, 0, 0], 3)
        size = _to_list(det.get("size", [0, 0, 0]), [0, 0, 0], 3)
        orientation = _to_list(det.get("orientation", [1, 0, 0, 0]), [1, 0, 0, 0], 4)

        cx, cy, cz = cam[0], cam[1], cam[2]

        qw, qx, qy, qz = orientation[0], orientation[1], orientation[2], orientation[3]
        qnorm = (qw**2 + qx**2 + qy**2 + qz**2) ** 0.5
        if qnorm > 0 and abs(qnorm - 1.0) < QUAT_NORM_THRESHOLD:
            qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        if ego_pose is not None:
            cam_ext = _get_cam_extrinsics(det)
            world = cam_to_world(cx, cy, cz, ego_pose, cam_ext)
            wx, wy, wz = float(world[0]), float(world[1]), float(world[2])

            det_rot = Rotation.from_quat([qx, qy, qz, qw])
            if cam_ext is not None:
                # For orientations use only the camera mounting rotation (RPY),
                # not the full sensor-to-rig transform.  R_s2r includes
                # R_CAM2FLU which converts position axes (camera→FLU) but is
                # already implicit in the model's quaternion convention where
                # identity means "vehicle forward = camera-z".
                rpy_rot = Rotation.from_euler(
                    "xyz", cam_ext["rpy"], degrees=True,
                )
                world_rot = ego_pose["R"] * rpy_rot * det_rot
            else:
                world_rot = ego_pose["R"] * det_rot
            wq = world_rot.as_quat()  # scipy convention: [x, y, z, w]
            qx, qy, qz, qw = float(wq[0]), float(wq[1]), float(wq[2]), float(wq[3])
        else:
            wx, wy, wz = cx, cy, cz

        results.append({
            "obstacle": {
                "category": det.get("category", "car"),
                "center": {"x": wx, "y": wy, "z": wz},
                "orientation": {"w": qw, "x": qx, "y": qy, "z": qz},
                "size": {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])},
            },
            "key": {"timestamp_micros": timestamp_micros},
        })
    return results


# ---------------------------------------------------------------------------
# Coordinate transform: camera-frame predictions → ego-vehicle frame
# ---------------------------------------------------------------------------

QUAT_NORM_THRESHOLD = 0.5

IDENTITY_QUAT = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


def normalize_obstacle_quat(obstacle: dict) -> dict:
    """Return a copy of *obstacle* with its quaternion normalised.

    Falls back to identity if the quaternion is degenerate.
    """
    o = obstacle["orientation"]
    qnorm = np.sqrt(o["w"] ** 2 + o["x"] ** 2 + o["y"] ** 2 + o["z"] ** 2)
    if qnorm > 0 and abs(qnorm - 1.0) < QUAT_NORM_THRESHOLD:
        new_o = {k: v / qnorm for k, v in o.items()}
    else:
        new_o = IDENTITY_QUAT.copy()
    return {**obstacle, "orientation": new_o}


def deduplicate_predictions(boxes: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Greedy NMS to remove duplicate detections across camera views.

    Keeps the box with the larger volume when two boxes overlap above
    *iou_threshold*.
    """
    if len(boxes) <= 1:
        return boxes

    volumes = []
    for b in boxes:
        s = b["size"]
        volumes.append(s["x"] * s["y"] * s["z"])
    order = np.argsort(volumes)[::-1]

    keep: list[int] = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for jdx in order:
            if jdx in suppressed or jdx == idx:
                continue
            iou = compute_3d_iou(boxes[idx], boxes[jdx])
            if iou >= iou_threshold:
                suppressed.add(jdx)

    return [boxes[i] for i in sorted(keep)]


def center_distance_nms(boxes: list[dict], dist_threshold: float = 3.0) -> list[dict]:
    """Simple distance-based NMS for when IoU is too strict.

    Merges boxes whose centres are within *dist_threshold* metres, keeping the
    one with the larger volume.
    """
    if len(boxes) <= 1:
        return boxes

    centers = np.array([[b["center"]["x"], b["center"]["y"], b["center"]["z"]] for b in boxes])
    volumes = np.array([b["size"]["x"] * b["size"]["y"] * b["size"]["z"] for b in boxes])
    order = np.argsort(volumes)[::-1]

    keep: list[int] = []
    suppressed = set()
    for idx in order:
        if int(idx) in suppressed:
            continue
        keep.append(int(idx))
        for jdx in order:
            if int(jdx) in suppressed or jdx == idx:
                continue
            d = np.linalg.norm(centers[idx] - centers[jdx])
            if d < dist_threshold:
                suppressed.add(int(jdx))

    return [boxes[i] for i in sorted(keep)]


# ---------------------------------------------------------------------------
# 3-D oriented IoU
# ---------------------------------------------------------------------------

def _footprint_polygon(center: dict, size: dict, orientation: dict) -> Polygon:
    """Return the bird's-eye-view footprint of an oriented 3-D box."""
    rot = Rotation.from_quat([
        orientation["x"], orientation["y"], orientation["z"], orientation["w"],
    ])
    hx, hy = size["x"] / 2, size["y"] / 2
    corners_local = np.array([
        [hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0], [hx, -hy, 0],
    ])
    corners_2d = rot.apply(corners_local)[:, :2] + np.array([center["x"], center["y"]])
    return Polygon(corners_2d)


def compute_3d_iou(box_a: dict, box_b: dict) -> float:
    """Compute oriented 3-D IoU between two bounding boxes.

    Uses BEV polygon intersection × vertical overlap.
    """
    ca, sa, oa = box_a["center"], box_a["size"], box_a["orientation"]
    cb, sb, ob = box_b["center"], box_b["size"], box_b["orientation"]

    poly_a = _footprint_polygon(ca, sa, oa)
    poly_b = _footprint_polygon(cb, sb, ob)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    inter_2d = poly_a.intersection(poly_b).area

    z_overlap = max(
        0.0,
        min(ca["z"] + sa["z"] / 2, cb["z"] + sb["z"] / 2)
        - max(ca["z"] - sa["z"] / 2, cb["z"] - sb["z"] / 2),
    )

    inter_3d = inter_2d * z_overlap
    vol_a = sa["x"] * sa["y"] * sa["z"]
    vol_b = sb["x"] * sb["y"] * sb["z"]
    union_3d = vol_a + vol_b - inter_3d
    return inter_3d / union_3d if union_3d > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pred_path: Path,
    gt_path: Path,
    *,
    normalize_quats: bool = True,
    apply_nms: bool = True,
    nms_dist: float = 3.0,
) -> dict:
    """Compare predictions to ground truth with 3-D IoU + Hungarian matching.

    Predictions are expected to already be in the FLU ego-vehicle frame
    (the camera→ego remap is applied at save time in ``detections_to_gt_format``).

    When *normalize_quats* is True, prediction quaternions are normalised
    (degenerate ones are replaced with identity).  When *apply_nms* is True,
    duplicate detections are suppressed using centre-distance NMS.

    Returns a summary dict with per-timestamp and aggregate metrics.
    """
    with open(pred_path) as f:
        preds = json.load(f)["data"]
    with open(gt_path) as f:
        gt = json.load(f)["data"]

    pred_by_ts: dict[int, list] = defaultdict(list)
    gt_by_ts: dict[int, list] = defaultdict(list)
    for p in preds:
        obs = p["obstacle"]
        if normalize_quats:
            obs = normalize_obstacle_quat(obs)
        pred_by_ts[p["key"]["timestamp_micros"]].append(obs)
    for g in gt:
        gt_by_ts[g["key"]["timestamp_micros"]].append(g["obstacle"])

    gt_total_before = sum(len(v) for v in gt_by_ts.values())
    for ts in gt_by_ts:
        seen_ids: set[int] = set()
        deduped: list[dict] = []
        for obs in gt_by_ts[ts]:
            tid = obs.get("trackline_id")
            if tid is not None and tid in seen_ids:
                continue
            if tid is not None:
                seen_ids.add(tid)
            deduped.append(obs)
        gt_by_ts[ts] = deduped
    gt_total_after = sum(len(v) for v in gt_by_ts.values())
    if gt_total_before != gt_total_after:
        print(f"  GT dedup: {gt_total_before} → {gt_total_after} objects")

    if apply_nms:
        total_before = sum(len(v) for v in pred_by_ts.values())
        for ts in pred_by_ts:
            pred_by_ts[ts] = center_distance_nms(pred_by_ts[ts], dist_threshold=nms_dist)
        total_after = sum(len(v) for v in pred_by_ts.values())
        print(f"  NMS ({nms_dist}m): {total_before} → {total_after} predictions")

    timestamps = sorted(set(pred_by_ts) | set(gt_by_ts))
    per_ts: list[dict] = []
    all_matched_ious: list[float] = []
    total_gt = 0
    total_pred = 0
    total_matched = 0

    for ts in timestamps:
        pb = pred_by_ts.get(ts, [])
        gb = gt_by_ts.get(ts, [])
        total_gt += len(gb)
        total_pred += len(pb)

        if not pb or not gb:
            per_ts.append({
                "timestamp_micros": ts,
                "num_pred": len(pb),
                "num_gt": len(gb),
                "matched_ious": [],
                "mean_iou": 0.0,
            })
            continue

        iou_matrix = np.zeros((len(pb), len(gb)))
        for i, p_box in enumerate(pb):
            for j, g_box in enumerate(gb):
                iou_matrix[i, j] = compute_3d_iou(p_box, g_box)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched = iou_matrix[row_ind, col_ind].tolist()
        all_matched_ious.extend(matched)
        total_matched += len(matched)

        mean_iou = float(np.mean(matched)) if matched else 0.0
        per_ts.append({
            "timestamp_micros": ts,
            "num_pred": len(pb),
            "num_gt": len(gb),
            "matched_ious": [round(v, 4) for v in matched],
            "mean_iou": round(mean_iou, 4),
        })

        print(
            f"  t={ts:>8d} us | pred={len(pb):>3d} | gt={len(gb):>3d} | "
            f"matched={len(matched):>3d} | mean IoU={mean_iou:.4f}"
        )

    overall_mean = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    print(SEPARATOR)
    print(f"Total GT objects:      {total_gt}")
    print(f"Total predicted:       {total_pred}")
    print(f"Total matched pairs:   {total_matched}")
    print(f"Overall mean 3D IoU:   {overall_mean:.4f}")
    print(SEPARATOR)

    summary = {
        "overall_mean_3d_iou": round(overall_mean, 4),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_matched": total_matched,
        "per_timestamp": per_ts,
    }
    return summary


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    with open(args.prompt_file) as f:
        prompts = yaml.safe_load(f)

    calibration: dict[str, dict] | None = None
    if args.calibration and args.calibration.exists():
        calibration = load_calibration(args.calibration)
        print(f"Loaded calibration for {len(calibration)} cameras")

    ego_poses: dict[int, dict] | None = None
    if args.ego_motion and args.ego_motion.exists():
        ego_poses = load_ego_motion(args.ego_motion)
        print(f"Loaded ego motion for {len(ego_poses)} timestamps")

    transformers.set_seed(0)
    print(f"Loading model: {args.model}")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)

    processor.image_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }

    print("Extracting frames from camera videos …")
    camera_frames: dict[str, list] = {}
    for cam_name in CAMERA_ORDER:
        video_path = args.scene_dir / f"{cam_name}.mp4"
        camera_frames[cam_name] = extract_frames(video_path)
        print(f"  {cam_name}: {len(camera_frames[cam_name])} frames")

    num_frames = min(len(v) for v in camera_frames.values())
    frame_indices = list(range(0, num_frames, args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]

    if calibration is not None:
        camera_block = build_camera_block(calibration)
    else:
        camera_block = "\n".join(
            f"  Image {i + 1}: {cam.replace('auto_multiview_', '')}"
            for i, cam in enumerate(CAMERA_ORDER)
        )

    user_text = (
        f"The following {len(CAMERA_ORDER)} images are synchronized frames from "
        f"cameras mounted on an autonomous vehicle.\n\n"
        f"{camera_block}\n\n"
        + prompts["user_prompt"]
    )

    all_predictions: list[dict] = []

    for step, frame_idx in enumerate(frame_indices):
        timestamp_us = frame_idx * TIMESTAMP_STEP_US
        print(f"\n[{step + 1}/{len(frame_indices)}] frame {frame_idx}  t={timestamp_us} us")

        images = [camera_frames[cam][frame_idx] for cam in CAMERA_ORDER]

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompts["system_prompt"]}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": user_text}],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            add_vision_ids=True,
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        generated_ids_trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(f"  Response preview: {response[:300]}…")

        detections = parse_json_response(response)
        detections = filter_detections(detections)
        ego_pose = ego_poses.get(timestamp_us) if ego_poses else None
        frame_results = detections_to_gt_format(
            detections, timestamp_us, ego_pose, calibration,
        )
        all_predictions.extend(frame_results)
        print(f"  Detected {len(detections)} objects")

    output = {
        "metadata": {
            "source": f"cosmos-reason2-multiview-inference ({args.model})",
            "num_rows": len(all_predictions),
            "num_columns": 2,
            "columns": ["obstacle", "key"],
            "calibration_used": calibration is not None,
            "ego_motion_used": ego_poses is not None,
        },
        "data": all_predictions,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nPredictions saved to {args.output}")
    print(f"Total objects detected: {len(all_predictions)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiview 3D bounding-box inference with Cosmos-Reason2",
    )
    p.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    p.add_argument(
        "--scene-dir", type=Path,
        default=ROOT / "assets" / "pre_trained_golden_hour" / "scene_001",
    )
    p.add_argument(
        "--prompt-file", type=Path,
        default=ROOT / "multiview_bbox_prompt.yaml",
    )
    p.add_argument(
        "--calibration", type=Path,
        default=ROOT / "camera_calibration.json",
        help="Camera calibration JSON (omit to run without calibration context)",
    )
    p.add_argument(
        "--ego-motion", type=Path,
        default=ROOT / "ego_motion.json",
        help="Ego motion JSON with per-timestamp pose (position + orientation)",
    )
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "bounding_box_predictions.json",
    )
    p.add_argument(
        "--gt-file", type=Path,
        default=ROOT / "bounding_box_ground_truth.json",
    )
    p.add_argument("--evaluate", action="store_true", help="Run 3D IoU evaluation after inference")
    p.add_argument("--evaluate-only", action="store_true", help="Skip inference; only evaluate existing predictions")
    p.add_argument(
        "--no-nms", action="store_true",
        help="Disable cross-view NMS deduplication during evaluation",
    )
    p.add_argument(
        "--nms-dist", type=float, default=3.0,
        help="Centre-distance threshold (m) for cross-view NMS (default: 3.0)",
    )
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="Cap the number of frames to process")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--min-vision-tokens", type=int, default=256)
    p.add_argument("--max-vision-tokens", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    apply_nms = not args.no_nms

    eval_kwargs = dict(
        apply_nms=apply_nms,
        nms_dist=args.nms_dist,
    )

    if args.evaluate_only:
        print(SEPARATOR)
        print(f"3D IoU Evaluation (NMS: {apply_nms})")
        print(SEPARATOR)
        summary = evaluate(args.output, args.gt_file, **eval_kwargs)
        eval_out = args.output.with_suffix(".eval.json")
        with open(eval_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation results saved to {eval_out}")
        return

    run_inference(args)

    if args.evaluate:
        print(f"\n{SEPARATOR}")
        print(f"3D IoU Evaluation (NMS: {apply_nms})")
        print(SEPARATOR)
        summary = evaluate(args.output, args.gt_file, **eval_kwargs)
        eval_out = args.output.with_suffix(".eval.json")
        with open(eval_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation results saved to {eval_out}")


if __name__ == "__main__":
    main()
