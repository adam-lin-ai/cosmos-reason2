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
#   "av==16.1.0",
#   "numpy",
#   "pillow==12.0.0",
#   "scipy",
# ]
# ///

"""Overlay predicted and ground-truth 3D bounding boxes on multiview camera videos.

Projects world-frame 3D boxes back onto each camera using the polynomial
distortion model, camera extrinsics, and per-timestamp ego motion.
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).parents[1]

VIDEO_TO_SENSOR = {
    "auto_multiview_front_wide": "FRONT_CENTER",
    "auto_multiview_front_tele": "FRONT_CENTER_NARROW",
    "auto_multiview_cross_left": "FRONT_LEFT",
    "auto_multiview_cross_right": "FRONT_RIGHT",
    "auto_multiview_rear": "REAR_CENTER",
    "auto_multiview_rear_left": "REAR_LEFT",
    "auto_multiview_rear_right": "REAR_RIGHT",
}

CAMERA_ORDER = list(VIDEO_TO_SENSOR.keys())

TIMESTAMP_STEP_US = 100_000

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

PRED_COLOR = (255, 60, 60)
GT_COLOR = (60, 255, 60)

# Rotation from OpenCV camera frame (x=right, y=down, z=forward)
# to FLU (x=forward, y=left, z=up).  Used as the base sensor2rig
# rotation when RPY = [0, 0, 0].
R_CAM2FLU = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
], dtype=float)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_full_calibration(calib_path: Path) -> dict[str, dict]:
    """Load calibration with full polynomial + extrinsics per sensor."""
    with open(calib_path) as f:
        raw = json.load(f)
    rig = json.loads(raw["data"][0]["calibration_estimate"]["rig_json"])
    sensors: dict[str, dict] = {}
    for s in rig["rig"]["sensors"]:
        props = s["properties"]
        pose = s["nominalSensor2Rig_FLU"]
        poly_coeffs = [float(c) for c in props["polynomial"].split()]

        rpy_deg = pose["roll-pitch-yaw"]
        R_rpy = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
        R_s2r = R_rpy @ R_CAM2FLU
        t_s2r = np.array(pose["t"], dtype=float)

        # Precompute r→θ lookup for polynomial inversion (θ→r)
        max_r = max(props["width"], props["height"])
        r_table = np.arange(0, max_r + 1, dtype=float)
        theta_table = np.zeros_like(r_table)
        for i, c in enumerate(poly_coeffs):
            theta_table += c * r_table**i

        sensors[s["name"]] = {
            "poly_coeffs": poly_coeffs,
            "cx": props["cx"],
            "cy": props["cy"],
            "width": props["width"],
            "height": props["height"],
            "R_s2r": R_s2r,
            "t_s2r": t_s2r,
            "r_table": r_table,
            "theta_table": theta_table,
        }
    return sensors


def load_ego_motion(ego_path: Path) -> dict[int, dict]:
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


def load_boxes(path: Path) -> dict[int, list[dict]]:
    """Load boxes grouped by timestamp."""
    with open(path) as f:
        data = json.load(f)["data"]
    by_ts: dict[int, list[dict]] = {}
    for item in data:
        ts = item["key"]["timestamp_micros"]
        by_ts.setdefault(ts, []).append(item["obstacle"])
    return by_ts


def extract_frames(video_path: Path) -> list[Image.Image]:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
    return frames


# ---------------------------------------------------------------------------
# 3-D → 2-D projection
# ---------------------------------------------------------------------------

def get_box_corners(center: dict, size: dict, orientation: dict) -> np.ndarray:
    """Return the 8 corners of an oriented 3-D box (N×3)."""
    R = Rotation.from_quat([
        orientation["x"], orientation["y"], orientation["z"], orientation["w"],
    ])
    hx = size["x"] / 2
    hy = size["y"] / 2
    hz = size["z"] / 2
    local = np.array([
        [-hx, -hy, -hz],
        [+hx, -hy, -hz],
        [+hx, +hy, -hz],
        [-hx, +hy, -hz],
        [-hx, -hy, +hz],
        [+hx, -hy, +hz],
        [+hx, +hy, +hz],
        [-hx, +hy, +hz],
    ])
    c = np.array([center["x"], center["y"], center["z"]])
    return R.apply(local) + c


def world_to_camera(points_world: np.ndarray, ego_pose: dict, cam: dict) -> np.ndarray:
    """Transform N×3 world points into the camera (OpenCV) frame."""
    points_flu = ego_pose["R"].inv().apply(points_world - ego_pose["t"])
    points_cam = (cam["R_s2r"].T @ (points_flu - cam["t_s2r"]).T).T
    return points_cam


def project_to_image(points_cam: np.ndarray, cam: dict) -> np.ndarray | None:
    """Project N×3 camera-frame points to N×2 pixel coordinates.

    Returns None if no points are in front of the camera.
    """
    depths = points_cam[:, 2]
    if np.all(depths <= 0):
        return None

    xy = points_cam[:, :2]
    r3d = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(r3d, depths)

    # Invert polynomial (θ → r_pixels) via precomputed lookup
    r_pixels = np.interp(theta, cam["theta_table"], cam["r_table"])

    safe = r3d > 1e-6
    dx = np.where(safe, xy[:, 0] / r3d, 0.0)
    dy = np.where(safe, xy[:, 1] / r3d, 0.0)

    u = cam["cx"] + r_pixels * dx
    v = cam["cy"] + r_pixels * dy
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_box_wireframe(
    draw: ImageDraw.ImageDraw,
    pixels: np.ndarray,
    depths: np.ndarray,
    color: tuple,
    width: int,
    img_w: int,
    img_h: int,
    label: str = "",
):
    """Draw a 3-D box wireframe from projected 2-D corner pixels."""
    for i, j in BOX_EDGES:
        if depths[i] <= 0 or depths[j] <= 0:
            continue
        x0, y0 = pixels[i]
        x1, y1 = pixels[j]
        if (
            min(x0, x1) > img_w + 200
            or max(x0, x1) < -200
            or min(y0, y1) > img_h + 200
            or max(y0, y1) < -200
        ):
            continue
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    if label:
        visible = [k for k in range(8) if depths[k] > 0]
        if visible:
            top_idx = min(visible, key=lambda k: pixels[k][1])
            lx, ly = pixels[top_idx]
            if 0 <= lx < img_w and 0 <= ly < img_h:
                draw.text((lx + 3, ly - 12), label, fill=color)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Overlay 3D boxes on multiview videos")
    p.add_argument(
        "--scene-dir", type=Path,
        default=ROOT / "assets" / "pre_trained_golden_hour" / "scene_001",
    )
    p.add_argument("--calibration", type=Path, default=ROOT / "camera_calibration.json")
    p.add_argument("--ego-motion", type=Path, default=ROOT / "ego_motion.json")
    p.add_argument("--predictions", type=Path, default=ROOT / "bounding_box_predictions.json")
    p.add_argument("--gt", type=Path, default=ROOT / "bounding_box_ground_truth.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "viz_output")
    p.add_argument("--cameras", nargs="*", default=None, help="Subset of cameras to render (e.g. auto_multiview_front_wide)")
    p.add_argument("--no-gt", action="store_true", help="Skip ground truth overlay")
    p.add_argument("--no-pred", action="store_true", help="Skip prediction overlay")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_full_calibration(args.calibration)
    ego_poses = load_ego_motion(args.ego_motion)
    pred_boxes = load_boxes(args.predictions) if not args.no_pred else {}
    gt_boxes = load_boxes(args.gt) if not args.no_gt else {}

    cam_list = args.cameras if args.cameras else CAMERA_ORDER

    for vid_name in cam_list:
        sensor_name = VIDEO_TO_SENSOR[vid_name]
        cam = cameras[sensor_name]
        video_path = args.scene_dir / f"{vid_name}.mp4"
        if not video_path.exists():
            print(f"  Skipping {vid_name}: video not found")
            continue

        print(f"Processing {vid_name} …")
        frames = extract_frames(video_path)

        out_path = args.output_dir / f"{vid_name}_overlay.mp4"
        out_container = av.open(str(out_path), mode="w")
        out_stream = out_container.add_stream("h264", rate=10)
        out_stream.width = cam["width"]
        out_stream.height = cam["height"]
        out_stream.pix_fmt = "yuv420p"

        for frame_idx, frame_img in enumerate(frames):
            ts = frame_idx * TIMESTAMP_STEP_US
            ego_pose = ego_poses.get(ts)
            if ego_pose is None:
                nearest_ts = min(ego_poses.keys(), key=lambda t: abs(t - ts))
                ego_pose = ego_poses[nearest_ts]

            draw = ImageDraw.Draw(frame_img)

            for boxes, color, line_w, tag in [
                (gt_boxes.get(ts, []), GT_COLOR, 2, "GT"),
                (pred_boxes.get(ts, []), PRED_COLOR, 2, "pred"),
            ]:
                for box in boxes:
                    corners_world = get_box_corners(
                        box["center"], box["size"], box["orientation"],
                    )
                    corners_cam = world_to_camera(corners_world, ego_pose, cam)
                    pixels = project_to_image(corners_cam, cam)
                    if pixels is None:
                        continue
                    label = f"{tag}:{box['category']}"
                    draw_box_wireframe(
                        draw, pixels, corners_cam[:, 2], color, line_w,
                        cam["width"], cam["height"], label,
                    )

            av_frame = av.VideoFrame.from_image(frame_img)
            for packet in out_stream.encode(av_frame):
                out_container.mux(packet)

        for packet in out_stream.encode():
            out_container.mux(packet)
        out_container.close()
        print(f"  → {out_path}")

    print("\nDone. Legend: green = ground truth, red = predictions")


if __name__ == "__main__":
    main()
