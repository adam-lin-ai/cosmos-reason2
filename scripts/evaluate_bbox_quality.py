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
#   "pillow==12.0.0",
#   "transformers==4.57.3",
#   "torch==2.9.0",
#   "torchvision",
#   "torchcodec==0.9.1; platform_machine != 'aarch64'",
# ]
# ///

"""Evaluate bounding-box overlay quality on multiview videos using Cosmos-Reason2.

Each camera view is evaluated independently: keyframes are extracted from the
overlay video and sent as images so the model can inspect bounding-box wireframes
at full resolution. Results are aggregated into an overall score.

Scoring rules:
  - Bounding boxes on vehicles not visible in a view are expected (ground truth
    may include occluded / out-of-frame objects) and should NOT lower the score.
  - Visible vehicles that have NO bounding box at all (missed detections) SHOULD
    lower the score.
"""

import argparse
import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import av
import torch
import transformers

ROOT = Path(__file__).parents[1]

PIXELS_PER_TOKEN = 32**2

CAMERA_NAMES = [
    "auto_multiview_front_wide",
    "auto_multiview_front_tele",
    "auto_multiview_cross_left",
    "auto_multiview_cross_right",
    "auto_multiview_rear",
    "auto_multiview_rear_left",
    "auto_multiview_rear_right",
]

CAMERA_LABELS = {
    "auto_multiview_front_wide": "Front Wide",
    "auto_multiview_front_tele": "Front Telephoto",
    "auto_multiview_cross_left": "Cross Left",
    "auto_multiview_cross_right": "Cross Right",
    "auto_multiview_rear": "Rear Center",
    "auto_multiview_rear_left": "Rear Left",
    "auto_multiview_rear_right": "Rear Right",
}

SYSTEM_PROMPT = """\
You are an expert evaluator of 3-D bounding-box overlays for autonomous driving.

You will be shown frames from a single camera on a vehicle. Each frame has \
coloured 3-D bounding-box wireframes drawn on top of the image. The wireframes \
are lines forming rectangular boxes projected onto the image — they may appear \
as coloured (red or green) rectangular outlines around vehicles.

Your job is to carefully judge overlay quality.

SCORING RULES:
1. Wireframe boxes that appear where NO vehicle is visible are EXPECTED. The \
ground-truth labels include vehicles that may be occluded or out of frame. \
These extra boxes must NOT lower the score.
2. If a vehicle (car, truck, motorcycle) is clearly visible but has NO \
wireframe box on it, that is a MISSED DETECTION — this SHOULD lower the score.
3. For boxes that DO overlap a vehicle, judge the FIT: does the wireframe \
tightly enclose the vehicle, or is it too large, offset, or wrong shape? \
Poor fit should lower the score."""

PER_CAMERA_PROMPT = """\
These images are frames from the "{camera_label}" camera with bounding-box \
wireframes overlayed. Look carefully at every frame.

Analyse:
1. How many distinct vehicles (cars, trucks, motorcycles) are clearly visible?
2. How many of those vehicles have a wireframe bounding box on them?
3. Are there any visible vehicles with NO box at all (missed detections)?
4. For vehicles that do have boxes, how well does the wireframe fit — is it \
tight around the vehicle, or too large / offset / wrong shape?
5. Ignore any wireframe boxes where no vehicle is visible — those are expected.

Respond with ONLY this JSON (no markdown fences, no extra text):
{{
  "camera": "{camera_label}",
  "visible_vehicles": <int>,
  "vehicles_with_bbox": <int>,
  "missed_vehicles": <int>,
  "fit_quality": "good" | "acceptable" | "poor",
  "fit_details": "<describe specific fit issues if any, or say tight/accurate>",
  "notes": "<one sentence summary>"
}}"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate bounding-box overlay quality using Cosmos-Reason2",
    )
    p.add_argument(
        "--scene-dir",
        type=Path,
        default=ROOT / "assets" / "overlayed_videos" / "scene_001",
        help="Directory containing overlayed multiview videos",
    )
    p.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="HuggingFace model name or local path",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--min-vision-tokens",
        type=int,
        default=256,
    )
    p.add_argument(
        "--max-vision-tokens",
        type=int,
        default=8192,
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Number of keyframes to extract per video",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON evaluation results",
    )
    return p.parse_args()


def discover_videos(scene_dir: Path) -> list[tuple[str, Path]]:
    """Return (camera_name, path) pairs for every overlay video found."""
    found = []
    for cam in CAMERA_NAMES:
        path = scene_dir / f"{cam}_overlay.mp4"
        if path.exists():
            found.append((cam, path))
    return found


def extract_keyframes(video_path: Path, num_frames: int) -> list[Path]:
    """Extract evenly-spaced frames from a video, save as temp PNGs."""
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total == 0:
            for _ in container.decode(video=0):
                total += 1
            container.seek(0)

        step = max(total // (num_frames + 1), 1)
        target_indices = [step * (i + 1) for i in range(num_frames)]

        idx = 0
        for frame in container.decode(video=0):
            if idx in target_indices:
                frames.append(frame.to_image())
            if len(frames) >= num_frames:
                break
            idx += 1

    paths = []
    for i, img in enumerate(frames):
        p = Path(tempfile.mktemp(suffix=f"_frame{i}.png"))
        img.save(p)
        paths.append(p)
    return paths


def build_single_camera_conversation(
    camera_label: str,
    frame_paths: list[Path],
) -> list[dict]:
    """Build conversation for evaluating a single camera view."""
    content: list[dict] = []
    for fp in frame_paths:
        content.append({"type": "image", "image": f"file://{fp}"})

    content.append({
        "type": "text",
        "text": PER_CAMERA_PROMPT.format(camera_label=camera_label),
    })

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def clean_model_output(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def try_parse_camera_result(text: str) -> dict | None:
    """Parse a single camera evaluation JSON from model output."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    import re
    match = re.search(r'\{[^{}]*"camera"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def compute_overall_score(per_camera: list[dict]) -> float:
    """Derive an overall score from per-camera entries."""
    total_visible = 0
    total_with_bbox = 0
    fit_weights = {"good": 1.0, "acceptable": 0.7, "poor": 0.3}

    fit_sum = 0.0
    fit_count = 0
    for cam in per_camera:
        v = cam.get("visible_vehicles", 0)
        b = cam.get("vehicles_with_bbox", 0)
        total_visible += v
        total_with_bbox += b
        fit = cam.get("fit_quality", "acceptable")
        fit_sum += fit_weights.get(fit, 0.7)
        fit_count += 1

    detection_rate = total_with_bbox / max(total_visible, 1)
    avg_fit = fit_sum / max(fit_count, 1)
    return round(detection_rate * 0.6 + avg_fit * 0.4, 3)


def run_inference(model, processor, conversation, max_new_tokens: int) -> str:
    """Run a single inference pass and return decoded text."""
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main():
    args = parse_args()
    transformers.set_seed(0)

    videos = discover_videos(args.scene_dir)
    if not videos:
        print(f"No overlay videos found in {args.scene_dir}")
        return

    print(f"Found {len(videos)} camera views in {args.scene_dir}:")
    for cam, path in videos:
        print(f"  {CAMERA_LABELS[cam]:20s}  {path.name}")

    print(f"\nLoading model: {args.model}")
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)

    processor.image_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }

    per_camera_results = []
    temp_files = []

    for cam_name, video_path in videos:
        label = CAMERA_LABELS[cam_name]
        print(f"\nEvaluating: {label} …")

        print(f"  Extracting {args.num_frames} keyframes …")
        frame_paths = extract_keyframes(video_path, args.num_frames)
        temp_files.extend(frame_paths)
        print(f"  Extracted {len(frame_paths)} frames")

        conversation = build_single_camera_conversation(label, frame_paths)

        print("  Running inference …")
        raw_output = run_inference(model, processor, conversation, args.max_new_tokens)
        cleaned = clean_model_output(raw_output)

        cam_result = try_parse_camera_result(cleaned)
        if cam_result is not None:
            cam_result["camera"] = label
            per_camera_results.append(cam_result)
            v = cam_result.get("visible_vehicles", "?")
            b = cam_result.get("vehicles_with_bbox", "?")
            m = cam_result.get("missed_vehicles", "?")
            fit = cam_result.get("fit_quality", "?")
            print(f"  visible={v}  with_bbox={b}  missed={m}  fit={fit}")
            notes = cam_result.get("notes", "")
            if notes:
                print(f"  {notes}")
        else:
            print(f"  WARNING: Could not parse output for {label}")
            print(f"  Raw: {cleaned[:200]}")
            per_camera_results.append({
                "camera": label,
                "visible_vehicles": -1,
                "vehicles_with_bbox": -1,
                "missed_vehicles": -1,
                "fit_quality": "unknown",
                "notes": f"Parse failure. Raw: {cleaned[:300]}",
            })

    for f in temp_files:
        f.unlink(missing_ok=True)

    overall_score = compute_overall_score(
        [c for c in per_camera_results if c.get("visible_vehicles", -1) >= 0]
    )

    result = {
        "per_camera": per_camera_results,
        "overall_score": overall_score,
        "overall_notes": _build_summary(per_camera_results, overall_score),
    }

    formatted = json.dumps(result, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(formatted)
    print("=" * 60)
    print(f"\nOverall Score: {overall_score}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(formatted)
        print(f"\nResults written to {args.output}")


def _build_summary(per_camera: list[dict], score: float) -> str:
    valid = [c for c in per_camera if c.get("visible_vehicles", -1) >= 0]
    total_visible = sum(c.get("visible_vehicles", 0) for c in valid)
    total_missed = sum(c.get("missed_vehicles", 0) for c in valid)
    fit_counts: dict[str, int] = {}
    for c in valid:
        f = c.get("fit_quality", "unknown")
        fit_counts[f] = fit_counts.get(f, 0) + 1
    fit_str = ", ".join(f"{k}: {v}" for k, v in sorted(fit_counts.items()))
    return (
        f"Score {score:.2f} across {len(valid)} cameras. "
        f"{total_visible} vehicles visible, {total_missed} missed. "
        f"Fit quality — {fit_str}."
    )


if __name__ == "__main__":
    main()
