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

Feeds overlayed multiview camera videos to Cosmos-Reason2 and asks the model to
assess how well the drawn bounding boxes fit the vehicles in the scene. The model
produces per-camera assessments and an overall quality score.

Scoring rules communicated to the model:
  - Bounding boxes on vehicles not visible in a view are expected (ground truth
    may include occluded / out-of-frame objects) and should NOT lower the score.
  - Visible vehicles that have NO bounding box at all (missed detections) SHOULD
    lower the score.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
You are an expert evaluator of 3-D bounding-box predictions for autonomous \
driving. You will receive videos from calibrated cameras mounted on a vehicle. \
Each video already has bounding-box wireframes overlayed on it — these are \
the predictions being evaluated.

Your task is to judge how well the overlayed bounding boxes match the actual \
vehicles visible in the scene.

IMPORTANT SCORING RULES — read carefully:
1. Some bounding boxes may appear in locations where NO vehicle is visible in \
that particular camera view. This is EXPECTED and NORMAL — the ground-truth \
annotations include vehicles that may be occluded, out of frame, or only \
visible from other camera angles. These "extra" boxes must NOT reduce the \
score.
2. If a vehicle is clearly visible in a camera view but has NO bounding box \
on it at all, that is a MISSED DETECTION and SHOULD reduce the score.
3. For vehicles that DO have bounding boxes, evaluate how well the box fits: \
is it tightly aligned with the vehicle's shape, position, and orientation? \
Loose, offset, or incorrectly sized boxes should reduce the score."""

USER_PROMPT_TEMPLATE = """\
You are given EXACTLY {num_cameras} videos — one per camera. The cameras are:
{camera_list}

There are NO other cameras. Your output must contain EXACTLY {num_cameras} \
entries in "per_camera", one for each camera listed above, in the same order. \
Do NOT invent additional cameras.

For each of the {num_cameras} camera views:
1. Count how many vehicles are clearly visible in the scene.
2. Count how many of those visible vehicles have a bounding box on them.
3. Assess the fit quality of each bounding box (tight/good, loose, offset, \
wrong size).
4. Note any bounding boxes that appear where no vehicle is visible — remember, \
these are expected and should NOT lower the score.

After analyzing all {num_cameras} views, output your evaluation as JSON:
{{
  "per_camera": [
    {{
      "camera": "<camera name from the list above>",
      "visible_vehicles": <int>,
      "vehicles_with_bbox": <int>,
      "missed_vehicles": <int>,
      "fit_quality": "good" | "acceptable" | "poor",
      "notes": "<brief explanation, one sentence>"
    }}
  ],
  "overall_score": <float 0.0 to 1.0>,
  "overall_notes": "<one-sentence summary>"
}}

Score guide: 1.0 = perfect, 0.7+ = good, 0.4–0.7 = significant issues, \
<0.4 = poor.

CRITICAL: Output EXACTLY {num_cameras} per_camera entries, then \
overall_score and overall_notes, then stop. No extra text, no markdown \
fences."""


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
        default=4096,
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


def build_conversation(
    videos: list[tuple[str, Path]],
) -> list[dict]:
    """Build the chat conversation with all video inputs and the evaluation prompt."""
    video_content: list[dict] = []
    for _cam_name, video_path in videos:
        video_content.append({"type": "video", "video": str(video_path)})

    camera_list = "\n".join(
        f"  {i + 1}. {CAMERA_LABELS[cam]}" for i, (cam, _) in enumerate(videos)
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        num_cameras=len(videos),
        camera_list=camera_list,
    )
    video_content.append({"type": "text", "text": user_prompt})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": video_content,
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


def try_parse_result(text: str, expected_cameras: int) -> dict | None:
    """Parse JSON from model output, handling truncation and hallucinated cameras.

    If the model produced more per_camera entries than expected, truncate to
    the correct count and recompute an overall_score from the kept entries.
    If the JSON is incomplete (truncated), attempt to salvage what we can.
    """
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = _try_salvage_truncated(text)
        if result is None:
            return None

    if "per_camera" in result:
        result["per_camera"] = result["per_camera"][:expected_cameras]

    if "overall_score" not in result and "per_camera" in result:
        result["overall_score"] = _compute_score(result["per_camera"])
        result["overall_notes"] = "(score computed from per-camera data)"

    return result


def _try_salvage_truncated(text: str) -> dict | None:
    """Try to recover valid per_camera entries from truncated JSON."""
    import re

    match = re.search(r'"per_camera"\s*:\s*\[', text)
    if not match:
        return None

    entries = []
    entry_pattern = re.compile(
        r'\{\s*"camera"\s*:.*?"notes"\s*:\s*"[^"]*"\s*\}', re.DOTALL
    )
    for m in entry_pattern.finditer(text, match.end()):
        try:
            entries.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue

    if not entries:
        return None

    score_match = re.search(r'"overall_score"\s*:\s*([\d.]+)', text)
    notes_match = re.search(r'"overall_notes"\s*:\s*"([^"]*)"', text)

    result: dict = {"per_camera": entries}
    if score_match:
        result["overall_score"] = float(score_match.group(1))
    if notes_match:
        result["overall_notes"] = notes_match.group(1)
    return result


def _compute_score(per_camera: list[dict]) -> float:
    """Derive an overall score from per-camera entries."""
    total_visible = 0
    total_with_bbox = 0
    fit_scores = {"good": 1.0, "acceptable": 0.7, "poor": 0.3}

    fit_sum = 0.0
    fit_count = 0
    for cam in per_camera:
        v = cam.get("visible_vehicles", 0)
        b = cam.get("vehicles_with_bbox", 0)
        total_visible += v
        total_with_bbox += b
        fit = cam.get("fit_quality", "acceptable")
        fit_sum += fit_scores.get(fit, 0.7)
        fit_count += 1

    detection_rate = total_with_bbox / max(total_visible, 1)
    avg_fit = fit_sum / max(fit_count, 1)
    return round(detection_rate * 0.6 + avg_fit * 0.4, 3)


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
    processor.video_processor.size = {
        "shortest_edge": args.min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": args.max_vision_tokens * PIXELS_PER_TOKEN,
    }

    conversation = build_conversation(videos)

    print("Processing inputs …")
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )
    inputs = inputs.to(model.device)

    print("Running inference …")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.2,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output_text = clean_model_output(output_text)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)

    result = try_parse_result(output_text, len(videos))
    if result is not None:
        formatted = json.dumps(result, indent=2)
        print(formatted)
        print("=" * 60)
        print(f"\nOverall Score: {result.get('overall_score', 'N/A')}")
        if "per_camera" in result:
            print("\nPer-Camera Breakdown:")
            for cam_eval in result["per_camera"]:
                cam_name = cam_eval.get("camera", "?")
                visible = cam_eval.get("visible_vehicles", "?")
                with_bbox = cam_eval.get("vehicles_with_bbox", "?")
                missed = cam_eval.get("missed_vehicles", "?")
                fit = cam_eval.get("fit_quality", "?")
                print(
                    f"  {cam_name:20s}  visible={visible}  "
                    f"with_bbox={with_bbox}  missed={missed}  fit={fit}"
                )
                notes = cam_eval.get("notes")
                if notes:
                    print(f"    {notes}")
        overall_notes = result.get("overall_notes")
        if overall_notes:
            print(f"\nSummary: {overall_notes}")
        output_text = formatted
    else:
        print(output_text)
        print("=" * 60)
        print("\n(Could not parse structured JSON from model output)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
