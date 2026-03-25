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
driving. You will receive videos from multiple calibrated cameras mounted on \
a vehicle. Each video already has bounding-box wireframes overlayed on it — \
these are the predictions being evaluated.

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
Loose, offset, or incorrectly sized boxes should reduce the score.

Evaluate each camera view independently, then provide an overall assessment."""

USER_PROMPT_TEMPLATE = """\
The videos above show {num_cameras} camera views from an autonomous vehicle, \
each with bounding-box wireframes overlayed. The cameras are, in order:
{camera_list}

For each camera view, please:
1. Count how many vehicles are clearly visible in the scene.
2. Count how many of those visible vehicles have a bounding box on them.
3. Assess the fit quality of each bounding box (tight/good, loose, offset, \
wrong size).
4. Note any bounding boxes that appear where no vehicle is visible — remember, \
these are expected and should NOT lower the score.

After analyzing all views, output your evaluation as JSON with this structure:
{{
  "per_camera": [
    {{
      "camera": "<camera name>",
      "visible_vehicles": <int>,
      "vehicles_with_bbox": <int>,
      "missed_vehicles": <int>,
      "fit_quality": "good" | "acceptable" | "poor",
      "notes": "<brief explanation>"
    }}
  ],
  "overall_score": <float 0.0 to 1.0>,
  "overall_notes": "<summary of strengths and weaknesses>"
}}

The overall_score should be 0.0–1.0 where:
- 1.0 = all visible vehicles are detected, boxes fit tightly
- 0.7+ = most vehicles detected, boxes are reasonably accurate
- 0.4–0.7 = significant missed detections or poor box fit
- <0.4 = many missed vehicles or very poor box alignment

Output ONLY the JSON, no extra text."""


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
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # Attempt to parse and pretty-print the score
    try:
        result = json.loads(output_text)
        score = result.get("overall_score", "N/A")
        print(f"\nOverall Score: {score}")
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
    except (json.JSONDecodeError, TypeError):
        print("\n(Could not parse structured JSON from model output)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
