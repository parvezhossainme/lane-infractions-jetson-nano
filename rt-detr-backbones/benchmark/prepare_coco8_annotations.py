#!/usr/bin/env python3
"""Convert YOLO-format coco8 labels into COCO-format instances JSON for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# COCO class names in YOLO/Ultralytics order (0..79).
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Mapping from 80 contiguous YOLO ids to official COCO category ids.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create COCO val annotation JSON from coco8 YOLO labels")
    parser.add_argument("--images", default="datasets/coco8/images/val", help="Path to val images directory")
    parser.add_argument("--labels", default="datasets/coco8/labels/val", help="Path to val YOLO labels directory")
    parser.add_argument(
        "--output",
        default="datasets/coco8/annotations/instances_val2017.json",
        help="Output COCO annotation JSON file",
    )
    return parser.parse_args()


def yolo_to_xywh_abs(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    abs_w = w * img_w
    abs_h = h * img_h
    x_min = (xc * img_w) - (abs_w / 2.0)
    y_min = (yc * img_h) - (abs_h / 2.0)
    return x_min, y_min, abs_w, abs_h


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    images = []
    annotations = []
    ann_id = 1

    for idx, image_path in enumerate(image_files, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        height, width = img.shape[:2]

        image_id = idx
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x, y, bw, bh = yolo_to_xywh_abs(xc, yc, w, h, width, height)
            category_id = COCO80_TO_91[cls]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": COCO80_TO_91[i], "name": COCO80_NAMES[i], "supercategory": "none"}
        for i in range(80)
    ]

    coco = {
        "info": {"description": "coco8 converted for RT-DETR evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path.write_text(json.dumps(coco), encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
