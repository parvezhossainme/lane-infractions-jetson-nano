#!/usr/bin/env python3
"""Prepare a COCO-format traffic+person subset from Ultralytics COCO128 labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

# COCO original IDs and names for traffic-person subset.
TARGET_CLASSES = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (10, "traffic light"),
    (13, "stop sign"),
]

# Mapping from COCO category id to YOLO class index in COCO80 ordering.
# COCO id 1->idx0, 2->idx1, ... with gaps in ids.
COCO_ID_TO_YOLO_INDEX = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


def yolo_to_coco_bbox(xc: float, yc: float, w: float, h: float, iw: int, ih: int) -> list[float]:
    bw = w * iw
    bh = h * ih
    x = (xc * iw) - bw / 2.0
    y = (yc * ih) - bh / 2.0
    x = max(0.0, x)
    y = max(0.0, y)
    bw = max(1.0, min(float(iw) - x, bw))
    bh = max(1.0, min(float(ih) - y, bh))
    return [x, y, bw, bh]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True, help="COCO128 root containing images/train2017 and labels/train2017")
    p.add_argument("--out-root", required=True, help="Output dataset root")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    img_src = src_root / "images" / "train2017"
    lbl_src = src_root / "labels" / "train2017"

    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_ann = out_root / "annotations"
    out_img_train.mkdir(parents=True, exist_ok=True)
    out_img_val.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    target_yolo_idx_to_coco_id = {
        COCO_ID_TO_YOLO_INDEX[cid]: cid for cid, _ in TARGET_CLASSES
    }

    image_files = sorted([p for p in img_src.glob("*.jpg")])
    rng = random.Random(args.seed)
    rng.shuffle(image_files)

    n_val = int(round(len(image_files) * args.val_ratio))
    val_set = set(image_files[:n_val])

    def build_split(split_name: str, files: list[Path], out_img_dir: Path) -> dict[str, Any]:
        images = []
        annotations = []
        ann_id = 1
        img_id = 1

        for img_path in files:
            # Copy images into split folders for explicit dataset layout.
            out_path = out_img_dir / img_path.name
            if not out_path.exists():
                out_path.write_bytes(img_path.read_bytes())

            with Image.open(img_path) as im:
                iw, ih = im.size

            images.append(
                {
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": iw,
                    "height": ih,
                }
            )

            lbl_path = lbl_src / f"{img_path.stem}.txt"
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_idx = int(float(parts[0]))
                    if cls_idx not in target_yolo_idx_to_coco_id:
                        continue

                    xc, yc, w, h = map(float, parts[1:])
                    bbox = yolo_to_coco_bbox(xc, yc, w, h, iw, ih)
                    area = float(bbox[2] * bbox[3])
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": target_yolo_idx_to_coco_id[cls_idx],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

            img_id += 1

        categories = [{"id": cid, "name": name, "supercategory": "traffic"} for cid, name in TARGET_CLASSES]
        return {
            "info": {"description": f"traffic subset {split_name}"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    train_files = [p for p in image_files if p not in val_set]
    val_files = [p for p in image_files if p in val_set]

    train_json = build_split("train", train_files, out_img_train)
    val_json = build_split("val", val_files, out_img_val)

    (out_ann / "instances_train.json").write_text(json.dumps(train_json, indent=2), encoding="utf-8")
    (out_ann / "instances_val.json").write_text(json.dumps(val_json, indent=2), encoding="utf-8")

    # Lightweight class counts for sanity.
    def class_counts(data: dict[str, Any]) -> dict[str, int]:
        id2name = {c["id"]: c["name"] for c in data["categories"]}
        out = {name: 0 for _, name in TARGET_CLASSES}
        for ann in data["annotations"]:
            out[id2name[ann["category_id"]]] += 1
        return out

    summary = {
        "train_images": len(train_json["images"]),
        "val_images": len(val_json["images"]),
        "train_annotations": len(train_json["annotations"]),
        "val_annotations": len(val_json["annotations"]),
        "train_class_counts": class_counts(train_json),
        "val_class_counts": class_counts(val_json),
    }
    (out_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved dataset under: {out_root}")


if __name__ == "__main__":
    main()
