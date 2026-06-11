#!/usr/bin/env python3
"""Plot RT-DETR training curves from log.txt JSON lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RT-DETR Res50 training curves")
    parser.add_argument(
        "--log-file",
        required=True,
        help="Path to RT-DETR log.txt (JSON lines with train_* and test_* metrics)",
    )
    parser.add_argument(
        "--output-image",
        default="",
        help="Output PNG file. Defaults to <log_dir>/rtdetr_res50_training_curves.png",
    )
    parser.add_argument(
        "--title",
        default="RT-DETR Res50 Backbone Training Curves",
        help="Figure title",
    )
    return parser.parse_args()


def load_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "epoch" in obj:
            rows.append(obj)

    rows.sort(key=lambda x: int(x.get("epoch", -1)))
    return rows


def extract_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, (int, float)):
            xs.append(int(row["epoch"]))
            ys.append(float(v))
    return xs, ys


def extract_coco_index(rows: list[dict], index: int) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        vals = row.get("test_coco_eval_bbox")
        if isinstance(vals, list) and len(vals) > index and isinstance(vals[index], (int, float)):
            xs.append(int(row["epoch"]))
            ys.append(float(vals[index]))
    return xs, ys


def plot_panel(ax, x: list[int], y: list[float], title: str, ylabel: str, color: str) -> None:
    if not x:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return

    ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = load_rows(log_path)
    if not rows:
        raise RuntimeError(
            f"No JSON epoch rows found in {log_path}. "
            "This usually means training has not finished an epoch yet, or the run did not write per-epoch JSON stats. "
            "Wait until log.txt contains lines with an 'epoch' field, then rerun this plot command."
        )

    epoch_rows = [row for row in rows if any(key.startswith('train_') or key.startswith('test_') for key in row)]
    if not epoch_rows:
        raise RuntimeError(
            f"Found {len(rows)} epoch markers in {log_path}, but none contained train/test metrics to plot. "
            "Wait for at least one completed epoch with JSON metrics in log.txt."
        )

    output_image = Path(args.output_image).resolve() if args.output_image else log_path.parent / "rtdetr_res50_training_curves.png"
    output_image.parent.mkdir(parents=True, exist_ok=True)

    box_x, box_y = extract_series(epoch_rows, "train_loss_bbox")
    cls_x, cls_y = extract_series(epoch_rows, "train_loss_vfl")
    obj_x, obj_y = extract_series(epoch_rows, "train_loss_giou")

    # COCO metric indices in test_coco_eval_bbox:
    # 0: AP@[0.50:0.95], 1: AP@0.50, 8: AR@100
    prec_x, prec_y = extract_coco_index(epoch_rows, 1)
    rec_x, rec_y = extract_coco_index(epoch_rows, 8)
    map_x, map_y = extract_coco_index(epoch_rows, 0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    plot_panel(axes[0, 0], box_x, box_y, "Box Loss", "Loss", "#1f77b4")
    plot_panel(axes[0, 1], prec_x, prec_y, "Precision (AP@0.50)", "Score", "#2ca02c")
    plot_panel(axes[1, 0], obj_x, obj_y, "Object Loss (GIoU)", "Loss", "#ff7f0e")
    plot_panel(axes[1, 1], rec_x, rec_y, "Recall (AR@100)", "Score", "#d62728")
    plot_panel(axes[2, 0], cls_x, cls_y, "Class Loss (VFL)", "Loss", "#9467bd")
    plot_panel(axes[2, 1], map_x, map_y, "mAP@0.50:0.95", "Score", "#8c564b")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_image, dpi=180)
    print(f"Saved plot: {output_image}")
    print(f"Epoch rows: {len(epoch_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
