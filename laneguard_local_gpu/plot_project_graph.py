#!/usr/bin/env python3
"""Generate the Laneguard training/test graph from a trainer-written RT-DETR log.txt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Laneguard project training curves")
    parser.add_argument(
        "--log-file",
        default="/home/parvezdev/fydp/outputs/laneguard_local_gpu_100e/log.txt",
        help="Path to the RT-DETR JSON epoch log.txt",
    )
    parser.add_argument(
        "--output-image",
        default="/home/parvezdev/fydp/outputs/laneguard_local_gpu_100e/train_test_graph_project.png",
        help="Output PNG file",
    )
    parser.add_argument(
        "--title",
        default="Laneguard Project RT-DETRv2 R18 Training/Test Curves",
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
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "epoch" in row:
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("epoch", -1)))
    return rows


def extract_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            xs.append(int(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def extract_coco_index(rows: list[dict], index: int) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        values = row.get("test_coco_eval_bbox")
        if isinstance(values, list) and len(values) > index and isinstance(values[index], (int, float)):
            xs.append(int(row["epoch"]))
            ys.append(float(values[index]))
    return xs, ys


def plot_panel(ax, x: list[int], y: list[float], title: str, ylabel: str, color: str) -> None:
    if not x:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
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
            "Wait until the training run finishes at least one epoch and writes JSON metrics to log.txt."
        )

    metric_rows = [
        row for row in rows
        if any(key.startswith("train_") or key.startswith("test_") for key in row)
    ]
    if not metric_rows:
        raise RuntimeError(
            f"Found {len(rows)} epoch markers in {log_path}, but none contained train/test metrics yet."
        )

    output_image = Path(args.output_image).resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    box_x, box_y = extract_series(metric_rows, "train_loss_bbox")
    cls_x, cls_y = extract_series(metric_rows, "train_loss_vfl")
    giou_x, giou_y = extract_series(metric_rows, "train_loss_giou")
    ap50_x, ap50_y = extract_coco_index(metric_rows, 1)
    ar100_x, ar100_y = extract_coco_index(metric_rows, 8)
    map_x, map_y = extract_coco_index(metric_rows, 0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    plot_panel(axes[0, 0], box_x, box_y, "Box Loss", "Loss", "#1f77b4")
    plot_panel(axes[0, 1], ap50_x, ap50_y, "Precision (AP@0.50)", "Score", "#2ca02c")
    plot_panel(axes[1, 0], giou_x, giou_y, "Object Loss (GIoU)", "Loss", "#ff7f0e")
    plot_panel(axes[1, 1], ar100_x, ar100_y, "Recall (AR@100)", "Score", "#d62728")
    plot_panel(axes[2, 0], cls_x, cls_y, "Class Loss (VFL)", "Loss", "#9467bd")
    plot_panel(axes[2, 1], map_x, map_y, "mAP@0.50:0.95", "Score", "#8c564b")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_image, dpi=180)
    print(f"Saved plot: {output_image}")
    print(f"Epoch rows: {len(metric_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
