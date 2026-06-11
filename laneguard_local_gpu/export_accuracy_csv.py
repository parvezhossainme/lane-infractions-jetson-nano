#!/usr/bin/env python3
"""Export a simple train/test CSV from an RT-DETR JSON epoch log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/test values to CSV")
    parser.add_argument(
        "--log-file",
        default="/home/parvezdev/fydp/outputs/laneguard_local_gpu_100e/log.txt",
        help="Path to the RT-DETR epoch log.txt",
    )
    parser.add_argument(
        "--output-csv",
        default="/home/parvezdev/fydp/laneguard_local_gpu/real_metric_log.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--train-field",
        default="train_loss",
        help="Metric to use for the train curve before normalization",
    )
    parser.add_argument(
        "--test-field",
        default="test_coco_eval_bbox",
        help="Metric to use for the test curve",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=1,
        help="Index inside the test metric list to export when test-field is a list",
    )
    parser.add_argument(
        "--train-label",
        default="train_acc",
        help="Column name to write for the normalized train metric",
    )
    parser.add_argument(
        "--test-label",
        default="test_ap50",
        help="Column name to write for the test metric",
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


def get_scalar_test_value(row: dict, field: str, index: int) -> float | None:
    value = row.get(field)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) > index and isinstance(value[index], (int, float)):
        return float(value[index])
    return None


def to_accuracy_like(value: float) -> float:
    return max(0.0, min(1.0, 1.0 / (1.0 + value)))


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = load_rows(log_path)
    if not rows:
        raise RuntimeError(f"No epoch rows found in {log_path}")

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", args.train_label, args.test_label])

        written = 0
        for row in rows:
            train_value = row.get(args.train_field)
            test_value = get_scalar_test_value(row, args.test_field, args.test_index)
            if not isinstance(train_value, (int, float)) or test_value is None:
                continue

            writer.writerow([int(row["epoch"]), to_accuracy_like(float(train_value)), test_value])
            written += 1

    if written == 0:
        raise RuntimeError(
            f"No rows matched train-field={args.train_field!r} and test-field={args.test_field!r}."
        )

    print(f"Saved CSV: {output_csv}")
    print(f"Rows written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())