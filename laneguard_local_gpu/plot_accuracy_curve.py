#!/usr/bin/env python3
"""Plot a simple train/test curve from a CSV, JSON, or JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/test curves")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to a CSV, JSON, or JSONL file with epoch/train/test values",
    )
    parser.add_argument(
        "--output-image",
        default="train_test_accuracy.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--title",
        default="model accuracy",
        help="Plot title",
    )
    parser.add_argument(
        "--train-field",
        default="train_loss",
        help="Column name or metric key for the train curve",
    )
    parser.add_argument(
        "--test-field",
        default="test_ap50",
        help="Column name or metric key for the test curve",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=1,
        help="Index inside a list-valued test field when needed",
    )
    return parser.parse_args()


def normalize_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def pick_value(row: dict, candidates: tuple[str, ...]) -> float | None:
    normalized = {normalize_key(str(key)): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_rows(input_path: Path) -> list[dict]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = input_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text.startswith("["):
            data = json.loads(text)
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            raise ValueError("JSON input must be a list of objects")
        rows: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows

    raise ValueError(f"Unsupported file type: {input_path.suffix}")


def extract_series(
    rows: list[dict],
    train_field: str,
    test_field: str,
    test_index: int,
) -> tuple[list[float], list[float], list[float]]:
    epochs: list[float] = []
    train_values: list[float] = []
    test_values: list[float] = []

    train_keys = (
        train_field,
        "train_loss",
        "train_acc",
        "train_accuracy",
        "train",
        "accuracy_train",
        "train_score",
    )
    test_keys = (
        test_field,
        "test_ap50",
        "test_acc",
        "test_accuracy",
        "test",
        "accuracy_test",
        "test_score",
    )
    epoch_keys = ("epoch", "epochs", "step", "iteration")

    for index, row in enumerate(rows):
        epoch_value = pick_value(row, epoch_keys)
        train_value = pick_value(row, train_keys)
        test_value = pick_value(row, test_keys)

        if test_value is None:
            raw_test_value = row.get(test_field)
            if isinstance(raw_test_value, list) and len(raw_test_value) > test_index:
                candidate = raw_test_value[test_index]
                if isinstance(candidate, (int, float)):
                    test_value = float(candidate)

        if train_value is None or test_value is None:
            continue

        epochs.append(float(epoch_value if epoch_value is not None else index))
        train_values.append(train_value)
        test_values.append(test_value)

    return epochs, train_values, test_values


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_rows(input_path)
    epochs, train_values, test_values = extract_series(rows, args.train_field, args.test_field, args.test_index)
    if not epochs:
        raise RuntimeError(
            "No train/test accuracy rows found. Expected columns like "
            f"epoch, {args.train_field}, and {args.test_field}."
        )

    output_image = Path(args.output_image).resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label=args.train_field, linewidth=2.0, color="#1f77b4")
    plt.plot(epochs, test_values, label=args.test_field, linewidth=2.0, color="#d62728")
    plt.title(args.title)
    plt.xlabel("epoch")
    plt.ylabel("metric value")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_image, dpi=180)
    print(f"Saved plot: {output_image}")
    print(f"Epoch rows: {len(epochs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())