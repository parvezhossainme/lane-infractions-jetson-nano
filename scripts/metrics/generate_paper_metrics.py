#!/usr/bin/env python3
"""Generate paper-ready metrics for RT-DETR.

This script can compute:
- Runtime metrics: FPS, latency (ms), parameters (M)
- Detection metrics (optional, requires labeled dataset): mAP50, precision, recall, F1

Outputs:
- JSON report
- Markdown summary table
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch
from ultralytics import RTDETR


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _first_sample_image(project_root: Path) -> Optional[Path]:
    samples_dir = project_root / "samples" / "images"
    if not samples_dir.exists():
        return None

    candidates = sorted(
        [p for p in samples_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    return candidates[0] if candidates else None


def runtime_metrics(
    model: RTDETR,
    image_path: Path,
    conf: float,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    params_m = _to_float(sum(param.numel() for param in model.model.parameters()) / 1e6)

    for _ in range(warmup_runs):
        _ = model(image, conf=conf, verbose=False)

    latencies_ms = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        _ = model(image, conf=conf, verbose=False)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    latency_mean = _to_float(statistics.fmean(latencies_ms))
    latency_median = _to_float(statistics.median(latencies_ms))
    latency_std = _to_float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    fps_mean = _to_float(1000.0 / latency_mean) if latency_mean and latency_mean > 0 else None

    return {
        "image": str(image_path),
        "params_m": params_m,
        "latency_ms_mean": latency_mean,
        "latency_ms_median": latency_median,
        "latency_ms_std": latency_std,
        "fps_mean": fps_mean,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "device_cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def detection_metrics(
    model: RTDETR,
    dataset_yaml: Path,
    imgsz: int,
    conf: float,
    iou: float,
    split: str,
) -> Dict[str, Any]:
    metrics = {
        "dataset": str(dataset_yaml),
        "split": split,
        "mAP50": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "status": "not_run",
        "error": None,
    }

    try:
        val_result = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            plots=False,
            verbose=False,
        )

        box = getattr(val_result, "box", None)
        map50 = _to_float(getattr(box, "map50", None)) if box is not None else None
        precision = _to_float(getattr(box, "mp", None)) if box is not None else None
        recall = _to_float(getattr(box, "mr", None)) if box is not None else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1_score = _to_float((2 * precision * recall) / (precision + recall))
        else:
            f1_score = None

        metrics.update(
            {
                "mAP50": map50,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "status": "ok",
            }
        )
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc)

    return metrics


def format_value(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def write_markdown(output_path: Path, report: Dict[str, Any]) -> None:
    runtime = report["runtime"]
    detection = report["detection"]

    lines = [
        "# Paper Metrics Summary",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model: {report['model']}",
        f"- Device: {runtime['device_name']}",
        "",
        "## Runtime Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| FPS | {format_value(runtime.get('fps_mean'), 2)} |",
        f"| Latency (ms, mean) | {format_value(runtime.get('latency_ms_mean'), 2)} |",
        f"| Latency (ms, median) | {format_value(runtime.get('latency_ms_median'), 2)} |",
        f"| Latency (ms, std) | {format_value(runtime.get('latency_ms_std'), 2)} |",
        f"| Params (M) | {format_value(runtime.get('params_m'), 3)} |",
        "",
        "## Detection Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mAP50 | {format_value(detection.get('mAP50'))} |",
        f"| Precision | {format_value(detection.get('precision'))} |",
        f"| Recall | {format_value(detection.get('recall'))} |",
        f"| F1 Score | {format_value(detection.get('f1_score'))} |",
        "",
        f"- Detection status: {detection.get('status')}",
    ]

    if detection.get("error"):
        lines.append(f"- Detection error: {detection['error']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper metrics for RT-DETR.")
    parser.add_argument("--model", default="models/rtdetr-l.pt", help="Path/name of RT-DETR weights")
    parser.add_argument("--image", default=None, help="Path to image for runtime benchmark")
    parser.add_argument("--dataset", default=None, help="Dataset YAML path for mAP/precision/recall/F1")
    parser.add_argument(
        "--default-dataset",
        default="coco8.yaml",
        help="Fallback dataset to auto-evaluate when --dataset is not provided",
    )
    parser.add_argument("--split", default="val", help="Dataset split for evaluation (default: val)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for validation")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=60, help="Timed iterations")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    image_path = Path(args.image).resolve() if args.image else _first_sample_image(project_root)
    if image_path is None:
        raise FileNotFoundError("No image provided and no sample image found in samples/images")

    outdir = (project_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(args.model)

    runtime = runtime_metrics(
        model=model,
        image_path=image_path,
        conf=args.conf,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
    )

    dataset_source = "user"
    if args.dataset:
        dataset_arg = args.dataset
    else:
        dataset_arg = args.default_dataset
        dataset_source = "fallback"

    dataset_path = Path(dataset_arg)
    dataset_yaml = dataset_path.resolve() if dataset_path.exists() else dataset_path

    detection = detection_metrics(
        model=model,
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
    )
    detection["dataset_source"] = dataset_source

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "runtime": runtime,
        "detection": detection,
    }

    json_path = outdir / "paper_metrics.json"
    md_path = outdir / "paper_metrics_summary.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print("Runtime -> "
          f"FPS: {format_value(runtime['fps_mean'], 2)}, "
          f"Latency(ms): {format_value(runtime['latency_ms_mean'], 2)}, "
          f"Params(M): {format_value(runtime['params_m'], 3)}")
    print("Detection -> "
          f"mAP50: {format_value(detection.get('mAP50'))}, "
          f"Precision: {format_value(detection.get('precision'))}, "
          f"Recall: {format_value(detection.get('recall'))}, "
          f"F1: {format_value(detection.get('f1_score'))}, "
          f"Status: {detection.get('status')}")


if __name__ == "__main__":
    main()
