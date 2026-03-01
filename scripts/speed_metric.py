#!/usr/bin/env python3
"""
Speed metric pipeline for Sample-Main (or any video).

Generates:
- Annotated speed-violation output video
- JSON report with required metrics:
  FPS, Latency (ms), Params (M), mAP50, F1 Score, Precision, Recall
- Input/output duration parity check
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
from ultralytics import RTDETR


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_reference_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "opened": False,
            "frames": -1,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_sec": None,
        }

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = (frames / fps) if fps > 0 else None
    return {
        "opened": True,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def run_speed_metric(
    input_video: str,
    output_video: str,
    output_report: str,
    confidence: float,
    min_speed_kmh: float,
    max_speed_kmh: float,
    pixels_per_meter: float,
    speed_zone: str,
    max_frames: int | None,
) -> dict:
    from violations.speed_detection import SpeedDetector

    input_meta = get_video_meta(input_video)
    if not input_meta["opened"]:
        raise RuntimeError(f"Could not open input video: {input_video}")

    model = RTDETR("rtdetr-l.pt")
    detector = SpeedDetector(
        pixels_per_meter=pixels_per_meter,
        fps=input_meta["fps"] if input_meta["fps"] > 0 else 30.0,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        speed_zone=speed_zone,
    )

    os.makedirs(Path(output_video).parent, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    fps_out = input_meta["fps"] if input_meta["fps"] > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps_out,
        (input_meta["width"], input_meta["height"]),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    frame_idx = 0
    latencies_ms: list[float] = []
    all_violations: list[dict] = []
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        t0 = time.perf_counter()
        results = model(frame, conf=confidence, verbose=False)

        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "class_id": class_id,
                            "confidence": float(box.conf[0]),
                        }
                    )

        violations = detector.update(detections, frame_idx)
        all_violations.extend(violations)
        vis_frame = detector.visualize(frame, violations)
        writer.write(vis_frame)

        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        frame_idx += 1

    elapsed = time.perf_counter() - start
    cap.release()
    writer.release()

    output_meta = get_video_meta(output_video)

    reference_metrics = load_reference_metrics(ROOT_DIR / "outputs" / "paper_metrics.json")
    runtime_ref = reference_metrics.get("runtime", {})
    detection_ref = reference_metrics.get("detection", {})

    speeding = sum(1 for v in all_violations if v.get("type") == "SPEEDING")
    too_slow = sum(1 for v in all_violations if v.get("type") == "TOO_SLOW")

    fps_runtime = (frame_idx / elapsed) if elapsed > 0 else 0.0
    lat_mean = statistics.mean(latencies_ms) if latencies_ms else 0.0
    lat_median = statistics.median(latencies_ms) if latencies_ms else 0.0

    duration_match = (
        input_meta["frames"] == output_meta["frames"]
        and abs((input_meta["fps"] or 0.0) - (output_meta["fps"] or 0.0)) < 1e-6
    )

    report = {
        "video": {
            "input_path": input_video,
            "output_path": output_video,
            "input_frames": input_meta["frames"],
            "output_frames": output_meta["frames"],
            "input_fps": input_meta["fps"],
            "output_fps": output_meta["fps"],
            "input_duration_sec": input_meta["duration_sec"],
            "output_duration_sec": output_meta["duration_sec"],
            "duration_match": duration_match,
        },
        "speed_violation": {
            "frames_processed": frame_idx,
            "total_violations": len(all_violations),
            "speeding_violations": speeding,
            "too_slow_violations": too_slow,
        },
        "required_metrics": {
            "FPS": fps_runtime,
            "Latency_ms": lat_mean,
            "Latency_ms_median": lat_median,
            "Params_M": runtime_ref.get("params_m"),
            "mAP50": detection_ref.get("mAP50"),
            "F1_Score": detection_ref.get("f1_score"),
            "Precision": detection_ref.get("precision"),
            "Recall": detection_ref.get("recall"),
        },
        "metric_sources": {
            "runtime_fps_latency": "Computed during this speed metric run",
            "params_map_f1_precision_recall": "outputs/paper_metrics.json",
        },
    }

    os.makedirs(Path(output_report).parent, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speed metrics and duration validation report")
    parser.add_argument("--input", default="samples/videos/Sample-Main.mp4", help="Input video path")
    parser.add_argument(
        "--output-video",
        default="outputs/output/speed_detection_sample_main_complete.mp4",
        help="Annotated output video path",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/output/speed_detection_sample_main_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--min-speed", type=float, default=20.0, help="Minimum speed threshold km/h")
    parser.add_argument("--max-speed", type=float, default=60.0, help="Maximum speed threshold km/h")
    parser.add_argument("--pixels-per-meter", type=float, default=10.0, help="Scene calibration value")
    parser.add_argument("--speed-zone", default="urban", help="Speed zone label")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests")
    args = parser.parse_args()

    report = run_speed_metric(
        input_video=args.input,
        output_video=args.output_video,
        output_report=args.output_report,
        confidence=args.confidence,
        min_speed_kmh=args.min_speed,
        max_speed_kmh=args.max_speed,
        pixels_per_meter=args.pixels_per_meter,
        speed_zone=args.speed_zone,
        max_frames=args.max_frames,
    )

    print("Speed metric generation complete")
    print(f"Report: {args.output_report}")
    print(f"Output video: {args.output_video}")
    print(f"Duration match: {report['video']['duration_match']}")


if __name__ == "__main__":
    main()
