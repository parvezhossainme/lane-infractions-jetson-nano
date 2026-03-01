#!/usr/bin/env python3
"""
Illegal stopping metric pipeline.

Outputs:
- Annotated illegal-stopping video
- JSON report with:
  FPS, Latency (ms), Params (M), mAP50, F1 Score, Precision, Recall
  plus illegal-stopping violation stats and duration parity check
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

from violations.illegal_stopping import IllegalStoppingDetector, ZoneType


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


def setup_default_zones(detector: IllegalStoppingDetector, width: int, height: int) -> None:
    """Create scene-agnostic default zones by splitting frame into 4 vertical bands."""
    zone_width = width // 4
    detector.add_zone(0, 0, zone_width, height, ZoneType.NO_STOPPING, 0)
    detector.add_zone(zone_width, 0, zone_width * 2, height, ZoneType.NO_PARKING, 180)
    detector.add_zone(zone_width * 2, 0, zone_width * 3, height, ZoneType.BUS_STOP, 60)
    detector.add_zone(zone_width * 3, 0, width, height, ZoneType.ALLOWED, 0)


def run_illegal_stopping_metric(
    input_video: str,
    output_video: str,
    output_report: str,
    confidence: float,
    movement_threshold: float,
    stationary_time: float,
    max_frames: int | None,
) -> dict:
    input_meta = get_video_meta(input_video)
    if not input_meta["opened"]:
        raise RuntimeError(f"Could not open input video: {input_video}")

    model = RTDETR("rtdetr-l.pt")
    detector = IllegalStoppingDetector(
        movement_threshold=movement_threshold,
        stationary_time=stationary_time,
    )
    setup_default_zones(detector, input_meta["width"], input_meta["height"])

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

        timestamp = frame_idx / fps_out if fps_out > 0 else frame_idx / 30.0

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

        violations = detector.update(detections, timestamp)
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

    violation_types: dict[str, int] = {}
    for violation in all_violations:
        vtype = str(violation.get("type", "UNKNOWN"))
        violation_types[vtype] = violation_types.get(vtype, 0) + 1

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
        "illegal_stopping": {
            "frames_processed": frame_idx,
            "total_violations": len(all_violations),
            "violation_types": violation_types,
            "zones_configured": len(detector.zones),
            "stationary_time_sec": stationary_time,
            "movement_threshold_px": movement_threshold,
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
            "runtime_fps_latency": "Computed during this illegal-stopping metric run",
            "params_map_f1_precision_recall": "outputs/paper_metrics.json",
        },
    }

    os.makedirs(Path(output_report).parent, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate illegal-stopping metrics and duration validation report")
    parser.add_argument("--input", default="samples/videos/Sample-Main.mp4", help="Input video path")
    parser.add_argument(
        "--output-video",
        default="outputs/output/illegal_stopping_sample_main_complete.mp4",
        help="Annotated output video path",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/output/illegal_stopping_sample_main_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--movement-threshold", type=float, default=5.0, help="Movement threshold in pixels")
    parser.add_argument("--stationary-time", type=float, default=3.0, help="Seconds to treat as stopped")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests")
    args = parser.parse_args()

    report = run_illegal_stopping_metric(
        input_video=args.input,
        output_video=args.output_video,
        output_report=args.output_report,
        confidence=args.confidence,
        movement_threshold=args.movement_threshold,
        stationary_time=args.stationary_time,
        max_frames=args.max_frames,
    )

    print("Illegal-stopping metric generation complete")
    print(f"Report: {args.output_report}")
    print(f"Output video: {args.output_video}")
    print(f"Duration match: {report['video']['duration_match']}")
    print(f"Total violations: {report['illegal_stopping']['total_violations']}")


if __name__ == "__main__":
    main()
