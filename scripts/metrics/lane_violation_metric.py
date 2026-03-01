#!/usr/bin/env python3
"""
Lane violation metric pipeline for Sample-Main (or any video).

Generates:
- Annotated lane-violation output video
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
import numpy as np
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


def detect_lane_lines_and_boundaries(frame: np.ndarray) -> tuple[np.ndarray, list[float]]:
    from detectors.lane_detection import fixed_roi_for_this_image, preprocess_lane

    roi_img, _ = fixed_roi_for_this_image(frame)
    lane_binary = preprocess_lane(roi_img)

    edges = cv2.Canny(lane_binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)

    lane_img = np.zeros_like(frame)
    x_positions: list[int] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)

            if 60 < angle < 120 and length >= 100:
                cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                x_positions.extend([x1, x2])

    boundaries: list[float] = []
    if x_positions:
        x_positions = sorted(x_positions)
        boundaries.append(float(x_positions[0]))
        min_gap = max(70, frame.shape[1] // 20)
        for x in x_positions[1:]:
            if x - boundaries[-1] >= min_gap:
                boundaries.append(float(x))

    # Fallback boundaries for occluded/noisy frames
    if len(boundaries) < 2:
        width = frame.shape[1]
        boundaries = [width * 0.38, width * 0.52, width * 0.66]

    return lane_img, boundaries


def run_lane_violation_metric(
    input_video: str,
    output_video: str,
    output_report: str,
    confidence: float,
    sequence_length: int,
    deviation_threshold: float,
    max_frames: int | None,
) -> dict:
    from violations.lane_violation import LaneViolationDetector

    input_meta = get_video_meta(input_video)
    if not input_meta["opened"]:
        raise RuntimeError(f"Could not open input video: {input_video}")

    model = RTDETR("models/rtdetr-l.pt")
    detector = LaneViolationDetector(
        num_lanes=3,
        sequence_length=sequence_length,
        deviation_threshold=deviation_threshold,
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
    active_trajectory_counts: list[int] = []
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        timestamp = frame_idx / fps_out if fps_out > 0 else frame_idx / 30.0

        t0 = time.perf_counter()

        lane_img, lane_boundaries = detect_lane_lines_and_boundaries(frame)
        detector.set_lane_boundaries(lane_boundaries)

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

        violations = detector.update(detections, frame_idx, timestamp)
        all_violations.extend(violations)
        active_trajectory_counts.append(len(detector.trajectories))

        vis_frame = detector.visualize(frame, lane_img, violations)
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

    improper_lane_changes = sum(1 for v in all_violations if v.get("type") == "IMPROPER_LANE_CHANGE")
    lane_deviations = sum(1 for v in all_violations if v.get("type") == "LANE_DEVIATION")

    fps_runtime = (frame_idx / elapsed) if elapsed > 0 else 0.0
    lat_mean = statistics.mean(latencies_ms) if latencies_ms else 0.0
    lat_median = statistics.median(latencies_ms) if latencies_ms else 0.0
    avg_active_trajectories = (
        statistics.mean(active_trajectory_counts) if active_trajectory_counts else 0.0
    )

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
        "lane_violation": {
            "frames_processed": frame_idx,
            "total_violations": len(all_violations),
            "improper_lane_change_violations": improper_lane_changes,
            "lane_deviation_violations": lane_deviations,
            "sequence_length": sequence_length,
            "deviation_threshold_px": deviation_threshold,
            "avg_active_trajectories": avg_active_trajectories,
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
            "runtime_fps_latency": "Computed during this lane-violation metric run",
            "params_map_f1_precision_recall": "outputs/paper_metrics.json",
        },
    }

    os.makedirs(Path(output_report).parent, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lane-violation metrics and duration validation report")
    parser.add_argument("--input", default="samples/videos/Sample-Main.mp4", help="Input video path")
    parser.add_argument(
        "--output-video",
        default="outputs/output/lane_violation_sample_main_complete.mp4",
        help="Annotated output video path",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/output/lane_violation_sample_main_report.json",
        help="Output JSON report path",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--sequence-length", type=int, default=10, help="LSTM sequence length")
    parser.add_argument("--deviation-threshold", type=float, default=50.0, help="Deviation threshold (pixels)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests")
    args = parser.parse_args()

    report = run_lane_violation_metric(
        input_video=args.input,
        output_video=args.output_video,
        output_report=args.output_report,
        confidence=args.confidence,
        sequence_length=args.sequence_length,
        deviation_threshold=args.deviation_threshold,
        max_frames=args.max_frames,
    )

    print("Lane-violation metric generation complete")
    print(f"Report: {args.output_report}")
    print(f"Output video: {args.output_video}")
    print(f"Duration match: {report['video']['duration_match']}")
    print(f"Total violations: {report['lane_violation']['total_violations']}")


if __name__ == "__main__":
    main()
