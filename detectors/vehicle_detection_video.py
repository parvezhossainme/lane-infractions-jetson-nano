#!/usr/bin/env python3
"""
Vehicle Detection for Videos using Ultralytics RT-DETR.

This script is separate from image detection code and provides:
1) Full video vehicle detection with saved annotated output video
2) Frame extraction + detection checks saved as annotated images
"""

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import RTDETR


VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_model(model_path: str = "rtdetr-l.pt"):
    print("Loading RT-DETR model...")
    print("(First run may download model weights)")
    model = RTDETR(model_path)
    print("✅ Model loaded")
    return model


def vehicle_detections_from_result(result):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue

        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].cpu().numpy().astype(int),
            }
        )
    return detections


def draw_detections(frame, detections, frame_idx=None):
    output_frame = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(output_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    status = f"Vehicles: {len(detections)}"
    if frame_idx is not None:
        status = f"Frame: {frame_idx} | {status}"
    cv2.putText(output_frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return output_frame


def extract_and_check_frames(
    video_path: str,
    model,
    output_dir: str,
    confidence: float,
    frame_interval: int,
    max_check_frames: int,
):
    """Extract interval-based frames from video, run detection, and save annotated frame checks."""
    frames_dir = os.path.join(output_dir, "frame_checks")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video for frame checks: {video_path}")
        return []

    frame_idx = 0
    checked = 0
    check_summary = []

    print(f"\nExtracting and checking frames every {frame_interval} frames...")
    while checked < max_check_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, conf=confidence, verbose=False)
            detections = vehicle_detections_from_result(results[0])
            annotated = draw_detections(frame, detections, frame_idx=frame_idx)

            frame_file = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_file, annotated)

            check_summary.append(
                {
                    "frame_index": frame_idx,
                    "vehicles_detected": len(detections),
                    "saved_frame": frame_file,
                }
            )
            print(f"  ✅ Checked frame {frame_idx}: {len(detections)} vehicles")
            checked += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Frame checks saved in: {frames_dir}")
    return check_summary


def process_video(
    video_path: str,
    model,
    output_dir: str,
    confidence: float,
    max_frames: int | None = None,
):
    """Run full video detection and save annotated output video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = Path(video_path).stem
    output_video = os.path.join(output_dir, f"{stem}_vehicle_detection.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"❌ Error: Could not create output file: {output_video}")
        return None

    print("\nProcessing full video detection...")
    print(f"Input: {video_path}")
    print(f"Output: {output_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    processed = 0
    peak_vehicles = 0
    cumulative_vehicles = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=confidence, verbose=False)
        detections = vehicle_detections_from_result(results[0])
        output_frame = draw_detections(frame, detections, frame_idx=processed)
        writer.write(output_frame)

        vehicle_count = len(detections)
        peak_vehicles = max(peak_vehicles, vehicle_count)
        cumulative_vehicles += vehicle_count

        processed += 1
        if processed % 50 == 0:
            progress = (processed / total_frames * 100) if total_frames > 0 else 0
            print(f"  Processed {processed} frames ({progress:.1f}%)")

        if max_frames is not None and processed >= max_frames:
            print(f"  Stopped early at {processed} frames (max_frames={max_frames})")
            break

    cap.release()
    writer.release()

    avg_vehicles = (cumulative_vehicles / processed) if processed > 0 else 0.0
    stats = {
        "input_video": video_path,
        "output_video": output_video,
        "frames_processed": processed,
        "peak_vehicles_in_frame": peak_vehicles,
        "average_vehicles_per_frame": round(avg_vehicles, 3),
        "confidence_threshold": confidence,
    }

    print("✅ Video output saved")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection for videos + frame checks")
    parser.add_argument("--input", default="samples/videos/Sample-Main.mp4", help="Input video path")
    parser.add_argument("--output-dir", default="outputs/output", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--frame-interval", type=int, default=60, help="Extract/check every Nth frame")
    parser.add_argument("--max-check-frames", type=int, default=10, help="Number of extracted frames to check")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for processed video frames")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Video not found: {args.input}")
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Vehicle Detection (Video) using RT-DETR")
    print("=" * 70)

    model = load_model("rtdetr-l.pt")

    frame_checks = extract_and_check_frames(
        video_path=args.input,
        model=model,
        output_dir=args.output_dir,
        confidence=args.confidence,
        frame_interval=max(1, args.frame_interval),
        max_check_frames=max(1, args.max_check_frames),
    )

    video_stats = process_video(
        video_path=args.input,
        model=model,
        output_dir=args.output_dir,
        confidence=args.confidence,
        max_frames=args.max_frames,
    )

    if video_stats is None:
        raise SystemExit(1)

    report = {
        "video_stats": video_stats,
        "frame_checks": frame_checks,
    }
    report_path = os.path.join(args.output_dir, f"{Path(args.input).stem}_vehicle_detection_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Report saved: {report_path}")
    print("🎉 Video vehicle detection complete")


if __name__ == "__main__":
    main()
