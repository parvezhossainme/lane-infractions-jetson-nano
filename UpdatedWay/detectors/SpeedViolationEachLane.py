import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque
from ultralytics import RTDETR


# =========================
# Load RT-DETR
# =========================
model = RTDETR("models/rtdetr-l.pt")
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck


# =========================
# Video
# =========================
video_path = "samples/videos/Sample-Main.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open input video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30.0

output_dir = Path("UpdatedWay/outputs")
output_dir.mkdir(parents=True, exist_ok=True)


def get_unique_output_path(directory: Path, base_name: str = "lane_violation_output", ext: str = ".mp4") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{base_name}_{timestamp}{ext}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = directory / f"{base_name}_{timestamp}_{index:02d}{ext}"
        if not candidate.exists():
            return candidate
        index += 1


output_path = get_unique_output_path(output_dir)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


# =========================
# Motion Alignment Setup
# =========================
ret_ref, reference_frame = cap.read()
if not ret_ref:
    cap.release()
    out.release()
    raise RuntimeError("Unable to read reference frame")

reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
feature_mask = np.zeros_like(reference_gray)
cv2.rectangle(feature_mask, (0, 0), (int(0.24 * width), height - 1), 255, -1)
cv2.rectangle(feature_mask, (int(0.76 * width), 0), (width - 1, height - 1), 255, -1)
cv2.rectangle(feature_mask, (0, 0), (width - 1, int(0.22 * height)), 255, -1)

reference_points = cv2.goodFeaturesToTrack(
    reference_gray,
    maxCorners=600,
    qualityLevel=0.01,
    minDistance=8,
    mask=feature_mask,
)


# =========================
# Pixel to Meter Conversion
# =========================
pixels_per_meter = 100


# =========================
# Vehicle Tracking Memory
# =========================
vehicle_positions = {}
vehicle_tracks = {}
next_vehicle_id = 0
max_match_distance = 30
speed_history_len = 5


# =========================
# Lane Speed Limits
# =========================
lane_limits = {
    1: (5, 20),
    2: (10, 25),
    3: (15, 35),
    4: (20, 45),
}


# =========================
# Lane Colors
# =========================
lane_colors = [
    ((0, 140, 255), "Lane1 5-20"),
    ((255, 170, 0), "Lane2 10-25"),
    ((60, 220, 60), "Lane3 15-35"),
    ((220, 80, 255), "Lane4 20-45"),
]

normal_box_color = (40, 220, 40)
violation_box_color = (30, 30, 230)
text_color = (255, 255, 255)
title_bar_color = (20, 20, 20)
title_text_color = (255, 255, 255)


# =========================
# Fixed Lane Polygons
# =========================
top_y = int(0.05 * height)
bottom_y = height - 1
lane12_shift_px = int(0.015 * width)

lane4 = np.array(
    [[(int(-0.12 * width), bottom_y), (int(0.2 * width), bottom_y), (int(0.41 * width), top_y), (int(0.385 * width), top_y)]],
    np.int32,
)
lane3 = np.array(
    [[(int(0.2 * width), bottom_y), (int(0.53 * width), bottom_y), (int(0.446 * width), top_y), (int(0.41 * width), top_y)]],
    np.int32,
)
lane2 = np.array(
    [[
        (int(0.53 * width) + lane12_shift_px, bottom_y),
        (int(0.81 * width) + lane12_shift_px, bottom_y),
        (int(0.485 * width) + lane12_shift_px, top_y),
        (int(0.446 * width) + lane12_shift_px, top_y),
    ]],
    np.int32,
)
lane1 = np.array(
    [[
        (int(0.81 * width) + lane12_shift_px, bottom_y),
        (min(width - 1, int(width) + lane12_shift_px), bottom_y),
        (int(0.53 * width) + lane12_shift_px, top_y),
        (int(0.485 * width) + lane12_shift_px, top_y),
    ]],
    np.int32,
)
base_lanes = [lane1, lane2, lane3, lane4]

lane_mask_template = np.zeros((height, width), dtype=np.uint8)
lane_overlay_template = np.zeros((height, width, 3), dtype=np.uint8)

for i, lane in enumerate(base_lanes):
    color, label = lane_colors[i]
    cv2.fillPoly(lane_mask_template, lane, 255)
    cv2.polylines(lane_overlay_template, lane, True, color, 4)

    lx = int(np.mean(lane[0][:, 0]))
    ly = int(np.mean(lane[0][:, 1]))
    cv2.putText(
        lane_overlay_template,
        label,
        (lx - 60, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        6,
    )
    cv2.putText(
        lane_overlay_template,
        label,
        (lx - 60, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
    )


# =========================
# Frame Loop
# =========================
written_frames = 0
aligned_frames = 0
total_processed_frames = 0


def transform_lane_artifacts(frame_in):
    if reference_points is None or len(reference_points) < 10:
        return lane_overlay_template, lane_mask_template, base_lanes, False

    gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        gray,
        reference_points,
        None,
    )

    if tracked_points is None or status is None:
        return lane_overlay_template, lane_mask_template, base_lanes, False

    good_ref = reference_points[status.flatten() == 1]
    good_cur = tracked_points[status.flatten() == 1]
    if len(good_ref) < 10 or len(good_cur) < 10:
        return lane_overlay_template, lane_mask_template, base_lanes, False

    transform, _ = cv2.estimateAffinePartial2D(
        good_ref,
        good_cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if transform is None:
        return lane_overlay_template, lane_mask_template, base_lanes, False

    warped_overlay = cv2.warpAffine(
        lane_overlay_template,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        lane_mask_template,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_lanes = []
    for lane in base_lanes:
        lane_f = lane.astype(np.float32)
        warped_lane = cv2.transform(lane_f, transform)
        warped_lane = np.clip(warped_lane, [0, 0], [width - 1, height - 1]).astype(np.int32)
        warped_lanes.append(warped_lane)

    return warped_overlay, warped_mask, warped_lanes, True


def process_frame(frame_in):
    global next_vehicle_id

    warped_overlay, warped_mask, lanes_for_frame, aligned = transform_lane_artifacts(frame_in)
    overlay = cv2.add(frame_in, warped_overlay)
    masked_frame = cv2.bitwise_and(frame_in, frame_in, mask=warped_mask)

    cv2.rectangle(overlay, (0, 0), (width - 1, 58), title_bar_color, -1)
    cv2.putText(
        overlay,
        "Speed Violation Detection",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        title_text_color,
        2,
    )

    # =========================
    # RT-DETR Detection
    # =========================
    results = model(masked_frame, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append((x1, y1, x2, y2, cx, cy))

    updated_positions = {}

    # =========================
    # Match Vehicles Between Frames
    # =========================
    for det in detections:
        x1, y1, x2, y2, cx, cy = det

        matched_id = None
        min_dist = 999999.0

        for vid, (px, py) in vehicle_positions.items():
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < min_dist and dist < max_match_distance:
                min_dist = dist
                matched_id = vid

        if matched_id is None:
            matched_id = next_vehicle_id
            next_vehicle_id += 1

        track = vehicle_tracks.setdefault(matched_id, deque(maxlen=speed_history_len))
        track.append((cx, cy))

        if len(track) >= 2:
            prev_x, prev_y = track[0]
            elapsed_seconds = (len(track) - 1) / fps
        else:
            prev_x, prev_y = cx, cy
            elapsed_seconds = 0.0

        # =========================
        # Speed Calculation
        # =========================
        distance_pixels = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
        distance_meters = distance_pixels / pixels_per_meter
        speed = (distance_meters / elapsed_seconds) * 3.6 if elapsed_seconds > 0 else 0.0

        updated_positions[matched_id] = (cx, cy)

        # =========================
        # Lane Detection
        # =========================
        lane_id = None
        for i, lane in enumerate(lanes_for_frame):
            if cv2.pointPolygonTest(lane, (cx, cy), False) >= 0:
                lane_id = i + 1
                break

        # =========================
        # Violation Detection
        # =========================
        violator = False
        if speed < 1:
            violator = True

        if lane_id is not None:
            min_s, max_s = lane_limits[lane_id]
            if speed < min_s or speed > max_s:
                violator = True
        else:
            violator = True

        # =========================
        # Draw Bounding Box
        # =========================
        color = violation_box_color if violator else normal_box_color
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
        cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), -1)

        text = f"{speed:.1f} km/h"
        if violator:
            text += " VIOLATION"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_y = max(y1 - 12, th + 10)
        cv2.rectangle(
            overlay,
            (x1, text_y - th - baseline - 6),
            (x1 + tw + 8, text_y + 4),
            color,
            -1,
        )

        cv2.putText(
            overlay,
            text,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )

    return overlay, updated_positions, aligned


# Process reference frame first so frame counts remain aligned with input.
overlay_ref, updated_positions_ref, _ = process_frame(reference_frame)
vehicle_positions = updated_positions_ref
out.write(overlay_ref)
written_frames += 1
total_processed_frames += 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay, updated_positions, aligned = process_frame(frame)
    if aligned:
        aligned_frames += 1

    vehicle_positions = updated_positions
    out.write(overlay)
    written_frames += 1
    total_processed_frames += 1


cap.release()
out.release()

print(f"Output saved as {output_path}")
print(f"Input frames: {total_frames}, written frames: {written_frames}, fps: {fps:.2f}")
print(f"Alignment successful on {aligned_frames}/{max(1, total_processed_frames - 1)} non-reference frames")
