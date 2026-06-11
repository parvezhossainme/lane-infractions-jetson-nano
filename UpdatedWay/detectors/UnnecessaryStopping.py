from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR

# =========================================================
# LOAD MODEL
# =========================================================

model = RTDETR("models/rtdetr-l.pt")

# COCO vehicle classes
vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# =========================================================
# VIDEO INPUT
# =========================================================

video_path = "samples/videos/Sample-Main.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =========================================================
# OUTPUT VIDEO
# =========================================================

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = output_dir / f"stable_lane_tracking_{timestamp}.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

# =========================================================
# STATIC LANE SETUP
# =========================================================

lane_colors = [
    ((0, 255, 255), "Lane1"),
    ((255, 0, 255), "Lane2"),
    ((0, 255, 0), "Lane3"),
    ((255, 150, 0), "Lane4"),
]

top_y = int(0.05 * height)
bottom_y = height - 1

lane4 = np.array(
    [[
        (int(-0.12 * width), bottom_y),
        (int(0.2 * width), bottom_y),
        (int(0.41 * width), top_y),
        (int(0.385 * width), top_y),
    ]],
    np.int32,
)

lane3 = np.array(
    [[
        (int(0.2 * width), bottom_y),
        (int(0.53 * width), bottom_y),
        (int(0.446 * width), top_y),
        (int(0.41 * width), top_y),
    ]],
    np.int32,
)

lane2 = np.array(
    [[
        (int(0.53 * width), bottom_y),
        (int(0.81 * width), bottom_y),
        (int(0.485 * width), top_y),
        (int(0.446 * width), top_y),
    ]],
    np.int32,
)

lane1 = np.array(
    [[
        (int(0.81 * width), bottom_y),
        (width - 1, bottom_y),
        (int(0.53 * width), top_y),
        (int(0.485 * width), top_y),
    ]],
    np.int32,
)

lanes = [lane1, lane2, lane3, lane4]

# =========================================================
# LANE SMOOTHING
# =========================================================

lane_history = deque(maxlen=20)

def smooth_lane_points(lane_history):

    if len(lane_history) == 0:
        return None

    avg_lanes = []

    num_lanes = len(lane_history[0])

    for lane_idx in range(num_lanes):

        lane_points = []

        for history_item in lane_history:
            lane_points.append(history_item[lane_idx][0])

        lane_points = np.array(lane_points)

        avg_points = np.mean(
            lane_points,
            axis=0
        )

        avg_lanes.append(
            np.array([avg_points.astype(np.int32)])
        )

    return avg_lanes

# =========================================================
# TRACKING CONFIGURATION
# =========================================================

movement_threshold_px = 20.0
stationary_time_seconds = 4.0
history_length = 50
match_distance_px = 220
minimum_speed_px_per_sec = 6.0

vehicle_tracks = {}
vehicle_stop_start = {}
previous_lane_ids = {}

next_vehicle_id = 0

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_lane(point, vehicle_id=None):

    detected_lane = None

    for i, lane in enumerate(lanes):

        if cv2.pointPolygonTest(
            lane,
            point,
            False
        ) >= 0:

            detected_lane = i + 1
            break

    if vehicle_id is not None:

        previous_lane = previous_lane_ids.get(vehicle_id)

        if previous_lane is not None and detected_lane is None:
            return previous_lane

        if detected_lane is not None:
            previous_lane_ids[vehicle_id] = detected_lane

    return detected_lane


def match_vehicle(center, bbox):

    global vehicle_tracks

    best_id = None
    best_score = float("inf")

    cx, cy = center
    x1, y1, x2, y2 = bbox

    current_area = (x2 - x1) * (y2 - y1)

    for veh_id, track in vehicle_tracks.items():

        if len(track) == 0:
            continue

        prev_x, prev_y, _, prev_bbox = track[-1]

        px1, py1, px2, py2 = prev_bbox

        prev_area = (px2 - px1) * (py2 - py1)

        dist = np.hypot(
            cx - prev_x,
            cy - prev_y
        )

        area_diff = abs(
            current_area - prev_area
        )

        score = dist + (0.002 * area_diff)

        if score < best_score and dist < match_distance_px:
            best_score = score
            best_id = veh_id

    return best_id


def estimate_speed(track):

    if len(track) < 2:
        return 0.0

    x1, y1, t1, _ = track[0]
    x2, y2, t2, _ = track[-1]

    elapsed = max(t2 - t1, 0.001)

    distance = max(
        0,
        np.hypot(x2 - x1, y2 - y1) - 5
    )

    return distance / elapsed


def is_stationary(track):

    if len(track) < 10:
        return False

    xs = [p[0] for p in track]
    ys = [p[1] for p in track]

    x_std = np.std(xs)
    y_std = np.std(ys)

    total_variation = x_std + y_std

    return total_variation < movement_threshold_px

# =========================================================
# MAIN PROCESSING LOOP
# =========================================================

frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = frame_index / fps

    overlay = frame.copy()

    # =====================================================
    # SMOOTH LANE HISTORY
    # =====================================================

    current_lanes = [lane1, lane2, lane3, lane4]

    lane_history.append(current_lanes)

    smoothed_lanes = smooth_lane_points(
        lane_history
    )

    if smoothed_lanes is not None:
        lanes = smoothed_lanes

    # =====================================================
    # DRAW STABLE LANES
    # =====================================================

    for i, lane in enumerate(lanes):

        color, label = lane_colors[i]

        cv2.polylines(
            overlay,
            lane,
            True,
            color,
            3
        )

        center_x = int(
            np.mean(lane[0][:, 0])
        )

        center_y = int(
            np.mean(lane[0][:, 1])
        )

        cv2.putText(
            overlay,
            label,
            (center_x - 40, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # =====================================================
    # RT-DETR DETECTION
    # =====================================================

    results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    detections = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            class_id = int(cls)

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append(
                (x1, y1, x2, y2, cx, cy)
            )

    # =====================================================
    # PROCESS VEHICLES
    # =====================================================

    for x1, y1, x2, y2, cx, cy in detections:

        matched_id = match_vehicle(
            (cx, cy),
            (x1, y1, x2, y2)
        )

        if matched_id is None:

            matched_id = next_vehicle_id
            next_vehicle_id += 1

            vehicle_tracks[matched_id] = deque(
                maxlen=history_length
            )

            vehicle_stop_start[matched_id] = current_time

        lane_id = find_lane(
            (cx, cy),
            matched_id
        )

        track = vehicle_tracks[matched_id]

        track.append(
            (
                cx,
                cy,
                current_time,
                (x1, y1, x2, y2)
            )
        )

        speed = estimate_speed(track)

        if speed < 4:
            speed = 0

        stationary = is_stationary(track)

        # =================================================
        # STOP TIMER
        # =================================================

        if stationary:
            stop_duration = (
                current_time -
                vehicle_stop_start[matched_id]
            )
        else:
            vehicle_stop_start[matched_id] = current_time
            stop_duration = 0

        # =================================================
        # VEHICLE STATUS
        # =================================================

        if stationary:

            color = (0, 0, 255)

            status = (
                f"STOPPED VEHICLE "
                f"{stop_duration:.1f}s"
            )

        else:

            color = (0, 255, 0)

            status = (
                f"MOVING "
                f"{speed:.1f}px/s"
            )

        # =================================================
        # DRAW VEHICLE
        # =================================================

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"{status} | "
            f"ID {matched_id} | "
            f"Lane {lane_id}"
        )

        cv2.putText(
            overlay,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.circle(
            overlay,
            (cx, cy),
            5,
            (255, 255, 255),
            -1
        )

    # =====================================================
    # SAVE FRAME
    # =====================================================

    out.write(overlay)

    frame_index += 1

# =========================================================
# CLEANUP
# =========================================================

cap.release()
out.release()

print(
    f"\nSaved output video to:\n{output_path}"
)


# from __future__ import annotations

# from collections import deque
# from datetime import datetime
# from pathlib import Path

# import cv2
# import numpy as np
# from ultralytics import RTDETR

# # =========================================================
# # LOAD MODEL
# # =========================================================

# model = RTDETR("models/rtdetr-l.pt")

# # COCO vehicle classes
# vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# # =========================================================
# # VIDEO INPUT
# # =========================================================

# video_path = "samples/videos/Sample-Main.mp4"

# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     raise RuntimeError(f"Cannot open video: {video_path}")

# fps = cap.get(cv2.CAP_PROP_FPS)

# if fps <= 0:
#     fps = 30.0

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # =========================================================
# # OUTPUT VIDEO
# # =========================================================

# output_dir = Path("outputs")
# output_dir.mkdir(exist_ok=True)

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# output_path = output_dir / f"stopped_vehicle_detection_{timestamp}.mp4"

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# out = cv2.VideoWriter(
#     str(output_path),
#     fourcc,
#     fps,
#     (width, height)
# )

# # =========================================================
# # STATIC LANE SETUP
# # =========================================================

# lane_colors = [
#     ((0, 255, 255), "Lane1"),
#     ((255, 0, 255), "Lane2"),
#     ((0, 255, 0), "Lane3"),
#     ((255, 150, 0), "Lane4"),
# ]

# top_y = int(0.05 * height)
# bottom_y = height - 1

# lane4 = np.array(
#     [[
#         (int(-0.12 * width), bottom_y),
#         (int(0.2 * width), bottom_y),
#         (int(0.41 * width), top_y),
#         (int(0.385 * width), top_y),
#     ]],
#     np.int32,
# )

# lane3 = np.array(
#     [[
#         (int(0.2 * width), bottom_y),
#         (int(0.53 * width), bottom_y),
#         (int(0.446 * width), top_y),
#         (int(0.41 * width), top_y),
#     ]],
#     np.int32,
# )

# lane2 = np.array(
#     [[
#         (int(0.53 * width), bottom_y),
#         (int(0.81 * width), bottom_y),
#         (int(0.485 * width), top_y),
#         (int(0.446 * width), top_y),
#     ]],
#     np.int32,
# )

# lane1 = np.array(
#     [[
#         (int(0.81 * width), bottom_y),
#         (width - 1, bottom_y),
#         (int(0.53 * width), top_y),
#         (int(0.485 * width), top_y),
#     ]],
#     np.int32,
# )

# lanes = [lane1, lane2, lane3, lane4]

# # =========================================================
# # TRACKING CONFIGURATION
# # =========================================================

# movement_threshold_px = 20.0
# stationary_time_seconds = 4.0
# history_length = 30
# match_distance_px = 140
# minimum_speed_px_per_sec = 6.0

# vehicle_tracks = {}
# vehicle_stop_start = {}
# previous_lane_ids = {}

# next_vehicle_id = 0

# # =========================================================
# # HELPER FUNCTIONS
# # =========================================================

# def find_lane(point, vehicle_id=None):

#     detected_lane = None

#     for i, lane in enumerate(lanes):

#         if cv2.pointPolygonTest(lane, point, False) >= 0:
#             detected_lane = i + 1
#             break

#     if vehicle_id is not None:

#         previous_lane = previous_lane_ids.get(vehicle_id)

#         if previous_lane is not None and detected_lane is None:
#             return previous_lane

#         if detected_lane is not None:
#             previous_lane_ids[vehicle_id] = detected_lane

#     return detected_lane


# def match_vehicle(center):

#     best_id = None
#     best_dist = float("inf")

#     for veh_id, track in vehicle_tracks.items():

#         if len(track) == 0:
#             continue

#         prev_x, prev_y, _ = track[-1]

#         dist = np.hypot(center[0] - prev_x, center[1] - prev_y)

#         if dist < best_dist and dist < match_distance_px:
#             best_dist = dist
#             best_id = veh_id

#     return best_id


# def estimate_speed(track):

#     if len(track) < 2:
#         return 0.0

#     x1, y1, t1 = track[0]
#     x2, y2, t2 = track[-1]

#     elapsed = max(t2 - t1, 0.001)

#     distance = max(
#         0,
#         np.hypot(x2 - x1, y2 - y1) - 5
#     )

#     return distance / elapsed


# def is_stationary(track):

#     if len(track) < 10:
#         return False

#     xs = [p[0] for p in track]
#     ys = [p[1] for p in track]

#     x_std = np.std(xs)
#     y_std = np.std(ys)

#     total_variation = x_std + y_std

#     return total_variation < movement_threshold_px


# # =========================================================
# # MAIN PROCESSING LOOP
# # =========================================================

# frame_index = 0

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     current_time = frame_index / fps

#     overlay = frame.copy()

#     # =====================================================
#     # DRAW LANES
#     # =====================================================

#     for i, lane in enumerate(lanes):

#         color, label = lane_colors[i]

#         cv2.polylines(
#             overlay,
#             lane,
#             True,
#             color,
#             3
#         )

#         center_x = int(np.mean(lane[0][:, 0]))
#         center_y = int(np.mean(lane[0][:, 1]))

#         cv2.putText(
#             overlay,
#             label,
#             (center_x - 40, center_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             color,
#             2
#         )

#     # =====================================================
#     # RT-DETR DETECTION
#     # =====================================================

#     results = model.predict(
#         source=frame,
#         conf=0.4,
#         verbose=False
#     )

#     detections = []

#     for result in results:

#         boxes = result.boxes.xyxy.cpu().numpy()
#         classes = result.boxes.cls.cpu().numpy()

#         for box, cls in zip(boxes, classes):

#             class_id = int(cls)

#             if class_id not in vehicle_classes:
#                 continue

#             x1, y1, x2, y2 = map(int, box)

#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             detections.append(
#                 (x1, y1, x2, y2, cx, cy)
#             )

#     # =====================================================
#     # PROCESS VEHICLES
#     # =====================================================

#     for x1, y1, x2, y2, cx, cy in detections:

#         matched_id = match_vehicle((cx, cy))

#         if matched_id is None:

#             matched_id = next_vehicle_id
#             next_vehicle_id += 1

#             vehicle_tracks[matched_id] = deque(
#                 maxlen=history_length
#             )

#             vehicle_stop_start[matched_id] = current_time

#         lane_id = find_lane((cx, cy), matched_id)

#         track = vehicle_tracks[matched_id]

#         track.append((cx, cy, current_time))

#         speed = estimate_speed(track)

#         if speed < 4:
#             speed = 0

#         stationary = is_stationary(track)

#         # =================================================
#         # STOP DURATION
#         # =================================================

#         if stationary:
#             stop_duration = current_time - vehicle_stop_start[matched_id]
#         else:
#             vehicle_stop_start[matched_id] = current_time
#             stop_duration = 0

#         # =================================================
#         # DRAW STATUS
#         # =================================================

#         if stationary:

#             # RED BOX FOR STOPPED VEHICLES
#             color = (0, 0, 255)

#             status = f"STOPPED VEHICLE {stop_duration:.1f}s"

#         else:

#             # GREEN BOX FOR MOVING VEHICLES
#             color = (0, 255, 0)

#             status = f"MOVING {speed:.1f}px/s"

#         # =================================================
#         # DRAW BOUNDING BOX
#         # =================================================

#         cv2.rectangle(
#             overlay,
#             (x1, y1),
#             (x2, y2),
#             color,
#             3
#         )

#         label = f"{status} | ID {matched_id} | Lane {lane_id}"

#         cv2.putText(
#             overlay,
#             label,
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             color,
#             2
#         )

#         cv2.circle(
#             overlay,
#             (cx, cy),
#             5,
#             (255, 255, 255),
#             -1
#         )

#     # =====================================================
#     # SAVE FRAME
#     # =====================================================

#     out.write(overlay)

#     frame_index += 1

# # =========================================================
# # CLEANUP
# # =========================================================

# cap.release()
# out.release()

# print(f"\nSaved output video to:\n{output_path}")


# """Lane-first unnecessary/illegal stopping detection.

# The pipeline is:
# 1. Detect and overlay lanes.
# 2. Detect vehicles with RT-DETR.
# 3. Assign each vehicle to a lane.
# 4. Track vehicle motion over a short history.
# 5. Flag unnecessary stopping / illegal stopping when a vehicle remains
#    nearly stationary inside a lane beyond the configured threshold.
# """

# from __future__ import annotations

# from collections import deque
# from datetime import datetime
# from pathlib import Path

# import cv2
# import numpy as np
# from ultralytics import RTDETR


# # =========================
# # Model and input source
# # =========================
# model = RTDETR("models/rtdetr-l.pt")
# vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
# video_path = "samples/videos/Sample-Main.mp4"

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise RuntimeError(f"Unable to open input video: {video_path}")

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# if fps <= 0:
#     fps = 30.0

# output_dir = Path("UpdatedWay/outputs")
# output_dir.mkdir(parents=True, exist_ok=True)


# def get_unique_output_path(directory: Path, base_name: str = "illegal_stopping_output", ext: str = ".mp4") -> Path:
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     candidate = directory / f"{base_name}_{timestamp}{ext}"
#     if not candidate.exists():
#         return candidate

#     index = 1
#     while True:
#         candidate = directory / f"{base_name}_{timestamp}_{index:02d}{ext}"
#         if not candidate.exists():
#             return candidate
#         index += 1


# def create_video_writer(directory: Path, fps_value: float, frame_size: tuple[int, int]):
#     candidates = [
#         (".mp4", "mp4v"),
#         (".avi", "XVID"),
#         (".avi", "MJPG"),
#     ]

#     for ext, codec in candidates:
#         path = get_unique_output_path(directory, ext=ext)
#         writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps_value, frame_size)
#         if writer.isOpened():
#             return writer, path, codec
#         writer.release()

#     raise RuntimeError("Failed to open VideoWriter with all codec/container fallbacks")


# out, output_path, selected_codec = create_video_writer(output_dir, fps, (width, height))


# # =========================
# # Lane overlay setup
# # =========================
# ret_ref, reference_frame = cap.read()
# if not ret_ref:
#     cap.release()
#     out.release()
#     raise RuntimeError("Unable to read reference frame")

# reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
# feature_mask = np.zeros_like(reference_gray)
# cv2.rectangle(feature_mask, (0, 0), (int(0.24 * width), height - 1), 255, -1)
# cv2.rectangle(feature_mask, (int(0.76 * width), 0), (width - 1, height - 1), 255, -1)
# cv2.rectangle(feature_mask, (0, 0), (width - 1, int(0.22 * height)), 255, -1)

# reference_points = cv2.goodFeaturesToTrack(
#     reference_gray,
#     maxCorners=600,
#     qualityLevel=0.01,
#     minDistance=8,
#     mask=feature_mask,
# )

# lane_colors = [
#     ((0, 140, 255), "Lane1"),
#     ((255, 170, 0), "Lane2"),
#     ((60, 220, 60), "Lane3"),
#     ((220, 80, 255), "Lane4"),
# ]

# # Same static lane geometry used in the other detectors.
# top_y = int(0.05 * height)
# bottom_y = height - 1
# lane12_shift_px = int(0.015 * width)

# lane4 = np.array(
#     [[(int(-0.12 * width), bottom_y), (int(0.2 * width), bottom_y), (int(0.41 * width), top_y), (int(0.385 * width), top_y)]],
#     np.int32,
# )
# lane3 = np.array(
#     [[(int(0.2 * width), bottom_y), (int(0.53 * width), bottom_y), (int(0.446 * width), top_y), (int(0.41 * width), top_y)]],
#     np.int32,
# )
# lane2 = np.array(
#     [[
#         (int(0.53 * width) + lane12_shift_px, bottom_y),
#         (int(0.81 * width) + lane12_shift_px, bottom_y),
#         (int(0.485 * width) + lane12_shift_px, top_y),
#         (int(0.446 * width) + lane12_shift_px, top_y),
#     ]],
#     np.int32,
# )
# lane1 = np.array(
#     [[
#         (int(0.81 * width) + lane12_shift_px, bottom_y),
#         (min(width - 1, int(width) + lane12_shift_px), bottom_y),
#         (int(0.53 * width) + lane12_shift_px, top_y),
#         (int(0.485 * width) + lane12_shift_px, top_y),
#     ]],
#     np.int32,
# )
# base_lanes = [lane1, lane2, lane3, lane4]

# lane_overlay_template = np.zeros((height, width, 3), dtype=np.uint8)
# lane_mask_template = np.zeros((height, width), dtype=np.uint8)
# for i, lane in enumerate(base_lanes):
#     color, label = lane_colors[i]
#     cv2.fillPoly(lane_mask_template, lane, 255)
#     cv2.polylines(lane_overlay_template, lane, True, color, 4)

#     center_x = int(np.mean(lane[0][:, 0]))
#     center_y = int(np.mean(lane[0][:, 1]))
#     cv2.putText(
#         lane_overlay_template,
#         label,
#         (center_x - 50, center_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.1,
#         (0, 0, 0),
#         6,
#     )
#     cv2.putText(
#         lane_overlay_template,
#         label,
#         (center_x - 50, center_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.1,
#         color,
#         3,
#     )


# # =========================
# # Stopping logic
# # =========================
# movement_threshold_px = 5.0
# stationary_time_seconds = 3.0
# history_length = 6
# match_distance_px = 40
# minimum_speed_for_moving_px_per_sec = 8.0
# pixels_per_meter = 100.0

# vehicle_tracks: dict[int, deque[tuple[float, float, float]]] = {}
# vehicle_lane_state: dict[int, int | None] = {}
# vehicle_class_state: dict[int, int] = {}
# vehicle_stop_start: dict[int, float] = {}
# next_vehicle_id = 0

# normal_box_color = (40, 220, 40)
# warning_box_color = (30, 30, 230)
# outside_lane_color = (0, 165, 255)
# title_bar_color = (20, 20, 20)
# title_text_color = (255, 255, 255)
# text_color = (255, 255, 255)


# # =========================
# # Frame helpers
# # =========================
# def transform_lane_artifacts(frame_in: np.ndarray):
#     if reference_points is None or len(reference_points) < 10:
#         return lane_overlay_template, lane_mask_template, base_lanes, False

#     gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
#     tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
#         reference_gray,
#         gray,
#         reference_points,
#         None,
#     )

#     if tracked_points is None or status is None:
#         return lane_overlay_template, lane_mask_template, base_lanes, False

#     good_ref = reference_points[status.flatten() == 1]
#     good_cur = tracked_points[status.flatten() == 1]
#     if len(good_ref) < 10 or len(good_cur) < 10:
#         return lane_overlay_template, lane_mask_template, base_lanes, False

#     transform, _ = cv2.estimateAffinePartial2D(
#         good_ref,
#         good_cur,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=3.0,
#     )
#     if transform is None:
#         return lane_overlay_template, lane_mask_template, base_lanes, False

#     warped_overlay = cv2.warpAffine(
#         lane_overlay_template,
#         transform,
#         (width, height),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0),
#     )
#     warped_mask = cv2.warpAffine(
#         lane_mask_template,
#         transform,
#         (width, height),
#         flags=cv2.INTER_NEAREST,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=0,
#     )

#     warped_lanes = []
#     for lane in base_lanes:
#         warped_lane = cv2.transform(lane.astype(np.float32), transform)
#         warped_lane = np.clip(warped_lane, [0, 0], [width - 1, height - 1]).astype(np.int32)
#         warped_lanes.append(warped_lane)

#     return warped_overlay, warped_mask, warped_lanes, True


# def find_lane_for_point(lanes_for_frame: list[np.ndarray], point: tuple[int, int]) -> int | None:
#     for i, lane in enumerate(lanes_for_frame):
#         if cv2.pointPolygonTest(lane, point, False) >= 0:
#             return i + 1
#     return None


# def match_vehicle(center: tuple[float, float]) -> int | None:
#     matched_id = None
#     min_dist = float("inf")
#     for veh_id, track in vehicle_tracks.items():
#         if not track:
#             continue
#         prev_x, prev_y, _ = track[-1]
#         dist = float(np.hypot(center[0] - prev_x, center[1] - prev_y))
#         if dist < min_dist and dist <= match_distance_px:
#             min_dist = dist
#             matched_id = veh_id
#     return matched_id


# def estimate_speed_from_track(track: deque[tuple[float, float, float]]) -> float:
#     if len(track) < 2:
#         return 0.0

#     first_x, first_y, first_t = track[0]
#     last_x, last_y, last_t = track[-1]
#     elapsed = max(0.0, last_t - first_t)
#     if elapsed <= 0:
#         return 0.0

#     distance_pixels = float(np.hypot(last_x - first_x, last_y - first_y))
#     pixels_per_second = distance_pixels / elapsed
#     return pixels_per_second


# def is_stationary(track: deque[tuple[float, float, float]]) -> bool:
#     if len(track) < 3:
#         return False

#     movements = []
#     for idx in range(1, len(track)):
#         prev_x, prev_y, _ = track[idx - 1]
#         curr_x, curr_y, _ = track[idx]
#         movements.append(float(np.hypot(curr_x - prev_x, curr_y - prev_y)))

#     avg_movement = float(np.mean(movements[-5:])) if movements else 0.0
#     return avg_movement < movement_threshold_px


# # =========================
# # Video processing
# # =========================
# written_frames = 0
# aligned_frames = 0
# total_processed_frames = 0


# def process_frame(frame_in: np.ndarray, current_time: float):
#     global next_vehicle_id

#     warped_overlay, warped_mask, lanes_for_frame, aligned = transform_lane_artifacts(frame_in)
#     overlay = cv2.add(frame_in, warped_overlay)
#     masked_frame = cv2.bitwise_and(frame_in, frame_in, mask=warped_mask)

#     cv2.rectangle(overlay, (0, 0), (width - 1, 58), title_bar_color, -1)
#     cv2.putText(
#         overlay,
#         "Illegal / Unnecessary Stopping Detection",
#         (20, 38),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,
#         title_text_color,
#         2,
#     )

#     results = model(masked_frame, verbose=False)
#     detections: list[tuple[int, int, int, int, int, int, int]] = []

#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()
#         classes = result.boxes.cls.cpu().numpy()
#         for box, cls in zip(boxes, classes):
#             class_id = int(cls)
#             if class_id not in vehicle_classes:
#                 continue
#             x1, y1, x2, y2 = map(int, box)
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)
#             detections.append((x1, y1, x2, y2, cx, cy, class_id))

#     seen_ids: set[int] = set()
#     current_violations = []

#     # Lane first, then stopping status.
#     for x1, y1, x2, y2, cx, cy, class_id in detections:
#         lane_id = find_lane_for_point(lanes_for_frame, (cx, cy))
#         matched_id = match_vehicle((cx, cy))

#         if matched_id is None:
#             matched_id = next_vehicle_id
#             next_vehicle_id += 1
#             vehicle_tracks[matched_id] = deque(maxlen=history_length)
#             vehicle_class_state[matched_id] = class_id
#             vehicle_stop_start[matched_id] = current_time

#         vehicle_class_state[matched_id] = class_id
#         track = vehicle_tracks.setdefault(matched_id, deque(maxlen=history_length))
#         track.append((float(cx), float(cy), current_time))
#         vehicle_lane_state[matched_id] = lane_id
#         seen_ids.add(matched_id)

#         speed_px_per_sec = estimate_speed_from_track(track)
#         stationary = is_stationary(track)

#         if stationary:
#             vehicle_stop_start.setdefault(matched_id, current_time)
#         else:
#             vehicle_stop_start[matched_id] = current_time

#         stop_start_time = vehicle_stop_start.get(matched_id, current_time)
#         stop_duration = max(0.0, current_time - stop_start_time) if stationary else 0.0

#         illegal_stop = False
#         if lane_id is not None:
#             illegal_stop = stationary and stop_duration >= stationary_time_seconds and speed_px_per_sec < minimum_speed_for_moving_px_per_sec
#         else:
#             # If no lane is found, still surface the detection, but keep it separate from lane-based stopping.
#             illegal_stop = False

#         color = warning_box_color if illegal_stop else (outside_lane_color if lane_id is None else normal_box_color)
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
#         cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), -1)

#         lane_text = f"Lane {lane_id}" if lane_id is not None else "No lane"
#         speed_kmh = (speed_px_per_sec / pixels_per_meter) * 3.6
#         speed_text = f"{speed_kmh:.1f} km/h"
#         status_text = "STOPPING" if stationary else "MOVING"
#         text = f"{lane_text} | {speed_text} | {status_text}"
#         if illegal_stop:
#             text += " | ILLEGAL STOP"

#         (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         text_y = max(y1 - 12, th + 10)
#         cv2.rectangle(overlay, (x1, text_y - th - baseline - 6), (x1 + tw + 8, text_y + 4), color, -1)
#         cv2.putText(overlay, text, (x1 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

#         if illegal_stop:
#             current_violations.append(
#                 {
#                     "vehicle_id": matched_id,
#                     "lane_id": lane_id,
#                     "position": (cx, cy),
#                     "duration": stop_duration,
#                     "speed_px_per_sec": speed_px_per_sec,
#                 }
#             )

#     # Remove stale tracks.
#     stale_ids = []
#     for veh_id, track in vehicle_tracks.items():
#         if veh_id in seen_ids or not track:
#             continue
#         last_seen = track[-1][2]
#         if current_time - last_seen > 5.0:
#             stale_ids.append(veh_id)

#     for veh_id in stale_ids:
#         vehicle_tracks.pop(veh_id, None)
#         vehicle_lane_state.pop(veh_id, None)
#         vehicle_class_state.pop(veh_id, None)
#         vehicle_stop_start.pop(veh_id, None)

#     return overlay, aligned, current_violations


# try:
#     # Reference frame first.
#     reference_time = 0.0
#     overlay_ref, _, _ = process_frame(reference_frame, reference_time)
#     out.write(overlay_ref)
#     written_frames += 1
#     total_processed_frames += 1

#     frame_index = 1
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_time = frame_index / fps
#         overlay, aligned, _ = process_frame(frame, current_time)
#         if aligned:
#             aligned_frames += 1

#         out.write(overlay)
#         written_frames += 1
#         total_processed_frames += 1
#         frame_index += 1
# except Exception:
#     out.release()
#     cap.release()
#     if output_path.exists():
#         output_path.unlink()
#     raise
# finally:
#     if cap.isOpened():
#         cap.release()
#     out.release()

# print(f"Output saved as {output_path}")
# print(f"Video codec used: {selected_codec}")
# print(f"Input frames: {total_frames}, written frames: {written_frames}, fps: {fps:.2f}")
# print(f"Alignment successful on {aligned_frames}/{max(1, total_processed_frames - 1)} non-reference frames")
