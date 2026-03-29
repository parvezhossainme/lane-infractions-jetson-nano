import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


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


def get_unique_output_path(directory: Path, base_name: str = "only_lane_output", ext: str = ".mp4") -> Path:
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


def create_video_writer(directory: Path, fps_value: float, frame_size: tuple[int, int]):
    candidates = [
        (".mp4", "mp4v"),
        (".avi", "XVID"),
        (".avi", "MJPG"),
    ]

    for ext, codec in candidates:
        path = get_unique_output_path(directory, ext=ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps_value, frame_size)
        if writer.isOpened():
            return writer, path, codec
        writer.release()

    raise RuntimeError("Failed to open VideoWriter with all codec/container fallbacks")


out, output_path, selected_codec = create_video_writer(output_dir, fps, (width, height))


# =========================
# Motion Alignment Setup
# =========================
# Estimate camera motion relative to the first frame and transform lane overlay
# accordingly, so lane markings stay fixed to the road in the output.
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
# Fixed Lane Geometry (Static Camera)
# =========================
lane_colors = [
    ((0, 0, 255), "Lane1 25-35"),
    ((255, 0, 0), "Lane2 35-45"),
    ((0, 255, 0), "Lane3 45-55"),
    ((0, 255, 255), "Lane4 55-65"),
]

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
seed_lanes = [lane1, lane2, lane3, lane4]
lanes = seed_lanes

lane_overlay_template = np.zeros((height, width, 3), dtype=np.uint8)
for i, lane in enumerate(lanes):
    color, label = lane_colors[i]
    cv2.polylines(lane_overlay_template, lane, True, color, 4)

    cx = int(np.mean(lane[0][:, 0]))
    cy = int(np.mean(lane[0][:, 1]))
    cv2.putText(
        lane_overlay_template,
        label,
        (cx - 60, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
    )


# =========================
# Full Video Pass (Fixed lanes)
# =========================
written_frames = 0
aligned_frames = 0
total_processed_frames = 0


def transform_overlay_for_frame(frame_in):
    if reference_points is None or len(reference_points) < 10:
        return lane_overlay_template

    gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        gray,
        reference_points,
        None,
    )

    if tracked_points is None or status is None:
        return lane_overlay_template

    good_ref = reference_points[status.flatten() == 1]
    good_cur = tracked_points[status.flatten() == 1]
    if len(good_ref) < 10 or len(good_cur) < 10:
        return lane_overlay_template

    transform, _ = cv2.estimateAffinePartial2D(
        good_ref,
        good_cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if transform is None:
        return lane_overlay_template

    return cv2.warpAffine(
        lane_overlay_template,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# Process reference frame first.
overlay_ref = cv2.add(reference_frame, lane_overlay_template)
out.write(overlay_ref)
written_frames += 1
total_processed_frames += 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped_overlay = transform_overlay_for_frame(frame)
    if warped_overlay is not lane_overlay_template:
        aligned_frames += 1

    overlay = cv2.add(frame, warped_overlay)
    out.write(overlay)
    written_frames += 1
    total_processed_frames += 1

cap.release()
out.release()

print(f"Output saved as {output_path}")
print(f"Video codec used: {selected_codec}")
print(f"Input frames: {total_frames}, written frames: {written_frames}, fps: {fps:.2f}")
print(f"Alignment successful on {aligned_frames}/{max(1, total_processed_frames - 1)} non-reference frames")
