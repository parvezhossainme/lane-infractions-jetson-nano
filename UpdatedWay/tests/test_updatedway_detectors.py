"""
Standalone UpdatedWay test script.
- Does not modify tests/.
- Validates OnlyLaneDetect output generation.
- Validates time-based lane-overlap violation logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "samples/videos/Sample-Main.mp4"
OUTPUT_DIR = ROOT / "UpdatedWay/outputs"
ONLY_LANE_SCRIPT = ROOT / "UpdatedWay/detectors/OnlyLaneDetect.py"


@dataclass
class TimeBasedLaneOverlapDetector:
    """
    Marks violation when a vehicle remains on/over a lane boundary for a minimum time.
    """

    fps: float
    min_overlap_seconds: float = 2.0
    overlap_ratio_threshold: float = 0.25
    overlap_frames: dict[int, int] = field(default_factory=dict)

    def update(self, vehicle_id: int, overlap_ratio: float) -> bool:
        current = self.overlap_frames.get(vehicle_id, 0)

        if overlap_ratio >= self.overlap_ratio_threshold:
            current += 1
        else:
            current = 0

        self.overlap_frames[vehicle_id] = current

        required_frames = int(self.min_overlap_seconds * self.fps)
        return current >= required_frames


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def test_only_lane_detect_output() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    if not ONLY_LANE_SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {ONLY_LANE_SCRIPT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in OUTPUT_DIR.glob("only_lane_output_*")}

    subprocess.run([sys.executable, str(ONLY_LANE_SCRIPT)], cwd=str(ROOT), check=True)

    after = sorted(OUTPUT_DIR.glob("only_lane_output_*"), key=os.path.getmtime)
    new_files = [p for p in after if p.name not in before]
    if not new_files:
        raise AssertionError("OnlyLaneDetect did not create a new output file")

    latest = new_files[-1]

    in_cap = cv2.VideoCapture(str(INPUT_VIDEO))
    out_cap = cv2.VideoCapture(str(latest))
    if not in_cap.isOpened() or not out_cap.isOpened():
        raise RuntimeError("Could not open input/output video for validation")

    in_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = float(in_cap.get(cv2.CAP_PROP_FPS))
    out_fps = float(out_cap.get(cv2.CAP_PROP_FPS))

    in_cap.release()
    out_cap.release()

    if in_frames <= 0 or out_frames <= 0:
        raise AssertionError("Frame count is zero in input or output")

    if out_frames != in_frames:
        raise AssertionError(f"Frame mismatch input={in_frames}, output={out_frames}")

    if out_fps <= 0:
        raise AssertionError("Output FPS is invalid")

    print("[OK] OnlyLaneDetect output validation passed")
    print(f"     output: {latest}")
    print(f"     frames: {out_frames}, fps: {out_fps:.2f}, input_fps: {in_fps:.2f}")


def test_time_based_lane_violation() -> None:
    detector = TimeBasedLaneOverlapDetector(fps=30.0, min_overlap_seconds=2.0, overlap_ratio_threshold=0.25)

    vehicle_id = 101

    # 1 second over lane boundary -> no violation yet
    violation = False
    for _ in range(30):
        violation = detector.update(vehicle_id, overlap_ratio=0.35)
    if violation:
        raise AssertionError("Violation triggered too early")

    # Continue overlap beyond 2 seconds total -> should violate
    for _ in range(31):
        violation = detector.update(vehicle_id, overlap_ratio=0.40)
    if not violation:
        raise AssertionError("Violation not triggered after sustained overlap")

    # Move back inside lane -> overlap counter should reset
    for _ in range(10):
        _ = detector.update(vehicle_id, overlap_ratio=0.0)

    # Another short overlap should not trigger immediately
    violation = False
    for _ in range(20):
        violation = detector.update(vehicle_id, overlap_ratio=0.30)
    if violation:
        raise AssertionError("Violation should not trigger after reset with short overlap")

    print("[OK] Time-based lane-overlap violation logic passed")


def run_all() -> None:
    print("=" * 70)
    print("UPDATEDWAY DETECTOR TEST SUITE")
    print("=" * 70)

    test_only_lane_detect_output()
    test_time_based_lane_violation()

    print("=" * 70)
    print("ALL UPDATEDWAY TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
