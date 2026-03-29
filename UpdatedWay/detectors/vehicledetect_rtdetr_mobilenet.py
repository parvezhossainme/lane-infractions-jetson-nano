#!/usr/bin/env python3
"""Vehicle detection on video using RT-DETR (UpdatedWay pipeline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RTDETRV2_DIR = ROOT / "RT-DETR" / "rtdetrv2_pytorch"
if str(RTDETRV2_DIR) not in sys.path:
	sys.path.insert(0, str(RTDETRV2_DIR))

from src.core import YAMLConfig  # type: ignore[import-not-found]


VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_MOBILENET_MODEL = "outputs/mobilenet_train_coco8/best.pth"
DEFAULT_MOBILENET_CONFIG = "rt-detr-backbones/mobilenet/rtdetrv2_mobilenetv3_large_120e_coco.yml"


def is_mobilenet_model_path(model_path: str) -> bool:
	return "mobilenet" in model_path.lower()


class RTDETRv2DeployedModel(nn.Module):
	def __init__(self, cfg: YAMLConfig) -> None:
		super().__init__()
		self.model = cfg.model.deploy()
		self.postprocessor = cfg.postprocessor.deploy()

	def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
		outputs = self.model(images)
		return self.postprocessor(outputs, orig_target_sizes)


def load_model(config_path: str, model_path: str, device: str) -> tuple[nn.Module, T.Compose, torch.device]:
	if not is_mobilenet_model_path(model_path):
		raise ValueError(
			"This script must use a MobileNet backbone model. "
			"Provide a checkpoint path containing 'mobilenet'."
		)
	config = Path(config_path)
	if not config.exists():
		raise FileNotFoundError(f"RT-DETRv2 config not found: {config_path}")
	resume = Path(model_path)
	if not resume.exists():
		raise FileNotFoundError(
			f"MobileNet model checkpoint not found: {model_path}. "
			"Train/export a MobileNet RT-DETR checkpoint first."
		)

	print(f"Loading MobileNet RT-DETRv2 config: {config_path}")
	print(f"Loading MobileNet checkpoint: {model_path}")

	cfg = YAMLConfig(str(config), resume=str(resume))
	checkpoint = torch.load(str(resume), map_location="cpu")
	if isinstance(checkpoint.get("ema"), dict) and isinstance(checkpoint["ema"].get("module"), dict):
		state = checkpoint["ema"]["module"]
	elif isinstance(checkpoint.get("model"), dict):
		state = checkpoint["model"]
	else:
		raise RuntimeError("Unsupported checkpoint format for RT-DETRv2 resume model")

	cfg.model.load_state_dict(state)
	device_obj = torch.device(device)
	model = RTDETRv2DeployedModel(cfg).to(device_obj).eval()
	transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
	return model, transforms, device_obj


def filter_vehicle_detections(labels, boxes, scores, confidence: float, fallback_topk: int = 10) -> list[dict]:
	vehicle_candidates: list[dict] = []
	for cls_id, box, score in zip(labels, boxes, scores):
		cls_id = int(cls_id)
		if cls_id not in VEHICLE_CLASSES:
			continue
		score = float(score)
		vehicle_candidates.append(
			{
				"class_id": cls_id,
				"class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
				"confidence": score,
				"bbox": [int(v) for v in box],
			}
		)

	detections = [d for d in vehicle_candidates if d["confidence"] >= confidence]
	if detections:
		return detections

	# Fallback: when threshold removes all vehicles, still draw top vehicle candidates.
	vehicle_candidates.sort(key=lambda d: d["confidence"], reverse=True)
	return vehicle_candidates[: max(1, fallback_topk)]


def draw_detections(frame, detections: list[dict], frame_index: int):
	output = frame.copy()
	h, w = output.shape[:2]
	for det in detections:
		x1, y1, x2, y2 = det["bbox"]
		x1 = max(0, min(w - 1, x1))
		x2 = max(0, min(w - 1, x2))
		y1 = max(0, min(h - 1, y1))
		y2 = max(0, min(h - 1, y2))
		if x2 <= x1 or y2 <= y1:
			continue
		label = f"{det['class_name']} {det['confidence']:.2f}"

		cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
		cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw, y1), (0, 220, 0), -1)
		cv2.putText(output, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

	status = f"Frame: {frame_index} | Vehicles: {len(detections)}"
	cv2.putText(output, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2)
	return output


def process_video(input_video: Path, output_dir: Path, config_path: str, model_path: str, confidence: float, device: str) -> dict:
	model, transforms, device_obj = load_model(config_path=config_path, model_path=model_path, device=device)

	cap = cv2.VideoCapture(str(input_video))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {input_video}")

	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	output_video = output_dir / f"{input_video.stem}_vehicle_detection.mp4"
	writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
	if not writer.isOpened():
		cap.release()
		raise RuntimeError(f"Could not create output video: {output_video}")

	frame_idx = 0
	peak_vehicles = 0
	total_vehicles = 0

	print(f"Input: {input_video}")
	print(f"Output: {output_video}")
	print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames}")

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		im_pil = Image.fromarray(rgb)
		w, h = im_pil.size
		orig_size = torch.tensor([w, h], dtype=torch.float32)[None].to(device_obj)
		im_data = transforms(im_pil)[None].to(device_obj)

		with torch.no_grad():
			labels, boxes, scores = model(im_data, orig_size)

		labels_np = labels[0].detach().cpu().numpy().tolist()
		boxes_np = boxes[0].detach().cpu().numpy().tolist()
		scores_np = scores[0].detach().cpu().numpy().tolist()
		detections = filter_vehicle_detections(labels_np, boxes_np, scores_np, confidence)
		drawn = draw_detections(frame, detections, frame_idx)
		writer.write(drawn)

		count = len(detections)
		peak_vehicles = max(peak_vehicles, count)
		total_vehicles += count
		frame_idx += 1

	cap.release()
	writer.release()

	avg_vehicles = (total_vehicles / frame_idx) if frame_idx else 0.0
	return {
		"input_video": str(input_video),
		"output_video": str(output_video),
		"frames_processed": frame_idx,
		"peak_vehicles_in_frame": peak_vehicles,
		"average_vehicles_per_frame": round(avg_vehicles, 3),
		"confidence_threshold": confidence,
		"model_path": model_path,
	}


def main() -> int:
	parser = argparse.ArgumentParser(description="UpdatedWay RT-DETR vehicle detection on video")
	parser.add_argument(
		"--input",
		default="samples/videos/Sample-5Sec.mp4",
		help="Input video path (default: Sample-5Sec.mp4)",
	)
	parser.add_argument(
		"--output-dir",
		default="UpdatedWay/outputs",
		help="Output directory for video and report",
	)
	parser.add_argument(
		"--config",
		default=DEFAULT_MOBILENET_CONFIG,
		help="RT-DETRv2 MobileNet config",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MOBILENET_MODEL,
		help="MobileNet RT-DETR model weights",
	)
	parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
	parser.add_argument("--confidence", type=float, default=0.05, help="Detection confidence threshold")
	args = parser.parse_args()

	input_video = Path(args.input)
	if not input_video.exists():
		print(f"[ERROR] Video not found: {input_video}")
		return 1

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	try:
		stats = process_video(
			input_video=input_video,
			output_dir=output_dir,
			config_path=args.config,
			model_path=args.model,
			confidence=args.confidence,
			device=args.device,
		)
	except Exception as exc:
		print(f"[ERROR] {exc}")
		return 1

	report_path = output_dir / f"{input_video.stem}_vehicle_detection_report.json"
	report_path.write_text(json.dumps({"video_stats": stats}, indent=2), encoding="utf-8")
	print(f"[OK] Report saved: {report_path}")
	print("[OK] Video detection complete")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
