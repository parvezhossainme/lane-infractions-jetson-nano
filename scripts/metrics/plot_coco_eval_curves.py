#!/usr/bin/env python3
"""Plot COCOeval precision-confidence and recall-confidence curves from saved eval .pth."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Standard COCO 80 class names indexed by category id - 1.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def cat_name(cat_id: int) -> str:
    if 1 <= cat_id <= len(COCO80_NAMES):
        return COCO80_NAMES[cat_id - 1]
    return f"cat_{cat_id}"


def parse_alias_map(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if old:
            out[old] = new if new else old
    return out


def normalize_token(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def resolve_requested_indices(
    requested: list[str],
    cat_ids: list[int],
    id_to_name: dict[int, str],
) -> list[int]:
    """Resolve user-selected class names/ids to category indices in cat_ids order."""
    if not requested:
        return []

    id_to_index = {cid: idx for idx, cid in enumerate(cat_ids)}
    name_to_id = {normalize_token(name): cid for cid, name in id_to_name.items()}

    resolved: list[int] = []
    seen = set()
    for token in requested:
        tok = token.strip()
        if not tok:
            continue

        cid = None
        if tok.isdigit():
            maybe_id = int(tok)
            if maybe_id in id_to_index:
                cid = maybe_id
        else:
            cid = name_to_id.get(normalize_token(tok))

        if cid is None:
            continue

        k = id_to_index[cid]
        if k not in seen:
            resolved.append(k)
            seen.add(k)

    return resolved


def load_eval(pth_path: Path) -> dict:
    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    required = {"precision", "scores", "params"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")
    return data


def smooth_by_confidence(scores: np.ndarray, values: np.ndarray, bins: int = 101) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(scores) & np.isfinite(values)
    s = scores[mask]
    v = values[mask]
    if s.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(s)
    s = s[order]
    v = v[order]

    x = np.linspace(max(0.0, float(np.min(s))), min(1.0, float(np.max(s))), bins)
    y = np.interp(x, s, v)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-pth", required=True, help="Path to COCO eval .pth file")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--topk", type=int, default=8, help="Max number of per-class curves to draw")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Optional class filter by class names or COCO category IDs (e.g., person car 1 3)",
    )
    parser.add_argument(
        "--class-alias",
        nargs="*",
        default=[],
        help="Optional display renaming in legend using old=new (e.g., car=sedan truck=heavy_vehicle)",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_pth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_eval(eval_path)
    precision = np.asarray(data["precision"])  # [T, R, K, A, M]
    scores = np.asarray(data["scores"])        # [T, R, K, A, M]
    recall = np.asarray(data.get("recall")) if "recall" in data else None  # [T, K, A, M]

    params = data["params"]
    cat_ids = [int(x) for x in params.catIds]
    iou_thrs = np.asarray(params.iouThrs)
    id_to_name = {cid: cat_name(cid) for cid in cat_ids}
    alias_map = parse_alias_map(args.class_alias)

    # Use IoU=0.50 slice when available; else nearest IoU threshold.
    iou_idx = int(np.argmin(np.abs(iou_thrs - 0.5)))
    area_idx = 0   # all area
    maxdet_idx = 2 # typically maxDets=100

    cls_ap50 = []
    for k, cat_id in enumerate(cat_ids):
        p = precision[iou_idx, :, k, area_idx, maxdet_idx]
        valid = p[p > -1]
        ap = float(np.mean(valid)) if valid.size else -1.0
        cls_ap50.append((k, cat_id, ap))

    valid_classes = [(k, cid, ap) for (k, cid, ap) in cls_ap50 if ap >= 0.0]
    valid_classes.sort(key=lambda x: x[2], reverse=True)

    selected_indices = resolve_requested_indices(args.classes, cat_ids, id_to_name)
    if selected_indices:
        selected = []
        ap_by_k = {k: ap for (k, _cid, ap) in valid_classes}
        for k in selected_indices:
            cid = cat_ids[k]
            ap = ap_by_k.get(k, -1.0)
            if ap >= 0.0:
                selected.append((k, cid, ap))
    else:
        selected = valid_classes[: max(1, args.topk)]

    # Precision-Confidence plot.
    plt.figure(figsize=(10, 6))
    all_scores = []
    all_prec = []
    for k, cat_id, _ap in selected:
        p = precision[iou_idx, :, k, area_idx, maxdet_idx]
        s = scores[iou_idx, :, k, area_idx, maxdet_idx]
        mask = (p > -1) & (s >= 0)
        if np.sum(mask) < 2:
            continue
        x, y = smooth_by_confidence(s[mask], p[mask], bins=101)
        if x.size < 2:
            continue
        label = alias_map.get(cat_name(cat_id), cat_name(cat_id))
        plt.plot(x, y, linewidth=1.7, label=label)
        all_scores.append(s[mask])
        all_prec.append(p[mask])

    if all_scores:
        x_all, y_all = smooth_by_confidence(np.concatenate(all_scores), np.concatenate(all_prec), bins=101)
        if x_all.size:
            auc_like = float(np.trapezoid(y_all, x_all))
            plt.plot(x_all, y_all, color="navy", linewidth=2.5, label=f"all classes, AUC {auc_like:.3f}")

    plt.title("Precision-Confidence Curve")
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    p_out = out_dir / "precision_confidence_curve.png"
    plt.savefig(p_out, dpi=220)
    plt.close()

    # Recall-Confidence plot.
    plt.figure(figsize=(10, 6))
    all_scores_r = []
    all_rec = []
    for k, cat_id, _ap in selected:
        p = precision[iou_idx, :, k, area_idx, maxdet_idx]
        s = scores[iou_idx, :, k, area_idx, maxdet_idx]
        # COCO recThrs are 101 points from 0..1 corresponding to this axis.
        r = np.linspace(0.0, 1.0, p.shape[0], dtype=np.float64)
        mask = (p > -1) & (s >= 0)
        if np.sum(mask) < 2:
            continue
        x, y = smooth_by_confidence(s[mask], r[mask], bins=101)
        if x.size < 2:
            continue
        # Sort by confidence ascending; recall generally decreases as threshold rises.
        label = alias_map.get(cat_name(cat_id), cat_name(cat_id))
        plt.plot(x, y, linewidth=1.7, label=label)
        all_scores_r.append(s[mask])
        all_rec.append(r[mask])

    if all_scores_r:
        x_all, y_all = smooth_by_confidence(np.concatenate(all_scores_r), np.concatenate(all_rec), bins=101)
        if x_all.size:
            plt.plot(x_all, y_all, color="navy", linewidth=2.5, label="all classes")

    plt.title("Recall-Confidence Curve")
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    r_out = out_dir / "recall_confidence_curve.png"
    plt.savefig(r_out, dpi=220)
    plt.close()

    # Optional run summary.
    summary_path = out_dir / "curve_summary.txt"
    with summary_path.open("w", encoding="ascii") as f:
        f.write(f"eval_pth={eval_path}\n")
        f.write(f"iou_threshold_used={float(iou_thrs[iou_idx]):.2f}\n")
        if recall is not None:
            r_all = recall[iou_idx, :, area_idx, maxdet_idx]
            r_all = r_all[r_all >= 0]
            if r_all.size:
                f.write(f"mean_max_recall_iou{float(iou_thrs[iou_idx]):.2f}={float(np.mean(r_all)):.4f}\n")
        f.write("top_classes_ap50:\n")
        for _k, cid, ap in selected:
            shown = alias_map.get(cat_name(cid), cat_name(cid))
            f.write(f"  {shown} (id={cid}): {ap:.4f}\n")

    print(f"Saved: {p_out}")
    print(f"Saved: {r_out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
