#!/usr/bin/env python3
"""Run RT-DETR validation across backbones and summarize paper-ready metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import torch


COCO_METRIC_KEYS = [
    "AP@[0.50:0.95]_all_maxDets100",
    "AP@0.50_all_maxDets100",
    "AP@0.75_all_maxDets100",
    "AP@[0.50:0.95]_small_maxDets100",
    "AP@[0.50:0.95]_medium_maxDets100",
    "AP@[0.50:0.95]_large_maxDets100",
    "AR@[0.50:0.95]_all_maxDets1",
    "AR@[0.50:0.95]_all_maxDets10",
    "AR@[0.50:0.95]_all_maxDets100",
    "AR@[0.50:0.95]_small_maxDets100",
    "AR@[0.50:0.95]_medium_maxDets100",
    "AR@[0.50:0.95]_large_maxDets100",
]

COCO_LINE_PATTERN = re.compile(
    r"Average\s+(Precision|Recall)\s+\((AP|AR)\) @\[ IoU=\s*([^|]+)\| area=\s*([^|]+)\| maxDets=\s*(\d+) \] =\s*([0-9.]+|nan)"
)
TOTAL_TIME_PATTERN = re.compile(r"Test:\s+Total time:\s+([0-9:]+)\s+\(([0-9.]+)\s+s / it\)")
MAX_MEM_PATTERN = re.compile(r"max mem:\s*([0-9.]+)")
MAX_RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    peak_rss_mb_proc: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR backbones and summarize metrics")
    parser.add_argument(
        "--runs",
        default="rt-detr-backbones/benchmark/backbone_runs.json",
        help="JSON list of runs (name/config/checkpoint)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run RT-DETR validation",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to tools/train.py (e.g., cuda:0 or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/backbone_benchmark",
        help="Directory to store per-run logs and summary",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a run fails",
    )
    parser.add_argument(
        "--update",
        nargs="*",
        default=[],
        help=(
            "Optional YAML overrides passed to train.py -u, e.g. "
            "val_dataloader.dataset.img_folder=/abs/path/val "
            "val_dataloader.dataset.ann_file=/abs/path/instances_val.json"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds for stability stats (e.g. 0,1,2).",
    )
    return parser.parse_args()


def load_runs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Runs JSON must be a list")
    for item in data:
        if "name" not in item or "config" not in item:
            raise ValueError("Each run must include at least 'name' and 'config'")
    return data


def resolve_path(workspace: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (workspace / p).resolve()


def is_remote_checkpoint(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def parse_coco_metrics(stdout: str) -> dict[str, float | None]:
    rows = COCO_LINE_PATTERN.findall(stdout)
    values: list[float | None] = []
    for _, _, _, _, _, v in rows[:12]:
        values.append(None if v == "nan" else float(v))

    # Keep output shape stable even if parser captured fewer lines.
    while len(values) < 12:
        values.append(None)

    return {k: values[i] for i, k in enumerate(COCO_METRIC_KEYS)}


def parse_seeds(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [0]


def parse_update_items(update_items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in update_items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def read_ann_file_from_config(config_path: Path) -> str | None:
    try:
        txt = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"ann_file:\s*([^\n#]+)", txt)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def get_val_ann_file(workspace: Path, config_path: Path, update_items: list[str]) -> Path | None:
    updates = parse_update_items(update_items)
    candidate = updates.get("val_dataloader.dataset.ann_file")
    if candidate:
        p = Path(candidate)
        return p if p.is_absolute() else (workspace / p).resolve()

    cfg_ann = read_ann_file_from_config(config_path)
    if cfg_ann:
        p = Path(cfg_ann)
        return p if p.is_absolute() else (workspace / p).resolve()
    return None


def parse_total_eval_seconds(stdout: str) -> float | None:
    m = TOTAL_TIME_PATTERN.search(stdout)
    if not m:
        return None

    hms = m.group(1)
    parts = [int(x) for x in hms.split(":")]
    if len(parts) != 3:
        return None
    hh, mm, ss = parts
    return float(hh * 3600 + mm * 60 + ss)


def parse_peak_gpu_mem(stdout: str) -> float | None:
    vals = [float(x) for x in MAX_MEM_PATTERN.findall(stdout)]
    return max(vals) if vals else None


def parse_peak_rss_mb(text: str) -> float | None:
    m = MAX_RSS_PATTERN.search(text)
    if not m:
        return None
    kb = float(m.group(1))
    return kb / 1024.0


def safe_torch_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def metric_mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    xs = [x for x in values if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def checkpoint_num_params(checkpoint_raw: str) -> int | None:
    if not checkpoint_raw:
        return None
    state: Any = None
    try:
        if is_remote_checkpoint(checkpoint_raw):
            state = torch.hub.load_state_dict_from_url(checkpoint_raw, map_location="cpu")
        else:
            state = torch.load(checkpoint_raw, map_location="cpu", weights_only=False)
    except TypeError:
        if not is_remote_checkpoint(checkpoint_raw):
            state = torch.load(checkpoint_raw, map_location="cpu")
    except Exception:
        return None

    tensor_dict: dict[str, Any] | None = None
    if isinstance(state, dict):
        if isinstance(state.get("ema"), dict) and isinstance(state["ema"].get("module"), dict):
            tensor_dict = state["ema"]["module"]
        elif isinstance(state.get("model"), dict):
            tensor_dict = state["model"]
        elif all(isinstance(v, torch.Tensor) for v in state.values()):
            tensor_dict = state

    if not tensor_dict:
        return None

    total = 0
    for v in tensor_dict.values():
        if isinstance(v, torch.Tensor):
            total += int(v.numel())
    return total or None


def extract_eval_advanced_metrics(run_dir: Path, ann_file: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "per_class_ap": None,
        "best_f1": None,
        "best_f1_threshold": None,
    }

    eval_path = run_dir / "eval.pth"
    eval_obj = safe_torch_load(eval_path)
    if not isinstance(eval_obj, dict):
        return out

    precision = eval_obj.get("precision")
    scores = eval_obj.get("scores")
    params = eval_obj.get("params")
    if precision is None or params is None:
        return out

    try:
        area_labels = list(params.areaRngLbl)
        max_dets = list(params.maxDets)
        cat_ids = list(params.catIds)
        rec_thrs = list(params.recThrs)
        iou_thrs = list(params.iouThrs)

        area_idx = area_labels.index("all")
        maxdet_idx = max_dets.index(100)

        # Per-class AP over IoU=0.50:0.95
        p = precision[:, :, :, area_idx, maxdet_idx]  # [T, R, K]
        per_class = {}
        name_by_id: dict[int, str] = {}
        if ann_file and ann_file.exists():
            ann = json.loads(ann_file.read_text(encoding="utf-8"))
            name_by_id = {int(c["id"]): str(c["name"]) for c in ann.get("categories", [])}

        for k, cat_id in enumerate(cat_ids):
            vals = p[:, :, k]
            vals = vals[vals > -1]
            if vals.size == 0:
                ap_val = None
            else:
                ap_val = float(vals.mean())
            key = name_by_id.get(int(cat_id), f"cat_{cat_id}")
            per_class[key] = ap_val
        out["per_class_ap"] = per_class

        # Best F1 and confidence threshold at IoU=0.50, area=all, maxDets=100
        iou_idx = min(range(len(iou_thrs)), key=lambda i: abs(float(iou_thrs[i]) - 0.5))
        p_iou = precision[iou_idx, :, :, area_idx, maxdet_idx]  # [R, K]
        s_iou = None
        if scores is not None:
            s_iou = scores[iou_idx, :, :, area_idx, maxdet_idx]  # [R, K]

        best = (-1.0, None, None)
        for r_i, r in enumerate(rec_thrs):
            pv = p_iou[r_i]
            valid = pv > -1
            if valid.sum() == 0:
                continue
            p_mean = float(pv[valid].mean())
            r_val = float(r)
            f1 = 0.0 if (p_mean + r_val) == 0 else (2 * p_mean * r_val) / (p_mean + r_val)
            thr_val = None
            if s_iou is not None:
                sv = s_iou[r_i]
                s_valid = sv[valid]
                if s_valid.size > 0:
                    thr_val = float(s_valid.mean())
            if f1 > best[0]:
                best = (f1, thr_val, r_val)

        if best[0] >= 0:
            out["best_f1"] = best[0]
            out["best_f1_threshold"] = best[1]
    except Exception:
        return out

    return out


def build_cmd(
    python_bin: str,
    train_py: Path,
    config: Path,
    checkpoint: str,
    checkpoint_mode: str,
    seed: int,
    device: str,
    run_output_dir: Path,
    update_items: list[str],
    supports_device_output_update: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(train_py),
        "-c",
        str(config),
        "--test-only",
    ]
    if supports_device_output_update:
        cmd.extend([
            "--device",
            device,
            "--output-dir",
            str(run_output_dir),
        ])
    cmd.extend(["--seed", str(seed)])
    if checkpoint:
        if checkpoint_mode == "tuning":
            cmd.extend(["-t", checkpoint])
        else:
            cmd.extend(["-r", checkpoint])
    if supports_device_output_update and update_items:
        cmd.extend(["-u", *update_items])
    return cmd


def resolve_framework_run_context(
    workspace: Path,
    framework: str,
) -> tuple[Path, Path, bool]:
    fw = framework.strip().lower()
    if fw in {"rtdetr_pytorch", "rtdetr-v1", "v1"}:
        return (
            (workspace / "RT-DETR" / "rtdetr_pytorch" / "tools" / "train.py").resolve(),
            (workspace / "RT-DETR" / "rtdetr_pytorch").resolve(),
            False,
        )

    # Default: RT-DETRv2 train entrypoint with support for --device/--output-dir/-u
    return (
        (workspace / "RT-DETR" / "rtdetrv2_pytorch" / "tools" / "train.py").resolve(),
        (workspace / "RT-DETR" / "rtdetrv2_pytorch").resolve(),
        True,
    )


def _read_proc_rss_mb(pid: int) -> float | None:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def run_one(cmd: list[str], cwd: Path) -> RunResult:
    # Wrap with GNU time when available to capture peak RSS memory.
    time_bin = Path("/usr/bin/time")
    final_cmd = cmd
    if time_bin.exists():
        final_cmd = [str(time_bin), "-v", *cmd]

    proc = subprocess.Popen(final_cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss_mb_proc: float | None = None
    while True:
        rss_mb = _read_proc_rss_mb(proc.pid)
        if rss_mb is not None:
            peak_rss_mb_proc = rss_mb if peak_rss_mb_proc is None else max(peak_rss_mb_proc, rss_mb)
        try:
            stdout, stderr = proc.communicate(timeout=0.1)
            return RunResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                peak_rss_mb_proc=peak_rss_mb_proc,
            )
        except subprocess.TimeoutExpired:
            time.sleep(0.05)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Backbone Benchmark Summary",
        "",
        "| Backbone | Status | AP | AP50 | AP75 | AP_S | AP_M | AP_L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        def fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.4f}"

        lines.append(
            "| {name} | {status} | {ap} | {ap50} | {ap75} | {aps} | {apm} | {apl} |".format(
                name=row["name"],
                status=row["status"],
                ap=fmt(row.get("AP@[0.50:0.95]_all_maxDets100")),
                ap50=fmt(row.get("AP@0.50_all_maxDets100")),
                ap75=fmt(row.get("AP@0.75_all_maxDets100")),
                aps=fmt(row.get("AP@[0.50:0.95]_small_maxDets100")),
                apm=fmt(row.get("AP@[0.50:0.95]_medium_maxDets100")),
                apl=fmt(row.get("AP@[0.50:0.95]_large_maxDets100")),
            )
        )

    lines += [
        "",
        "## Full COCO Metrics (AP + AR)",
        "",
        "| Backbone | Status | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR@1 | AR@10 | AR@100 | AR_S | AR_M | AR_L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        def fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.4f}"

        lines.append(
            "| {name} | {status} | {ap} | {ap50} | {ap75} | {aps} | {apm} | {apl} | {ar1} | {ar10} | {ar100} | {ars} | {arm} | {arl} |".format(
                name=row["name"],
                status=row["status"],
                ap=fmt(row.get("AP@[0.50:0.95]_all_maxDets100")),
                ap50=fmt(row.get("AP@0.50_all_maxDets100")),
                ap75=fmt(row.get("AP@0.75_all_maxDets100")),
                aps=fmt(row.get("AP@[0.50:0.95]_small_maxDets100")),
                apm=fmt(row.get("AP@[0.50:0.95]_medium_maxDets100")),
                apl=fmt(row.get("AP@[0.50:0.95]_large_maxDets100")),
                ar1=fmt(row.get("AR@[0.50:0.95]_all_maxDets1")),
                ar10=fmt(row.get("AR@[0.50:0.95]_all_maxDets10")),
                ar100=fmt(row.get("AR@[0.50:0.95]_all_maxDets100")),
                ars=fmt(row.get("AR@[0.50:0.95]_small_maxDets100")),
                arm=fmt(row.get("AR@[0.50:0.95]_medium_maxDets100")),
                arl=fmt(row.get("AR@[0.50:0.95]_large_maxDets100")),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    workspace = Path.cwd().resolve()
    runs_path = resolve_path(workspace, args.runs)
    out_root = resolve_path(workspace, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_path)
    seeds = parse_seeds(args.seeds)
    results: list[dict[str, Any]] = []

    for run in runs:
        name = run["name"]
        config_path = resolve_path(workspace, run["config"])
        framework = str(run.get("framework", "rtdetrv2")).strip()
        checkpoint_raw = str(run.get("checkpoint", "")).strip()
        if checkpoint_raw:
            checkpoint = checkpoint_raw if is_remote_checkpoint(checkpoint_raw) else str(resolve_path(workspace, checkpoint_raw))
        else:
            checkpoint = ""
        checkpoint_mode = str(run.get("checkpoint_mode", "resume")).strip().lower()
        if checkpoint_mode not in {"resume", "tuning"}:
            results.append(
                {
                    "name": name,
                    "status": "failed(bad-checkpoint-mode)",
                    "config": str(config_path),
                    "framework": framework,
                    "checkpoint": checkpoint_raw,
                    "checkpoint_mode": checkpoint_mode,
                }
            )
            if args.stop_on_error:
                break
            continue

        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        ann_file = get_val_ann_file(workspace, config_path, args.update)

        if not checkpoint:
            results.append({
                "name": name,
                "status": "skipped(no-checkpoint)",
                "config": str(config_path),
                "framework": framework,
                "checkpoint": checkpoint_raw,
            })
            continue

        if not config_path.exists():
            results.append({
                "name": name,
                "status": "failed(config-missing)",
                "config": str(config_path),
                "framework": framework,
                "checkpoint": checkpoint_raw,
            })
            continue

        if checkpoint and (not is_remote_checkpoint(checkpoint_raw)) and (not Path(checkpoint).exists()):
            results.append({
                "name": name,
                "status": "failed(checkpoint-missing)",
                "config": str(config_path),
                "framework": framework,
                "checkpoint": checkpoint_raw,
            })
            if args.stop_on_error:
                break
            continue

        train_py, run_cwd, supports_device_output_update = resolve_framework_run_context(
            workspace=workspace,
            framework=framework,
        )
        seed_rows: list[dict[str, Any]] = []
        any_failed = False

        for seed in seeds:
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            log_path = seed_dir / "stdout.log"

            cmd = build_cmd(
                python_bin=args.python,
                train_py=train_py,
                config=config_path,
                checkpoint=checkpoint,
                checkpoint_mode=checkpoint_mode,
                device=args.device,
                run_output_dir=seed_dir,
                update_items=args.update,
                supports_device_output_update=supports_device_output_update,
                seed=seed,
            )
            proc = run_one(cmd, cwd=run_cwd)
            log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

            if proc.returncode != 0:
                any_failed = True
                seed_rows.append(
                    {
                        "seed": seed,
                        "status": f"failed(exit-{proc.returncode})",
                        "log": str(log_path),
                    }
                )
                if args.stop_on_error:
                    break
                continue

            coco_values = parse_coco_metrics(proc.stdout)
            adv = extract_eval_advanced_metrics(seed_dir, ann_file)

            eval_seconds = parse_total_eval_seconds(proc.stdout)
            num_images = None
            if ann_file and ann_file.exists():
                try:
                    num_images = len(json.loads(ann_file.read_text(encoding="utf-8")).get("images", []))
                except Exception:
                    num_images = None

            latency_ms = None
            fps = None
            if eval_seconds and num_images and num_images > 0:
                latency_ms = (eval_seconds / num_images) * 1000.0
                fps = num_images / eval_seconds if eval_seconds > 0 else None

            seed_rows.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "log": str(log_path),
                    **coco_values,
                    "per_class_ap": adv.get("per_class_ap"),
                    "best_f1": adv.get("best_f1"),
                    "best_f1_threshold": adv.get("best_f1_threshold"),
                    "latency_ms_mean": latency_ms,
                    "latency_ms_p95": latency_ms,
                    "throughput_fps": fps,
                    "peak_gpu_mem_mb": parse_peak_gpu_mem(proc.stdout),
                    "peak_rss_mb": parse_peak_rss_mb(proc.stderr + "\n" + proc.stdout) or proc.peak_rss_mb_proc,
                }
            )

        params_count = checkpoint_num_params(checkpoint_raw)
        params_m = (params_count / 1_000_000.0) if params_count is not None else None
        model_size_mb = (params_count * 4 / (1024 * 1024)) if params_count is not None else None

        if not seed_rows:
            results.append(
                {
                    "name": name,
                    "status": "failed(no-seed-runs)",
                    "config": str(config_path),
                    "framework": framework,
                    "checkpoint": checkpoint_raw,
                    "checkpoint_mode": checkpoint_mode,
                }
            )
            continue

        ok_rows = [r for r in seed_rows if r.get("status") == "ok"]
        if not ok_rows:
            first_fail = seed_rows[0]
            results.append(
                {
                    "name": name,
                    "status": first_fail.get("status", "failed"),
                    "config": str(config_path),
                    "framework": framework,
                    "checkpoint": checkpoint_raw,
                    "checkpoint_mode": checkpoint_mode,
                    "log": first_fail.get("log"),
                    "seed_runs": seed_rows,
                }
            )
            if args.stop_on_error:
                break
            continue

        agg: dict[str, Any] = {
            "name": name,
            "status": "ok" if not any_failed else "ok(partial-seed-failures)",
            "config": str(config_path),
            "framework": framework,
            "checkpoint": checkpoint_raw,
            "checkpoint_mode": checkpoint_mode,
            "seed_runs": seed_rows,
            "stability_n_seeds": len(ok_rows),
            "params_m": params_m,
            "model_size_mb": model_size_mb,
            "flops_g": run.get("flops_g"),
            "per_class_ap": ok_rows[0].get("per_class_ap"),
        }

        for key in COCO_METRIC_KEYS + [
            "best_f1",
            "best_f1_threshold",
            "latency_ms_mean",
            "latency_ms_p95",
            "throughput_fps",
            "peak_gpu_mem_mb",
            "peak_rss_mb",
        ]:
            vals = [_to_float(r.get(key)) for r in ok_rows]
            mean_v, std_v = metric_mean_std(vals)
            agg[key] = mean_v
            agg[f"{key}_std"] = std_v

        # keep top-level log for convenience
        agg["log"] = ok_rows[0].get("log")
        results.append(agg)

    successful = [r for r in results if str(r.get("status", "")).startswith("ok")]
    successful.sort(key=lambda r: (r.get("AP@[0.50:0.95]_all_maxDets100") or -1.0), reverse=True)

    failed_or_skipped = [r for r in results if r.get("status") != "ok"]
    final_rows = successful + failed_or_skipped

    summary_json = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "device": args.device,
        "runs_file": str(runs_path),
        "results": final_rows,
    }

    json_path = out_root / "summary.json"
    md_path = out_root / "summary.md"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    write_markdown(md_path, final_rows)

    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary markdown: {md_path}")

    if successful:
        top = successful[0]
        print(
            "Best AP backbone: "
            f"{top['name']} (AP={top['AP@[0.50:0.95]_all_maxDets100']:.4f})"
        )
    else:
        print("No successful runs. Check per-run logs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
