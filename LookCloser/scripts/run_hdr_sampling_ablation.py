#!/usr/bin/env python3
"""Evaluate ray-sampling variants on one frozen HDR LookCloser checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


VARIANTS: dict[str, dict[str, Any]] = {
    "adaptive_max3": {"adaptive_interval_level_mode": "max3"},
    "adaptive_corrected": {"corrected_arm_allocator": True},
    "adaptive_minfreq4": {"adaptive_min_frequency_level": 4.0, "max_steps_per_ray": 2048},
    "adaptive_dense2x": {"adaptive_coarse_step_size": 0.003125, "max_steps_per_ray": 2048},
    "adaptive_dense2x_corrected": {
        "adaptive_coarse_step_size": 0.003125,
        "max_steps_per_ray": 2048,
        "corrected_arm_allocator": True,
    },
    "adaptive_fallback64": {"adaptive_fixed_fallback_samples_per_ray": 64},
    "fixed1024": {
        "ray_sampling_mode": "fixed",
        "enable_adaptive_ray_marching": False,
        "fixed_num_samples_per_ray": 1024,
    },
    "fixed2048": {
        "ray_sampling_mode": "fixed",
        "enable_adaptive_ray_marching": False,
        "fixed_num_samples_per_ray": 2048,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--baseline-render-dir", type=Path, required=True)
    parser.add_argument("--baseline-eval-json", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_row(name: str, eval_json: Path, edge_json: Path, render_dir: Path, seconds: float) -> dict[str, Any]:
    evaluation = json.loads(eval_json.read_text(encoding="utf-8"))["results"]
    edge = json.loads(edge_json.read_text(encoding="utf-8"))["aggregate"]
    return {
        "variant": name,
        "psnr": float(evaluation["psnr"]),
        "ssim": float(evaluation["ssim"]),
        "lpips": float(evaluation["lpips"]),
        "edge_recall": float(edge["edge_recall"]),
        "edge_f1": float(edge["edge_f1"]),
        "long_gap_fraction": float(edge["long_gap_fraction"]),
        "long_gap_count": int(edge["long_gap_count"]),
        "seconds": float(seconds),
        "eval_json": str(eval_json),
        "edge_json": str(edge_json),
        "render_dir": str(render_dir),
    }


def score_edges(render_dir: Path, output_dir: Path, data: Path) -> Path:
    edge_dir = output_dir / "edge_continuity"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "score_hdr_edge_continuity.py"),
        "--render-dir",
        str(render_dir),
        "--output-dir",
        str(edge_dir),
        "--data",
        str(data),
    ]
    subprocess.run(command, check=True)
    return edge_dir / "edge_continuity.json"


def evaluate_variant(args: argparse.Namespace, name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    variant_dir = args.output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    eval_json = variant_dir / "eval.json"
    render_dir = variant_dir / "renders"
    edge_json = variant_dir / "edge_continuity" / "edge_continuity.json"
    if eval_json.is_file() and edge_json.is_file() and not args.force:
        return metric_row(name, eval_json, edge_json, render_dir, 0.0)

    config = yaml.load(args.base_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    model = config.pipeline.model
    model.eval_num_rays_per_chunk = int(args.eval_num_rays_per_chunk)
    for key, value in overrides.items():
        if not hasattr(model, key):
            raise AttributeError(f"Model config has no field {key!r}")
        setattr(model, key, value)
    config_path = variant_dir / "config.yml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    log_path = variant_dir / "eval_stdout.log"
    ns_eval = Path(sys.executable).with_name("ns-eval")
    command = [
        str(ns_eval),
        "--load-config",
        str(config_path),
        "--output-path",
        str(eval_json),
        "--render-output-path",
        str(render_dir),
    ]
    print(f"variant={name} command={' '.join(command)}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    edge_json = score_edges(render_dir, variant_dir, args.data)
    return metric_row(name, eval_json, edge_json, render_dir, time.monotonic() - started)


def main() -> int:
    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = executable_dir + os.pathsep + os.environ.get("PATH", "")
    cuda_home = Path("/usr/local/cuda-12.6")
    if cuda_home.is_dir():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ["PATH"] = str(cuda_home / "bin") + os.pathsep + os.environ["PATH"]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")
    extension_cache = Path.home() / ".cache" / "torch_extensions_lookcloser"
    if extension_cache.is_dir():
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_cache))
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = args.output_dir / "baseline_adaptive"
    baseline_dir.mkdir(exist_ok=True)
    baseline_edge = baseline_dir / "edge_continuity" / "edge_continuity.json"
    if not baseline_edge.is_file() or args.force:
        baseline_edge = score_edges(args.baseline_render_dir, baseline_dir, args.data)
    rows = [
        metric_row(
            "baseline_adaptive",
            args.baseline_eval_json,
            baseline_edge,
            args.baseline_render_dir,
            0.0,
        )
    ]
    for name in args.variant or list(VARIANTS):
        row = evaluate_variant(args, name, VARIANTS[name])
        rows.append(row)
        print(
            f"result variant={name} psnr={row['psnr']:.5f} ssim={row['ssim']:.6f} "
            f"lpips={row['lpips']:.6f} edge_recall={row['edge_recall']:.5f} "
            f"long_gap={row['long_gap_fraction']:.5f}",
            flush=True,
        )
    baseline = rows[0]
    for row in rows:
        row["delta_psnr"] = row["psnr"] - baseline["psnr"]
        row["delta_ssim"] = row["ssim"] - baseline["ssim"]
        row["delta_lpips"] = row["lpips"] - baseline["lpips"]
        row["delta_edge_recall"] = row["edge_recall"] - baseline["edge_recall"]
        row["delta_long_gap_fraction"] = row["long_gap_fraction"] - baseline["long_gap_fraction"]
    manifest = {"schema": 1, "base_config": str(args.base_config), "variants": VARIANTS, "rows": rows}
    atomic_json(args.output_dir / "sampling_ablation.json", manifest)
    print(
        "summary="
        + json.dumps(
            [
                {
                    key: row[key]
                    for key in ("variant", "psnr", "ssim", "lpips", "edge_recall", "long_gap_fraction")
                }
                for row in rows
            ],
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
