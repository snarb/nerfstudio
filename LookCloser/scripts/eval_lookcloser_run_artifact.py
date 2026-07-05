#!/usr/bin/env python3
"""Evaluate an existing LookCloser run checkpoint and score artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from run_lookcloser_quiet import (
    DEFAULT_ARTIFACT_ROI_CROPS,
    run_artifact_detector,
    run_final_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--eval-label", default="artifact_selection")
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--artifact-detector-preset", choices=("legacy", "significant", "micro"), default="micro")
    parser.add_argument("--artifact-render-name", default="eval_img_0000.png")
    parser.add_argument(
        "--artifact-render-names",
        default="eval_img_0000.png,eval_img_0001.png,eval_img_0002.png",
    )
    parser.add_argument("--artifact-crop-top", type=int, default=0)
    parser.add_argument("--artifact-crop-bottom", type=int, default=0)
    parser.add_argument("--artifact-crop-left", type=int, default=0)
    parser.add_argument("--artifact-crop-right", type=int, default=0)
    parser.add_argument("--artifact-roi-crop-names", default=DEFAULT_ARTIFACT_ROI_CROPS)
    parser.add_argument("--artifact-roi-drop-border-components", type=int, default=0)
    parser.add_argument("--no-artifact-roi-score", dest="artifact_roi_score", action="store_false")
    parser.set_defaults(artifact_score=True, artifact_roi_score=True)
    return parser.parse_args()


def checkpoint_for_step(run_dir: Path, step: int | None) -> Path:
    model_dir = run_dir / "nerfstudio_models"
    if step is None:
        checkpoints = sorted(model_dir.glob("step-*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        return checkpoints[-1]
    checkpoint = model_dir / f"step-{step:09d}.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", checkpoint.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {checkpoint}")
    return int(match.group(1))


def existing_or_run_eval(
    run_dir: Path,
    checkpoint: Path,
    eval_label: str,
    eval_num_rays_per_chunk: int,
) -> dict:
    output_json = run_dir / f"eval_{eval_label}_{checkpoint.stem}.json"
    render_dir = run_dir / f"renders_{eval_label}_{checkpoint.stem}"
    eval_config = run_dir / f"eval_config_step_{checkpoint_step(checkpoint)}.yml"
    log_path = run_dir / "eval_stdout.log"
    if output_json.exists() and render_dir.exists():
        data = json.loads(output_json.read_text(encoding="utf-8"))
        return {
            "checkpoint": data["checkpoint"],
            "results": data["results"],
            "render_dir": str(render_dir),
            "eval_json": str(output_json),
            "eval_log": str(log_path),
            "eval_config": str(eval_config),
            "eval_seconds": 0.0,
            "eval_seconds_note": "reused existing eval json and render directory",
        }
    return run_final_eval(run_dir, checkpoint, eval_label, eval_num_rays_per_chunk)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    checkpoint = checkpoint_for_step(run_dir, args.step)
    start = time.monotonic()

    eval_data = existing_or_run_eval(run_dir, checkpoint, args.eval_label, args.eval_num_rays_per_chunk)
    artifact = run_artifact_detector(run_dir, eval_data, args)
    eval_data["artifact"] = artifact

    summary = {
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_reason": f"eval_only_{args.eval_label}_step_{checkpoint.stem}",
        "train_seconds": 0.0,
        "eval_seconds": eval_data.get("eval_seconds"),
        "artifact_seconds": artifact.get("artifact_seconds") if isinstance(artifact, dict) else None,
        "total_seconds": time.monotonic() - start,
        "eval": eval_data,
        "artifact": artifact,
        "notes": "Eval-only summary for an existing/interpolated checkpoint; no training was run.",
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
