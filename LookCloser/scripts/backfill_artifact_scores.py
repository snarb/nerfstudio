#!/usr/bin/env python3
"""Backfill full-frame and curated ROI artifact scores into run_summary.json.

This is intentionally a no-training utility: it uses already rendered eval
panels from a LookCloser run and rewrites only artifact-related summary fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

from run_lookcloser_quiet import DEFAULT_ARTIFACT_ROI_CROPS, run_artifact_detector


DEFAULT_RENDER_NAMES = "eval_img_0000.png,eval_img_0001.png,eval_img_0002.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="LookCloser run directory.")
    parser.add_argument(
        "--render-dir",
        action="append",
        default=None,
        help="Optional RUN_DIR=RENDER_DIR override. Repeatable.",
    )
    parser.add_argument("--artifact-render-names", default=DEFAULT_RENDER_NAMES)
    parser.add_argument("--artifact-crop-top", type=int, default=0)
    parser.add_argument("--artifact-crop-bottom", type=int, default=0)
    parser.add_argument("--artifact-crop-left", type=int, default=0)
    parser.add_argument("--artifact-crop-right", type=int, default=0)
    parser.add_argument("--artifact-detector-preset", choices=("legacy", "significant"), default="legacy")
    parser.add_argument("--artifact-roi-drop-border-components", type=int, default=0)
    parser.add_argument("--artifact-roi-crop-names", default=DEFAULT_ARTIFACT_ROI_CROPS)
    parser.add_argument("--no-artifact-roi-score", dest="artifact_roi_score", action="store_false")
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.set_defaults(artifact_roi_score=True, backup=True)
    return parser.parse_args()


def parse_render_overrides(values: Optional[Iterable[str]]) -> Dict[Path, Path]:
    overrides: Dict[Path, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected RUN_DIR=RENDER_DIR, got {value!r}")
        left, right = value.split("=", 1)
        overrides[Path(left).resolve()] = Path(right)
    return overrides


def infer_render_dir(run_dir: Path, summary: Dict[str, object], overrides: Dict[Path, Path]) -> Path:
    override = overrides.get(run_dir.resolve())
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(override)
        return override
    eval_data = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    render_dir = eval_data.get("render_dir") if isinstance(eval_data, dict) else None
    if render_dir and Path(str(render_dir)).exists():
        return Path(str(render_dir))
    candidates = sorted(run_dir.glob("renders_*step-*"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Could not infer render dir for {run_dir}")


def detector_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_score=True,
        artifact_render_name="eval_img_0000.png",
        artifact_render_names=args.artifact_render_names,
        artifact_crop_top=args.artifact_crop_top,
        artifact_crop_bottom=args.artifact_crop_bottom,
        artifact_crop_left=args.artifact_crop_left,
        artifact_crop_right=args.artifact_crop_right,
        artifact_detector_preset=args.artifact_detector_preset,
        artifact_roi_score=args.artifact_roi_score,
        artifact_roi_drop_border_components=args.artifact_roi_drop_border_components,
        artifact_roi_crop_names=args.artifact_roi_crop_names,
    )


def selected_checkpoint(summary: Dict[str, object]) -> Optional[str]:
    checkpoint = summary.get("selected_checkpoint")
    if checkpoint:
        return str(checkpoint)
    eval_data = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    checkpoint = eval_data.get("checkpoint") if isinstance(eval_data, dict) else None
    return str(checkpoint) if checkpoint else None


def update_seconds(summary: Dict[str, object], artifact: Dict[str, object]) -> None:
    artifact_seconds = artifact.get("artifact_seconds")
    summary["artifact_seconds"] = artifact_seconds
    eval_data = summary.get("eval")
    if isinstance(eval_data, dict):
        eval_seconds = eval_data.get("eval_seconds") or summary.get("eval_seconds")
        if eval_seconds is not None:
            summary["eval_seconds"] = eval_seconds
    train_seconds = summary.get("train_seconds")
    eval_seconds = summary.get("eval_seconds")
    if train_seconds is not None and eval_seconds is not None and artifact_seconds is not None:
        summary["total_seconds"] = float(train_seconds) + float(eval_seconds) + float(artifact_seconds)


def backfill_one(run_dir: Path, args: argparse.Namespace, overrides: Dict[Path, Path]) -> Dict[str, object]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    render_dir = infer_render_dir(run_dir, summary, overrides)
    eval_data = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    eval_payload = dict(eval_data)
    eval_payload["render_dir"] = str(render_dir)
    checkpoint = selected_checkpoint(summary)
    if checkpoint is not None:
        eval_payload.setdefault("checkpoint", checkpoint)

    previous_artifact_present = summary.get("artifact") is not None or eval_payload.get("artifact") is not None
    artifact = run_artifact_detector(run_dir, eval_payload, detector_args(args))
    eval_payload["artifact"] = artifact
    summary["eval"] = eval_payload
    summary["artifact"] = artifact
    update_seconds(summary, artifact)
    summary["artifact_backfill"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "render_dir": str(render_dir),
        "artifact_render_names": [name.strip() for name in args.artifact_render_names.split(",") if name.strip()],
        "artifact_roi_crop_names": args.artifact_roi_crop_names,
        "artifact_detector_preset": args.artifact_detector_preset,
        "artifact_roi_drop_border_components": args.artifact_roi_drop_border_components,
        "previous_artifact_present": bool(previous_artifact_present),
        "script": str(Path(__file__).resolve()),
    }

    if args.backup:
        backup_path = summary_path.with_name(f"{summary_path.name}.bak-artifact-{int(time.time())}")
        shutil.copy2(summary_path, backup_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "artifact_score": artifact.get("artifact_score"),
        "serious_artifact_score": artifact.get("serious_artifact_score"),
        "roi_artifact_score": (artifact.get("roi") or {}).get("roi_artifact_score"),
        "roi_serious_artifact_score": (artifact.get("roi") or {}).get("roi_serious_artifact_score"),
        "roi_serious_count": (artifact.get("roi") or {}).get("roi_serious_count"),
        "stand_connector_score": (artifact.get("roi") or {}).get("stand_connector_score"),
        "artifact_seconds": artifact.get("artifact_seconds"),
    }


def main() -> int:
    args = parse_args()
    overrides = parse_render_overrides(args.render_dir)
    rows = []
    for run_dir in args.run_dir:
        rows.append(backfill_one(run_dir, args, overrides))
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
