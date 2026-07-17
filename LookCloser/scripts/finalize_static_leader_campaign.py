#!/usr/bin/env python3
"""Record a retried static-leader finalization without changing trained weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from run_static_leader_e2e import (
    FIXED_WARMUP_POINT_SAMPLES,
    LEADER_GATES,
    controller_protocol_fingerprint,
    estimate_legacy_adaptive_point_samples,
    read_eval_trajectory,
    sha256_file,
    write_json,
)
from evaluate_static_leader_candidate import require_completed_detail_result, required_detail_gate


NUMERIC_GATES = LEADER_GATES
EXPECTED_ARTIFACT_VIEWS = 3
EXPECTED_ARTIFACT_ROIS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--accepted-candidate-eval-json", type=Path)
    parser.add_argument("--accepted-candidate-render-dir", type=Path)
    parser.add_argument("--accepted-candidate-artifact-dir", type=Path)
    parser.add_argument("--accepted-candidate-detail-json", type=Path)
    parser.add_argument("--historical-worktree", type=Path)
    parser.add_argument(
        "--allow-legacy-protocol-migration",
        action="store_true",
        help=(
            "Explicitly stamp the current protocol onto a legacy manifest that predates the "
            "controller_protocol_fingerprint field. A recorded mismatch is never overridden."
        ),
    )
    parser.add_argument("--tcnn-build-provenance", type=Path, default=Path("/home/brans/deps/tcnn_2e757_py310/build_provenance.json"))
    return parser.parse_args()


def numeric_summary(eval_json: Path) -> Dict[str, Any]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    results = data["results"]
    metrics = {name: float(results[name]) for name in ("psnr", "ssim", "lpips")}
    return {
        "checkpoint": data["checkpoint"],
        "eval_json": str(eval_json),
        "metrics": metrics,
        "numeric_pass": (
            metrics["psnr"] >= NUMERIC_GATES["psnr"]
            and metrics["ssim"] >= NUMERIC_GATES["ssim"]
            and metrics["lpips"] <= NUMERIC_GATES["lpips"]
        ),
    }


def artifact_summary(artifact_dir: Path) -> Dict[str, Any]:
    pattern = re.compile(
        r"\[candidate\]\s+serious=(\w+)\s+artifact_score=([0-9.]+)\s+count=(\d+)\s+largest=(\d+)px"
    )
    views = []
    for log_path in sorted(artifact_dir.glob("eval_img_*_artifact_stdout.log")):
        match = pattern.search(log_path.read_text(encoding="utf-8"))
        if not match:
            views.append({"log": str(log_path), "status": "unparsed"})
            continue
        views.append(
            {
                "log": str(log_path),
                "status": "complete",
                "serious": match.group(1) == "True",
                "artifact_score": float(match.group(2)),
                "artifact_count": int(match.group(3)),
                "largest_area": int(match.group(4)),
            }
        )
    roi_json = artifact_dir / "roi_scores" / "roi_artifact_scores.json"
    roi_rows = json.loads(roi_json.read_text(encoding="utf-8")) if roi_json.exists() else []
    gate_complete = (
        len(views) == EXPECTED_ARTIFACT_VIEWS
        and all(view.get("status") == "complete" for view in views)
        and len(roi_rows) == EXPECTED_ARTIFACT_ROIS
    )
    return {
        "artifact_dir": str(artifact_dir),
        "views": views,
        "views_scored": sum(view.get("status") == "complete" for view in views),
        "significant_artifact_count": sum(int(view.get("artifact_count", 0)) for view in views),
        "serious_view_count": sum(bool(view.get("serious")) for view in views),
        "roi_json": str(roi_json),
        "roi_count": len(roi_rows),
        "roi_serious_count": sum(bool(row.get("serious")) for row in roi_rows),
        "gate_complete": gate_complete,
    }


def finalized_eval(eval_json: Path, render_dir: Path, artifact_dir: Path) -> Dict[str, Any]:
    result = numeric_summary(eval_json)
    result["render_dir"] = str(render_dir)
    result["artifacts"] = artifact_summary(artifact_dir)
    artifacts = result["artifacts"]
    result["automatic_pass"] = bool(
        result["numeric_pass"]
        and artifacts["gate_complete"]
        and artifacts["views_scored"] == EXPECTED_ARTIFACT_VIEWS
        and artifacts["significant_artifact_count"] == 0
        and artifacts["roi_count"] == EXPECTED_ARTIFACT_ROIS
        and artifacts["roi_serious_count"] == 0
    )
    return result


def finalized_candidate(
    eval_json: Path,
    render_dir: Path,
    artifact_dir: Path,
    detail_json: Optional[Path],
) -> Dict[str, Any]:
    """Build one promotion record and fail closed unless every frozen gate passes."""
    if detail_json is None:
        raise RuntimeError("Accepted candidate requires a complete micro-detail result")
    candidate = finalized_eval(eval_json, render_dir, artifact_dir)
    raw_detail = json.loads(detail_json.read_text(encoding="utf-8"))
    comparison = raw_detail.get("reference_comparison")
    aggregate_pass = comparison.get("pass") if isinstance(comparison, dict) else None
    if not isinstance(aggregate_pass, bool):
        raise RuntimeError("Detail scorer did not produce a valid aggregate result")
    candidate["detail"] = require_completed_detail_result(
        0 if aggregate_pass else 2,
        detail_json,
        expected_render_dir=render_dir,
    )
    candidate["detail_pass"] = required_detail_gate(candidate["detail"])
    candidate["quality_pass"] = bool(
        candidate["numeric_pass"] and candidate["automatic_pass"] and candidate["detail_pass"]
    )
    if not candidate["quality_pass"]:
        raise RuntimeError("Accepted candidate must pass numeric, artifact, and micro-detail gates")
    return candidate


def bind_or_require_campaign_protocol(
    manifest: Dict[str, Any],
    allow_legacy_protocol_migration: bool,
) -> None:
    """Fail closed on protocol drift; permit only an explicit migration from an absent legacy field."""
    current_protocol, current_protocol_sources = controller_protocol_fingerprint()
    recorded_protocol = manifest.get("controller_protocol_fingerprint")
    if recorded_protocol is None:
        if not allow_legacy_protocol_migration:
            raise RuntimeError(
                "Legacy campaign has no controller protocol fingerprint; "
                "review it and pass --allow-legacy-protocol-migration explicitly"
            )
        manifest["protocol_migration"] = {
            "kind": "explicit_legacy_manifest_migration",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "from": None,
            "to": current_protocol,
        }
        manifest["controller_protocol_fingerprint"] = current_protocol
        manifest["controller_protocol_source_sha256"] = current_protocol_sources
        return
    if recorded_protocol != current_protocol:
        raise RuntimeError(
            "Campaign protocol fingerprint differs from the current frozen protocol; "
            "refreeze through a reviewed migration, not retry finalization"
        )


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.campaign.read_text(encoding="utf-8"))
    bind_or_require_campaign_protocol(manifest, args.allow_legacy_protocol_migration)
    recorded_worktree = Path(manifest["historical_worktree"])
    historical_worktree = args.historical_worktree or recorded_worktree
    if historical_worktree.resolve() != recorded_worktree.resolve():
        raise RuntimeError("Finalization worktree must match the campaign-recorded worktree")
    stage = manifest["stage_a_fw03"]
    run_path = Path(stage["run_path"])
    checkpoint = Path(stage["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    original_returncode = int(stage.get("returncode", 1))
    stage["checkpoint_sha256"] = sha256_file(checkpoint)
    stage["trajectory"] = read_eval_trajectory(run_path / "metrics_compact.csv")
    continuation_points = estimate_legacy_adaptive_point_samples(run_path / "metrics_compact.csv")
    stage["estimated_adaptive_point_samples_legacy"] = continuation_points
    stage["estimated_total_point_samples"] = continuation_points
    stage["post_training_wrapper_returncode"] = original_returncode

    final_eval = finalized_eval(args.eval_json, args.render_dir, args.artifact_dir)
    stage["finalization_retry"] = {
        "status": "complete",
        "reason": "PyTorch 2.6 weights_only default in historical eval_utils",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "eval": final_eval,
    }

    stage_a_points = int(manifest["stage_a"]["estimated_adaptive_point_samples_legacy"])
    manifest["total_estimated_adaptive_point_samples_legacy"] = stage_a_points + continuation_points
    manifest["total_estimated_point_samples"] = (
        manifest["total_estimated_adaptive_point_samples_legacy"] + FIXED_WARMUP_POINT_SAMPLES
    )

    if args.accepted_candidate_eval_json is not None:
        required = (args.accepted_candidate_render_dir, args.accepted_candidate_artifact_dir)
        if any(value is None for value in required):
            raise ValueError("accepted candidate requires render and artifact directories")
        candidate = finalized_candidate(
            args.accepted_candidate_eval_json,
            args.accepted_candidate_render_dir,
            args.accepted_candidate_artifact_dir,
            args.accepted_candidate_detail_json,
        )
        candidate_checkpoint = Path(candidate["checkpoint"])
        ancestry_run_paths = [
            Path(manifest[name]["run_path"]).resolve() for name in ("stage_a", "stage_a_fw03")
        ]
        if not candidate_checkpoint.is_file():
            raise FileNotFoundError(candidate_checkpoint)
        if not any(path in candidate_checkpoint.resolve().parents for path in ancestry_run_paths):
            raise RuntimeError("Accepted candidate checkpoint is outside campaign ancestry")
        candidate["checkpoint_sha256"] = sha256_file(candidate_checkpoint)
        manifest["accepted_candidate"] = candidate

    patch_paths = [
        "nerfstudio/data/utils/data_utils.py",
        "nerfstudio/engine/trainer.py",
        "nerfstudio/utils/eval_utils.py",
    ]
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *patch_paths], cwd=historical_worktree
    )
    manifest["finalization_compatibility_patch_paths"] = patch_paths
    manifest["finalization_compatibility_patch_sha256"] = hashlib.sha256(diff).hexdigest()
    if args.tcnn_build_provenance.exists():
        manifest["tiny_cuda_nn"]["build_provenance"] = json.loads(
            args.tcnn_build_provenance.read_text(encoding="utf-8")
        )
    manifest["status"] = "complete_after_finalization_retry"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.campaign, manifest)
    print(json.dumps({"campaign": str(args.campaign), "status": manifest["status"], "final": final_eval}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
