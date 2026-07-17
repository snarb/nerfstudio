#!/usr/bin/env python3
"""Record the first fully evaluated static-leader candidate without changing campaign status."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from finalize_static_leader_campaign import finalized_eval
from evaluate_static_leader_candidate import require_completed_detail_result, required_detail_gate
from run_static_leader_e2e import checkpoint_file_identity, sha256_file, write_json


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--detail-json", type=Path)
    return parser.parse_args()


def checkpoint_step(path: str) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", path)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def campaign_checkpoint_sha256(manifest: dict, checkpoint: Path) -> str:
    """Reuse the controller's hash only while it remains bound to the exact same local file."""
    resolved_checkpoint = checkpoint.resolve()
    matches = []
    for stage_name in ("stage_a", "stage_a_fw03"):
        stage = manifest.get(stage_name)
        if not isinstance(stage, dict) or "checkpoint" not in stage:
            continue
        if Path(stage["checkpoint"]).resolve() == resolved_checkpoint:
            matches.append((stage_name, stage))
    if not matches:
        # Non-final intermediate candidates do not have a controller-recorded
        # identity, so retain the original full-file hash.
        return sha256_file(checkpoint)
    if len(matches) != 1:
        raise RuntimeError(f"Candidate checkpoint ambiguously matches {len(matches)} campaign stages")

    stage_name, stage = matches[0]
    if stage.get("returncode") != 0:
        raise RuntimeError(f"Campaign stage {stage_name} did not complete successfully")
    if stage.get("target_step") != checkpoint_step(str(checkpoint)):
        raise RuntimeError(f"Campaign target step does not match candidate checkpoint for {stage_name}")
    recorded_hash = stage.get("checkpoint_sha256")
    recorded_identity = stage.get("checkpoint_file_identity")
    if not isinstance(recorded_hash, str) or SHA256_PATTERN.fullmatch(recorded_hash) is None:
        raise RuntimeError(f"Campaign has an invalid checkpoint hash for {stage_name}")
    if not isinstance(recorded_identity, dict):
        raise RuntimeError(f"Campaign has incomplete checkpoint identity for {stage_name}")
    actual_identity = checkpoint_file_identity(checkpoint)
    if recorded_identity != actual_identity:
        raise RuntimeError(
            f"Candidate checkpoint changed after the campaign hash was computed: {checkpoint}"
        )
    return recorded_hash


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.campaign.read_text(encoding="utf-8"))
    candidate = finalized_eval(args.eval_json, args.render_dir, args.artifact_dir)
    checkpoint = Path(candidate["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    ancestry_run_paths = [
        Path(manifest[stage]["run_path"]).resolve()
        for stage in ("stage_a", "stage_a_fw03")
    ]
    if not any(run_path in checkpoint.resolve().parents for run_path in ancestry_run_paths):
        raise ValueError(f"Candidate checkpoint is outside campaign ancestry: {checkpoint}")
    candidate["checkpoint_sha256"] = campaign_checkpoint_sha256(manifest, checkpoint)
    if args.detail_json is None:
        raise RuntimeError("Candidate must have a complete micro-detail result before recording")
    raw_detail = json.loads(args.detail_json.read_text(encoding="utf-8"))
    comparison = raw_detail.get("reference_comparison")
    aggregate_pass = comparison.get("pass") if isinstance(comparison, dict) else None
    if not isinstance(aggregate_pass, bool):
        raise RuntimeError("Detail scorer did not produce a valid aggregate result")
    candidate["detail"] = require_completed_detail_result(
        0 if aggregate_pass else 2,
        args.detail_json,
        expected_render_dir=args.render_dir,
    )
    candidate["detail_pass"] = required_detail_gate(candidate["detail"])
    candidate["quality_pass"] = bool(
        candidate["numeric_pass"] and candidate["automatic_pass"] and candidate["detail_pass"]
    )
    if not candidate["quality_pass"]:
        raise RuntimeError("Candidate must pass numeric, artifact, and micro-detail gates before recording")
    prior = manifest.get("accepted_candidate")
    if prior is not None and checkpoint_step(prior["checkpoint"]) <= checkpoint_step(candidate["checkpoint"]):
        raise RuntimeError("Manifest already contains an earlier or equal accepted checkpoint")
    manifest["accepted_candidate"] = candidate
    write_json(args.campaign, manifest)
    print(json.dumps(candidate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
