#!/usr/bin/env python3
"""Reproduce the selected 007740 -> 007747 hash23 fine-tuning recipe.

This is deliberately a single-frame production runner, not a sweep.  One
ordinary invocation:

* verifies the canonical leader, target revision, JPEGs, standard frequency
  maps, runtime, source tree, reference renders, and available storage;
* full-evaluates the direct model-only transplant before any target update;
* replays the selected LR=0.015 / 300k decay schedule from the original
  step-91128 hash23 leader with entirely fresh target training state;
* mirrors the accepted process boundaries: direct training through step60752,
  then one full-resume interval at a time;
* saves and full-evaluates every 15188 local steps through step 151880; and
* writes all checkpoints, three-view renders, native hands/chain comparisons,
  exact configs, hashes, startup audit, wall timings, and a compact summary.

The fixed step-151880 horizon reproduces the plateau-selected checkpoint from
the completed v2 campaign.  Manual visual review remains evidence, never a
hard-coded pass: supplying ``--visual-decisions`` can certify the reproduced
target, while an ordinary invocation still completes training and emits the
native review artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    from scripts import run_lookcloser_007747_finetune_v2 as v2
except ImportError:
    import run_lookcloser_007747_finetune_v2 as v2


SCRIPT_PATH = Path(__file__).resolve()

RECIPE_ID = "007747_hash23_lr015_h300_step151880"
ARM_ID = "authoritative-R-L150-H300"
INITIAL_LR = 0.015
FINAL_LR = 0.0001
SCHEDULER_MAX_STEPS = 300_000
TARGET_STEP = 151_880
MAX_NUM_ITERATIONS = TARGET_STEP + 1
SEED = 42

DEFAULT_OUTPUT_ROOT = Path("/mnt/data/lookcloser_007747_finetune_v2_runs")
DEFAULT_VENV = v2.DEFAULT_VENV
DEFAULT_TCNN_OVERLAY = v2.DEFAULT_TCNN_OVERLAY

REFERENCE_CAMPAIGN = Path(
    "/mnt/data/lookcloser_007747_finetune_v2_runs/"
    "hash23_extended_scheduler_seed42_v3"
)
REFERENCE_SUMMARY = REFERENCE_CAMPAIGN / "summary.json"
REFERENCE_CHECKPOINT_SHA256 = (
    "000fbc9144505fe4041d61ba71f0f9f804c78de19517b70cd0584d519ae6a358"
)
REFERENCE_METRICS = {
    "psnr": 29.880142211914062,
    "ssim": 0.6756599545478821,
    "lpips": 0.2145330160856247,
}


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{RECIPE_ID}_{timestamp}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New artifact directory. By default a timestamped directory is "
            "created under /mnt/data/lookcloser_007747_finetune_v2_runs."
        ),
    )
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--tcnn-overlay", type=Path, default=DEFAULT_TCNN_OVERLAY)
    parser.add_argument(
        "--visual-decisions",
        type=Path,
        help=(
            "Optional v2 visual-decision JSON. The key for the selected target "
            f"is {ARM_ID}:{TARGET_STEP}."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse an explicitly supplied output directory.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.resume:
            parser.error("--resume requires an explicit --output-dir")
        args.output_dir = default_output_dir()
    return args


def fixed_arm() -> v2.Arm:
    return v2.Arm(
        arm_id=ARM_ID,
        lr_init=INITIAL_LR,
        scheduler_max_steps=SCHEDULER_MAX_STEPS,
        phase="authoritative",
    )


def initial_segment(args: argparse.Namespace) -> v2.Segment:
    arm = fixed_arm()
    return v2.authoritative_segment(
        args,
        arm,
        target_step=v2.INITIAL_FINAL_STEP,
        parent=None,
    )


def fixed_segments(args: argparse.Namespace) -> list[v2.Segment]:
    """Return the exact process/continuation boundaries of the selected run."""

    arm = fixed_arm()
    result = [initial_segment(args)]
    run_dir = result[0].run_dir
    for target_step in range(
        v2.INITIAL_FINAL_STEP + v2.INTERVAL,
        TARGET_STEP + 1,
        v2.INTERVAL,
    ):
        result.append(
            v2.authoritative_segment(
                args,
                arm,
                target_step=target_step,
                parent=v2.checkpoint_path(run_dir, target_step - v2.INTERVAL),
            )
        )
    if result[-1].target_step != TARGET_STEP:
        raise v2.InfrastructureError(
            "Frozen target step is not aligned with the resume interval"
        )
    return result


def recipe_manifest() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "recipe_id": RECIPE_ID,
        "frame": "007747",
        "seed": SEED,
        "parent_frame": "007740",
        "parent_checkpoint": str(v2.LEADER_CHECKPOINT),
        "parent_checkpoint_sha256": v2.EXPECTED_LEADER_SHA256,
        "checkpoint_load_mode": "model_parameters_only",
        "fresh_target_state": [
            "Adam",
            "scheduler",
            "scaler",
            "RNG",
            "occupancy_grid",
            "frequency_grid",
            "FAS_counter_and_buckets",
            "point_telemetry",
        ],
        "lr_initial": INITIAL_LR,
        "lr_final": FINAL_LR,
        "scheduler": "log-linear exponential decay",
        "scheduler_max_steps": SCHEDULER_MAX_STEPS,
        "scheduler_warmup_steps": 0,
        "target_local_step": TARGET_STEP,
        "max_num_iterations": MAX_NUM_ITERATIONS,
        "eval_and_save_interval": v2.INTERVAL,
        "process_boundaries": [segment.target_step for segment in fixed_segments(
            argparse.Namespace(output_dir=Path("<OUTPUT_DIR>"))
        )],
        "continuation_load_mode": "resume",
        "batch_rays": 4096,
        "mixed_precision": True,
        "log2_hashmap_size": 23,
        "hash_levels": 16,
        "hash_features_per_level": 2,
        "min_res": 16,
        "max_res": 8192,
        "max_res_base": 2048,
        "frequency_maps": str(v2.TARGET_MAPS),
        "fas_strength": 1.0,
        "feature_reweighting_strength": 0.3,
        "fixed_traversal_and_fresh_occupancy_warmup_updates": 4096,
        "fused_adam": False,
        "tcnn_network_jit": False,
        "cached_train_rays": False,
        "cpu_fas_prefetch": False,
        "independent_rng_streams": False,
        "reference_campaign": str(REFERENCE_CAMPAIGN),
        "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
        "reference_metrics": dict(REFERENCE_METRICS),
    }


def deterministic_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    segments = fixed_segments(args)
    described_segments = []
    for segment in segments:
        config, differences = v2.configured_segment(args, segment)
        described_segments.append(
            {
                **asdict(segment),
                "arm": asdict(segment.arm),
                "run_dir": str(segment.run_dir),
                "parent_checkpoint": str(segment.parent_checkpoint),
                "effective": {
                    "max_num_iterations": int(config.max_num_iterations),
                    "checkpoint_load_mode": config.checkpoint_load_mode,
                    "load_optimizers": bool(config.load_optimizers),
                    "load_scheduler": bool(config.load_scheduler),
                    "optimizer_lr": float(
                        config.optimizers["fields"]["optimizer"].lr
                    ),
                    "scheduler_lr_final": float(
                        config.optimizers["fields"]["scheduler"].lr_final
                    ),
                    "scheduler_max_steps": int(
                        config.optimizers["fields"]["scheduler"].max_steps
                    ),
                },
                "config_diff": differences,
            }
        )
    return {
        "schema_version": 1,
        "output_dir": str(args.output_dir),
        "recipe": recipe_manifest(),
        "segments": described_segments,
    }


def reproduction_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    payload = v2.full_preflight(
        args,
        initial_storage=not (
            args.resume and (args.output_dir / "campaign.json").is_file()
        ),
    )
    payload["reproduction"] = {
        "script": str(SCRIPT_PATH),
        "script_sha256": v2.sha256_file(SCRIPT_PATH),
        "recipe": recipe_manifest(),
        "reference_summary": str(REFERENCE_SUMMARY),
        "reference_summary_sha256": (
            v2.sha256_file(REFERENCE_SUMMARY)
            if REFERENCE_SUMMARY.is_file()
            else None
        ),
    }
    return payload


def _same_static_preflight(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    return (
        {key: value for key, value in previous.items() if key != "storage"}
        == {key: value for key, value in current.items() if key != "storage"}
    )


def _target_protocol_state(boundary: v2.Boundary) -> Dict[str, Any]:
    protocol = v2.protocol_payload(boundary)
    manual_verdict = str(protocol["visual_gate"]["verdict"])
    automatic_clean = (
        int(protocol["full_view_serious_count"]) == 0
        and protocol["roi"]["artifact"]["serious"] is False
    )
    return {
        "manual_verdict": manual_verdict,
        "automatic_clean": automatic_clean,
        "certified_pass": manual_verdict == "pass" and automatic_clean,
        "protocol": str(boundary.protocol_json),
        "contact_sheet": str(protocol["contact_sheet"]),
        "full_view_serious_count": int(protocol["full_view_serious_count"]),
        "roi_serious": bool(protocol["roi"]["artifact"]["serious"]),
    }


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    target = summary["target"]
    visual = summary["visual"]
    report = (
        "# 007740 → 007747 selected fine-tuning recipe reproduction\n\n"
        "## Recipe\n\n"
        f"- Initial/final LR: `{INITIAL_LR}` → `{FINAL_LR}`\n"
        f"- Exponential scheduler horizon: `{SCHEDULER_MAX_STEPS}`\n"
        f"- Target local step: `{TARGET_STEP}` "
        f"(`max_num_iterations={MAX_NUM_ITERATIONS}`)\n"
        "- Transfer: direct hash23 `model_parameters_only`; all target "
        "optimizer/grid/RNG/FAS state fresh\n\n"
        "## Result\n\n"
        "| Step | PSNR | SSIM | LPIPS | Numeric gate | Visual certification |\n"
        "|---:|---:|---:|---:|---|---|\n"
        f"| {target['local_step']} | {target['psnr']:.6f} | "
        f"{target['ssim']:.6f} | {target['lpips']:.6f} | "
        f"{'pass' if target['numeric_pass'] else 'fail'} | "
        f"{'pass' if visual['certified_pass'] else visual['manual_verdict']} |\n\n"
        "Evaluation loss is intentionally excluded.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_reproduction(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            json.dumps(
                deterministic_dry_run(args),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    preflight = reproduction_preflight(args)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = v2.CampaignStore(args.output_dir / "campaign.json", resume=args.resume)
    manifest = recipe_manifest()
    previous_recipe = store.data.get("recipe")
    if previous_recipe is not None and previous_recipe != manifest:
        raise v2.InfrastructureError("Frozen reproduction recipe changed on resume")
    previous_preflight = store.data.get("preflight")
    if previous_preflight is not None and not _same_static_preflight(
        previous_preflight, preflight
    ):
        raise v2.InfrastructureError(
            "Reproduction provenance changed since campaign creation"
        )
    store.data.update(
        {
            "runner": str(SCRIPT_PATH),
            "recipe": manifest,
            "preflight": preflight,
            "status": "running",
        }
    )
    store.flush()

    decisions = v2.visual_decisions(args.visual_decisions)
    store.data.setdefault("visual_review_snapshots", []).append(
        {
            "at": v2.utc_now(),
            "source": (
                str(args.visual_decisions) if args.visual_decisions else None
            ),
            "source_sha256": (
                v2.sha256_file(args.visual_decisions)
                if args.visual_decisions
                else None
            ),
            "decisions": decisions,
        }
    )
    store.flush()

    v2.run_baseline(args, store, decisions)
    segments = fixed_segments(args)
    records = [v2.run_segment(args, store, segment) for segment in segments]
    first_record = records[0]
    final_segment = segments[-1]
    boundaries = v2.discover_boundaries(
        ARM_ID,
        final_segment.run_dir,
        decisions,
    )
    target = next(
        (boundary for boundary in boundaries if boundary.local_step == TARGET_STEP),
        None,
    )
    if target is None:
        raise v2.InfrastructureError(
            f"Reproduction did not evaluate target step {TARGET_STEP}"
        )

    visual = _target_protocol_state(target)
    reference_drift = {
        name: float(getattr(target, name)) - value
        for name, value in REFERENCE_METRICS.items()
    }
    elapsed_to_target = (
        target.eval_completed_wall_time_ns
        - int(first_record["started_wall_time_ns"])
    ) / 1e9
    summary = {
        "schema_version": 1,
        "recipe": manifest,
        "target": v2.boundary_dict(target),
        "target_checkpoint_sha256": v2.sha256_file(target.checkpoint),
        "reference_metric_drift": reference_drift,
        "trainer_start_to_target_eval_seconds": elapsed_to_target,
        "visual": visual,
        "boundaries": [v2.boundary_dict(row) for row in boundaries],
        "startup_audit": first_record["worker_result"]["startup_audit"],
        "segments": [
            {
                "segment_id": record["segment_id"],
                "load_mode": record["load_mode"],
                "target_step": record["target_step"],
                "started_wall_time_ns": record["started_wall_time_ns"],
                "trainer_wall_seconds": record["trainer_wall_seconds"],
                "scheduled_eval_seconds_total": record[
                    "scheduled_eval_seconds_total"
                ],
            }
            for record in records
        ],
    }
    v2.atomic_json(args.output_dir / "summary.json", summary)
    _write_report(args.output_dir, summary)
    store.data["summary"] = summary
    if not target.numeric_pass:
        store.data["status"] = "quality_failure"
        store.flush()
        raise v2.FinalQualityFailure(
            "Reproduced target checkpoint missed one or more leader thresholds"
        )
    store.data["status"] = (
        "complete" if visual["certified_pass"] else "complete_visual_review_pending"
    )
    store.flush()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run_reproduction(args)
    except v2.FinalQualityFailure as error:
        print(f"QUALITY_FAILURE: {error}", file=sys.stderr)
        return v2.QUALITY_EXIT
    except (
        v2.InfrastructureError,
        FileNotFoundError,
        KeyError,
        StopIteration,
        ValueError,
    ) as error:
        print(f"INFRASTRUCTURE_ERROR: {error}", file=sys.stderr)
        return v2.INFRASTRUCTURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
