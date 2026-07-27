#!/usr/bin/env python3
"""Run one fixed-recipe temporal LookCloser trajectory for one frame and seed.

Cross-frame startup copies only field/model parameters from the accepted parent
snapshot. Same-frame continuation loads full state. The ordinary invocation
records the no-update transplant and reproduces every process boundary through
local step151880 by default. A controller may lower that initial horizon to a
complete evaluation boundary when the inherited per-frame iteration budget is
shorter. ``--extend-one-interval`` resumes exactly one further15188-step
interval for plateau confirmation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

try:
    from scripts import run_lookcloser_007747_finetune_v2 as v2
    from scripts import temporal_finetune_common as common
except ImportError:
    import run_lookcloser_007747_finetune_v2 as v2
    import temporal_finetune_common as common


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
ARM_ID = "fixed-L150-H300"
QUALITY_EXIT = 2
INFRASTRUCTURE_EXIT = 3


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-frame", required=True, choices=common.TARGET_FRAMES)
    parser.add_argument("--parent-snapshot", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=common.SEEDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--venv", type=Path, default=v2.DEFAULT_VENV)
    parser.add_argument("--tcnn-overlay", type=Path, default=v2.DEFAULT_TCNN_OVERLAY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-one-interval", action="store_true")
    parser.add_argument(
        "--initial-target-step",
        type=int,
        default=common.INITIAL_TARGET_STEP,
        help=(
            "Last complete evaluation boundary for the initial trajectory. "
            "Defaults to 151880 and may only be lowered to enforce the "
            "inherited per-frame iteration budget."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            common.CAMPAIGN_ROOT
            / args.target_frame
            / "seeds"
            / f"seed-{args.seed}"
        )
    if args.extend_one_interval and not args.resume:
        parser.error("--extend-one-interval requires --resume")
    if args.initial_target_step not in common.PROCESS_BOUNDARIES:
        parser.error("--initial-target-step must be a complete evaluation boundary")
    if args.initial_target_step > common.INITIAL_TARGET_STEP:
        parser.error("--initial-target-step cannot exceed 151880")
    return args


def _load_parent_config(args: argparse.Namespace, parent: Mapping[str, Any]) -> Any:
    config_path = Path(str(parent["config"]))
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    checkpoint = Path(str(parent["checkpoint"])).resolve()
    configured_checkpoint = Path(config.load_checkpoint).resolve()
    if configured_checkpoint != checkpoint:
        raise common.InfrastructureError(
            f"Parent config does not resolve its snapshot checkpoint: "
            f"{configured_checkpoint} != {checkpoint}"
        )
    configured_dataset = Path(config.pipeline.datamanager.dataparser.data).resolve()
    expected_dataset = (common.DATA_ROOT / str(parent["frame"])).resolve()
    if configured_dataset != expected_dataset:
        raise common.InfrastructureError(
            f"Parent config dataparser mismatch: {configured_dataset} != {expected_dataset}"
        )
    return config


def configure_v2(args: argparse.Namespace) -> Dict[str, Any]:
    expected_parent = common.previous_frame(args.target_frame)
    canonical_parent = common.DATA_ROOT / expected_parent / "snapshot"
    if args.parent_snapshot.resolve() != canonical_parent.resolve():
        raise common.InfrastructureError(
            f"{args.target_frame} must load only {canonical_parent}, "
            f"not {args.parent_snapshot}"
        )
    parent = common.validate_snapshot_files(
        args.parent_snapshot, expected_frame=expected_parent
    )
    _load_parent_config(args, parent)
    v2.LEADER_CONFIG = Path(str(parent["config"]))
    v2.LEADER_CHECKPOINT = Path(str(parent["checkpoint"]))
    v2.TARGET_DATASET = common.DATA_ROOT / args.target_frame
    v2.TARGET_MAPS = v2.TARGET_DATASET / "lookcloser_frequencies"
    v2.ACTIVE_SEED = args.seed
    return parent


def fixed_arm() -> v2.Arm:
    return v2.Arm(
        arm_id=ARM_ID,
        lr_init=common.INITIAL_LR,
        scheduler_max_steps=common.SCHEDULER_MAX_STEPS,
        phase="authoritative",
    )


def run_dir(args: argparse.Namespace) -> Path:
    return v2.arm_run_dir(args.output_dir, "authoritative", ARM_ID)


def initial_segments(args: argparse.Namespace) -> list[v2.Segment]:
    arm = fixed_arm()
    directory = run_dir(args)
    segments: list[v2.Segment] = []
    parent: Optional[Path] = None
    process_targets = [
        target_step
        for target_step in common.INITIAL_PROCESS_TARGETS
        if target_step <= args.initial_target_step
    ]
    if not process_targets or process_targets[-1] != args.initial_target_step:
        process_targets.append(args.initial_target_step)
    for target_step in process_targets:
        segments.append(
            v2.authoritative_segment(
                args,
                arm,
                target_step=target_step,
                parent=parent,
            )
        )
        parent = v2.checkpoint_path(directory, target_step)
    return segments


def extension_segment(args: argparse.Namespace) -> v2.Segment:
    directory = run_dir(args)
    checkpoints = sorted(directory.glob("nerfstudio_models/step-*.ckpt"))
    if not checkpoints:
        raise common.InfrastructureError(
            f"Cannot extend a trajectory without checkpoints: {directory}"
        )
    latest = max(checkpoints, key=common.checkpoint_step)
    latest_step = common.checkpoint_step(latest)
    if latest_step < common.INITIAL_TARGET_STEP:
        raise common.InfrastructureError(
            f"Cannot tail-extend before step {common.INITIAL_TARGET_STEP}: {latest}"
        )
    return v2.authoritative_segment(
        args,
        fixed_arm(),
        target_step=latest_step + common.INTERVAL,
        parent=latest,
    )


def recipe_manifest(args: argparse.Namespace, parent: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "frame": args.target_frame,
        "seed": args.seed,
        "parent_frame": parent["frame"],
        "parent_snapshot": str(args.parent_snapshot.resolve()),
        "parent_checkpoint": parent["checkpoint"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "checkpoint_load_mode": "model_parameters_only",
        "continuation_load_mode": "resume",
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
        "process_targets_through_151880": list(common.INITIAL_PROCESS_TARGETS),
        "initial_target_step": args.initial_target_step,
        "process_targets": [row.target_step for row in initial_segments(args)],
        "eval_and_save_interval": common.INTERVAL,
        "lr_initial": common.INITIAL_LR,
        "lr_final": common.FINAL_LR,
        "scheduler_max_steps": common.SCHEDULER_MAX_STEPS,
        "batch_rays": 4096,
        "mixed_precision": True,
        "log2_hashmap_size": 23,
        "frequency_maps": "lookcloser_frequencies",
        "fas_strength": 1.0,
        "feature_reweighting_strength": 0.3,
        "fixed_traversal_and_fresh_occupancy_warmup_updates": 4096,
        "fused_adam": False,
        "tcnn_network_jit": False,
        "cached_train_rays": False,
        "cpu_fas_prefetch": False,
        "independent_rng_streams": False,
    }


def git_preflight() -> Dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    if branch != "main":
        raise common.InfrastructureError(f"Temporal campaign requires main, got {branch}")
    if status.strip():
        raise common.InfrastructureError("Temporal campaign requires a clean committed main")
    tracked = (
        SCRIPT_PATH,
        SCRIPT_PATH.with_name("temporal_finetune_common.py"),
        Path(v2.SCRIPT_PATH),
        REPO_ROOT / "nerfstudio" / "engine" / "trainer.py",
        REPO_ROOT / "nerfstudio" / "engine" / "optimizers.py",
        REPO_ROOT / "nerfstudio" / "engine" / "schedulers.py",
        REPO_ROOT / "nerfstudio" / "configs" / "method_configs.py",
        REPO_ROOT / "nerfstudio" / "data" / "datamanagers" / "base_datamanager.py",
        REPO_ROOT / "nerfstudio" / "data" / "dataparsers" / "nerfstudio_dataparser.py",
        REPO_ROOT / "nerfstudio" / "fields" / "lookcloser_field.py",
        REPO_ROOT / "nerfstudio" / "model_components" / "lookcloser_grid.py",
        REPO_ROOT / "nerfstudio" / "models" / "lookcloser.py",
        REPO_ROOT / "nerfstudio" / "pipelines" / "lookcloser_pipeline.py",
        REPO_ROOT / "nerfstudio" / "scripts" / "eval.py",
        REPO_ROOT / "nerfstudio" / "scripts" / "train.py",
        REPO_ROOT / "nerfstudio" / "lookcloser_pixel_sampler.py",
    )
    hashes = {
        str(path.relative_to(REPO_ROOT)): common.sha256_file(path) for path in tracked
    }
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps({"commit": commit, "source": hashes}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "branch": branch,
        "commit": commit,
        "source_sha256": hashes,
        "source_fingerprint": fingerprint,
    }


def preflight(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    *,
    freeze: bool,
) -> Dict[str, Any]:
    dataset = common.DATA_ROOT / args.target_frame
    manifest_path = args.output_dir / "input_manifest.json"
    dataset_manifest = (
        common.freeze_dataset_manifest(args.target_frame, dataset, manifest_path)
        if freeze
        else common.compute_dataset_manifest(args.target_frame, dataset)
    )
    return {
        "schema_version": 2,
        "git": git_preflight(),
        "runtime": v2.runtime_preflight(args),
        "storage": v2.disk_guard(args.output_dir, initial=not args.output_dir.exists()),
        "dataset": dataset_manifest,
        "dataset_manifest": str(manifest_path),
        "parent": dict(parent),
        "recipe": recipe_manifest(args, parent),
    }


def _same_preflight(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    ignored = {"storage"}
    return (
        {key: value for key, value in previous.items() if key not in ignored}
        == {key: value for key, value in current.items() if key not in ignored}
    )


def run_preupdate_baseline(
    args: argparse.Namespace,
    store: v2.CampaignStore,
) -> None:
    existing = store.data.get("baseline")
    if isinstance(existing, Mapping) and existing.get("status") == "complete":
        return
    arm = fixed_arm()
    segment = v2.Segment(
        segment_id="baseline-preupdate",
        arm=arm,
        run_dir=v2.arm_run_dir(args.output_dir, "baseline", "preupdate"),
        target_step=0,
        load_mode="model_parameters_only",
        parent_checkpoint=v2.LEADER_CHECKPOINT,
    )
    common.freeze_dataset_manifest(
        args.target_frame,
        common.DATA_ROOT / args.target_frame,
        args.output_dir / "input_manifest.json",
    )
    config_path, differences = v2.write_segment_config(args, segment)
    result_path = args.output_dir / "worker_results" / "baseline-preupdate.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.venv / "bin" / "python"),
        str(v2.SCRIPT_PATH),
        "--worker-mode",
        "baseline",
        "--worker-config",
        str(config_path),
        "--worker-result",
        str(result_path),
    ]
    record: Dict[str, Any] = {
        "status": "running",
        "started_at": common.utc_now(),
        "command": command,
        "config": str(config_path),
        "config_diff": differences,
    }
    store.data["baseline"] = record
    store.flush()
    log_path = segment.run_dir / "baseline_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=v2.run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        record["status"] = "infrastructure_error"
        record["returncode"] = completed.returncode
        store.flush()
        raise common.InfrastructureError(f"Pre-update evaluation failed; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    record.update(
        {
            "status": "complete",
            "completed_at": common.utc_now(),
            "wall_seconds": time.monotonic() - started,
            "result": result,
        }
    )
    store.data["baseline"] = record
    store.flush()


def deterministic_dry_run(
    args: argparse.Namespace, parent: Mapping[str, Any]
) -> Dict[str, Any]:
    segments = (
        [extension_segment(args)]
        if args.extend_one_interval
        else initial_segments(args)
    )
    described = []
    for segment in segments:
        config, differences = v2.configured_segment(args, segment)
        described.append(
            {
                **asdict(segment),
                "arm": asdict(segment.arm),
                "run_dir": str(segment.run_dir),
                "parent_checkpoint": str(segment.parent_checkpoint),
                "max_num_iterations": int(config.max_num_iterations),
                "seed": int(config.machine.seed),
                "checkpoint_load_mode": config.checkpoint_load_mode,
                "load_optimizers": bool(config.load_optimizers),
                "load_scheduler": bool(config.load_scheduler),
                "config_diff": differences,
            }
        )
    return {
        "schema_version": 2,
        "output_dir": str(args.output_dir),
        "recipe": recipe_manifest(args, parent),
        "segments": described,
    }


def run(args: argparse.Namespace) -> int:
    parent = configure_v2(args)
    if args.dry_run:
        print(
            json.dumps(
                deterministic_dry_run(args, parent),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0
    current_preflight = preflight(args, parent, freeze=not args.preflight_only)
    if args.preflight_only:
        print(json.dumps(current_preflight, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = v2.CampaignStore(args.output_dir / "campaign.json", resume=args.resume)
    previous = store.data.get("preflight")
    if isinstance(previous, Mapping) and not _same_preflight(previous, current_preflight):
        raise common.InfrastructureError("Trajectory provenance changed since creation")
    store.data.update(
        {
            "runner": str(SCRIPT_PATH),
            "frame": args.target_frame,
            "seed": args.seed,
            "parent": dict(parent),
            "recipe": recipe_manifest(args, parent),
            "preflight": current_preflight,
            "status": "running",
        }
    )
    store.data.setdefault("storage_checks", []).append(
        {"at": common.utc_now(), **current_preflight["storage"]}
    )
    store.flush()

    if not args.extend_one_interval:
        run_preupdate_baseline(args, store)
        segments = initial_segments(args)
    else:
        segments = [extension_segment(args)]
    records = []
    for segment in segments:
        common.freeze_dataset_manifest(
            args.target_frame,
            common.DATA_ROOT / args.target_frame,
            args.output_dir / "input_manifest.json",
        )
        records.append(v2.run_segment(args, store, segment))

    boundaries = common.discover_boundaries(args.seed, run_dir(args))
    if not boundaries:
        raise common.InfrastructureError("Trajectory completed without evaluation boundaries")
    summary = {
        "schema_version": 2,
        "frame": args.target_frame,
        "seed": args.seed,
        "parent": dict(parent),
        "boundaries": [common.boundary_payload(row) for row in boundaries],
        "latest_step": max(row.local_step for row in boundaries),
        "latest_checkpoint": str(max(boundaries, key=lambda row: row.local_step).checkpoint),
        "segments": [
            {
                "segment_id": record["segment_id"],
                "load_mode": record["load_mode"],
                "target_step": record["target_step"],
                "trainer_wall_seconds": record["trainer_wall_seconds"],
                "scheduled_eval_seconds_total": record["scheduled_eval_seconds_total"],
            }
            for record in records
        ],
    }
    common.atomic_json(args.output_dir / "summary.json", summary)
    store.data["summary"] = summary
    store.data["status"] = "complete"
    store.flush()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except common.QualityFailure as error:
        print(f"QUALITY_FAILURE: {error}", file=sys.stderr)
        return QUALITY_EXIT
    except (
        common.InfrastructureError,
        v2.InfrastructureError,
        FileNotFoundError,
        KeyError,
        StopIteration,
        ValueError,
    ) as error:
        print(f"INFRASTRUCTURE_ERROR: {error}", file=sys.stderr)
        return INFRASTRUCTURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
