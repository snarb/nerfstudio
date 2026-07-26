#!/usr/bin/env python3
"""Sequentially train, review, promote, and validate the temporal frame chain."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

try:
    from scripts import run_lookcloser_007747_finetune_v2 as v2
    from scripts import run_lookcloser_temporal_finetune as trajectory
    from scripts import temporal_finetune_common as common
except ImportError:
    import run_lookcloser_007747_finetune_v2 as v2
    import run_lookcloser_temporal_finetune as trajectory
    import temporal_finetune_common as common


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
MNT_RESERVE_BYTES = 180 * 1024**3
ROOT_RESERVE_BYTES = 20 * 1024**3
VRAM_PER_JOB_MIB = 20 * 1024
VRAM_RESERVE_MIB = 20 * 1024
QUALITY_EXIT = 2
INFRASTRUCTURE_EXIT = 3


class CampaignStore:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        if path.is_file():
            if not resume:
                raise common.InfrastructureError(f"Campaign exists; use --resume: {path}")
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 2,
                "created_at": common.utc_now(),
                "status": "initialized",
                "frames": {},
            }
            self.flush()

    def flush(self) -> None:
        self.data["updated_at"] = common.utc_now()
        common.atomic_json(self.path, self.data)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=common.CAMPAIGN_ROOT)
    parser.add_argument("--start-frame", choices=common.TARGET_FRAMES, default=common.TARGET_FRAMES[0])
    parser.add_argument("--end-frame", choices=common.TARGET_FRAMES, default=common.TARGET_FRAMES[-1])
    parser.add_argument("--visual-decisions", type=Path)
    parser.add_argument("--venv", type=Path, default=v2.DEFAULT_VENV)
    parser.add_argument("--tcnn-overlay", type=Path, default=v2.DEFAULT_TCNN_OVERLAY)
    parser.add_argument(
        "--trajectory-script",
        type=Path,
        default=trajectory.SCRIPT_PATH,
        help=(
            "Frozen temporal trajectory runner. Existing trajectories must keep "
            "using the exact source revision recorded by their preflight."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--initial-seeds",
        type=int,
        nargs="+",
        choices=common.SEEDS,
        default=(43,),
        help=(
            "Independent initial trajectories. The user-authorized fast default "
            "starts seed 43 alone; add 42 and/or 44 for more candidate diversity."
        ),
    )
    args = parser.parse_args(argv)
    if common.frame_index(args.start_frame) > common.frame_index(args.end_frame):
        parser.error("--start-frame must not follow --end-frame")
    if args.visual_decisions is None:
        args.visual_decisions = args.campaign_root / "visual_decisions.json"
    args.initial_seeds = tuple(dict.fromkeys(args.initial_seeds))
    return args


def selected_frames(args: argparse.Namespace) -> tuple[str, ...]:
    start = common.frame_index(args.start_frame)
    end = common.frame_index(args.end_frame)
    return common.FRAME_NAMES[start : end + 1]


def frame_root(args: argparse.Namespace, frame: str) -> Path:
    return args.campaign_root / frame


def attempt_dir(args: argparse.Namespace, frame: str, seed: int, attempt: int) -> Path:
    return frame_root(args, frame) / "runs" / f"seed-{seed}-attempt-{attempt:02d}"


def authorized_tail_seeds(
    args: argparse.Namespace, candidates: Sequence[int]
) -> tuple[int, ...]:
    allowed = set(args.initial_seeds)
    return tuple(seed for seed in candidates if seed in allowed)


def _disk_usage_anchor(path: Path) -> shutil._ntuple_diskusage:
    anchor = path
    while not anchor.exists():
        anchor = anchor.parent
    return shutil.disk_usage(anchor)


def storage_preflight(
    args: argparse.Namespace,
    frame: str,
    parent_checkpoint: Path,
    jobs: int,
) -> Dict[str, Any]:
    checkpoint_size = parent_checkpoint.stat().st_size
    remaining = common.frame_index(args.end_frame) - common.frame_index(frame) + 1
    root_usage = _disk_usage_anchor(common.DATA_ROOT)
    root_forecast = remaining * checkpoint_size
    if root_usage.free - root_forecast < ROOT_RESERVE_BYTES:
        raise common.InfrastructureError(
            f"Projected root free space is too low: {root_usage.free / 1024**3:.1f} GiB "
            f"free, {root_forecast / 1024**3:.1f} GiB remaining snapshot forecast"
        )
    mnt_usage = _disk_usage_anchor(args.campaign_root)
    current_frame_forecast = jobs * len(common.PROCESS_BOUNDARIES) * checkpoint_size
    if mnt_usage.free - current_frame_forecast < MNT_RESERVE_BYTES:
        raise common.InfrastructureError(
            f"Projected /mnt/data free space is too low: {mnt_usage.free / 1024**3:.1f} GiB "
            f"free, {current_frame_forecast / 1024**3:.1f} GiB frame forecast"
        )
    return {
        "at": common.utc_now(),
        "frame": frame,
        "checkpoint_size": checkpoint_size,
        "remaining_frames": remaining,
        "root": {
            "free": root_usage.free,
            "forecast": root_forecast,
            "reserve": ROOT_RESERVE_BYTES,
            "projected_free": root_usage.free - root_forecast,
        },
        "mnt": {
            "free": mnt_usage.free,
            "forecast": current_frame_forecast,
            "reserve": MNT_RESERVE_BYTES,
            "projected_free": mnt_usage.free - current_frame_forecast,
        },
    }


def gpu_preflight(args: argparse.Namespace, requested: int = 3) -> Dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        env=v2.run_environment(args),
        text=True,
    ).strip()
    devices = []
    for line in output.splitlines():
        index, name, free, utilization = [part.strip() for part in line.split(",", 3)]
        devices.append(
            {
                "index": int(index),
                "name": name,
                "free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    if not devices:
        raise common.InfrastructureError("nvidia-smi returned no GPUs")
    best = max(devices, key=lambda row: row["free_mib"])
    selected = requested
    reason = "requested capacity fits"
    if requested == 3 and (
        best["free_mib"] < 3 * VRAM_PER_JOB_MIB + VRAM_RESERVE_MIB
        or best["utilization_percent"] >= 98
    ):
        selected = 2
        reason = "three-job VRAM/utilization preflight failed"
    if selected >= 1 and (
        best["free_mib"] < selected * VRAM_PER_JOB_MIB + VRAM_RESERVE_MIB
        or best["utilization_percent"] >= 98
    ):
        raise common.InfrastructureError(
            f"GPU preflight cannot support {selected} requested initial run(s)"
        )
    return {
        "at": common.utc_now(),
        "requested": requested,
        "selected": selected,
        "reason": reason,
        "devices": devices,
        "vram_per_job_mib": VRAM_PER_JOB_MIB,
        "reserve_mib": VRAM_RESERVE_MIB,
    }


def _trajectory_command(
    args: argparse.Namespace,
    *,
    frame: str,
    parent_snapshot: Path,
    seed: int,
    output_dir: Path,
    resume: bool,
    extend: bool = False,
) -> list[str]:
    command = [
        str(args.venv / "bin" / "python"),
        str(args.trajectory_script),
        "--target-frame",
        frame,
        "--parent-snapshot",
        str(parent_snapshot),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--venv",
        str(args.venv),
        "--tcnn-overlay",
        str(args.tcnn_overlay),
    ]
    if resume:
        command.append("--resume")
    if extend:
        command.append("--extend-one-interval")
    return command


def _run_one_trajectory(
    args: argparse.Namespace,
    *,
    frame: str,
    parent_snapshot: Path,
    seed: int,
    output_dir: Path,
    extend: bool,
) -> Dict[str, Any]:
    resume = (output_dir / "campaign.json").is_file()
    command = _trajectory_command(
        args,
        frame=frame,
        parent_snapshot=parent_snapshot,
        seed=seed,
        output_dir=output_dir,
        resume=resume,
        extend=extend,
    )
    log_dir = frame_root(args, frame) / "controller_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "extend" if extend else "initial"
    log_path = log_dir / f"seed-{seed}_{suffix}_{common.utc_now().replace(':', '-')}.log"
    started_ns = time.time_ns()
    started = time.monotonic()
    heartbeat_path = log_dir / f"seed-{seed}_{suffix}_hourly_checks.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=v2.run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                returncode = process.wait(timeout=3600)
                break
            except subprocess.TimeoutExpired:
                heartbeat = {
                    "at": common.utc_now(),
                    "frame": frame,
                    "seed": seed,
                    "phase": suffix,
                    "pid": process.pid,
                    "alive": process.poll() is None,
                    "elapsed_seconds": time.monotonic() - started,
                    "controller_log": str(log_path),
                    "controller_log_bytes": log_path.stat().st_size,
                }
                with heartbeat_path.open("a", encoding="utf-8") as heartbeat_log:
                    heartbeat_log.write(json.dumps(heartbeat, sort_keys=True) + "\n")
                print(
                    f"frame={frame} phase={suffix} seed={seed} "
                    f"hourly_check={heartbeat['elapsed_seconds'] / 3600:.1f}h "
                    f"alive={heartbeat['alive']}",
                    flush=True,
                )
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "command": command,
        "log": str(log_path),
        "returncode": returncode,
        "hourly_checks": str(heartbeat_path),
        "started_wall_time_ns": started_ns,
        "wall_seconds": time.monotonic() - started,
    }


def _contains_oom(output_dir: Path, controller_log: Path) -> bool:
    patterns = ("CUDA out of memory", "torch.OutOfMemoryError", "OutOfMemoryError")
    paths = [controller_log, *output_dir.rglob("*.log")]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(pattern in text for pattern in patterns):
            return True
    return False


def run_wave(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    parent_snapshot: Path,
    seeds: Sequence[int],
    active_runs: Mapping[str, str],
    extend: bool,
) -> list[Dict[str, Any]]:
    if not seeds:
        raise common.InfrastructureError("Cannot launch an empty trajectory wave")
    print(
        f"frame={frame} phase={'tail' if extend else 'initial'} "
        f"seeds={','.join(str(seed) for seed in seeds)} started",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(seeds), thread_name_prefix=f"temporal-{frame}") as executor:
        futures = {
            executor.submit(
                _run_one_trajectory,
                args,
                frame=frame,
                parent_snapshot=parent_snapshot,
                seed=seed,
                output_dir=Path(active_runs[str(seed)]),
                extend=extend,
            ): seed
            for seed in seeds
        }
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda row: int(row["seed"]))
    store.data.setdefault("waves", []).append(
        {
            "at": common.utc_now(),
            "frame": frame,
            "phase": "tail" if extend else "initial",
            "seeds": list(seeds),
            "results": results,
        }
    )
    store.flush()
    failures = [row for row in results if int(row["returncode"]) != 0]
    if failures:
        oom = [
            row
            for row in failures
            if _contains_oom(Path(row["output_dir"]), Path(row["log"]))
        ]
        if oom and not extend and len(seeds) == 3:
            raise RuntimeError("THREE_JOB_OOM")
        raise common.InfrastructureError(
            "Trajectory wave failed: "
            + ", ".join(f"seed {row['seed']} log={row['log']}" for row in failures)
        )
    print(
        f"frame={frame} phase={'tail' if extend else 'initial'} complete "
        + " ".join(f"seed{row['seed']}={row['wall_seconds'] / 3600:.2f}h" for row in results),
        flush=True,
    )
    return results


def _active_boundaries(
    frame_record: Mapping[str, Any],
) -> Dict[int, list[common.Boundary]]:
    result: Dict[int, list[common.Boundary]] = {}
    for seed_text, path in frame_record["active_runs"].items():
        seed = int(seed_text)
        directory = trajectory.v2.arm_run_dir(Path(path), "authoritative", trajectory.ARM_ID)
        result[seed] = common.discover_boundaries(seed, directory)
    return result


def build_frame_comparisons(
    args: argparse.Namespace,
    frame: str,
    boundaries: Mapping[int, Sequence[common.Boundary]],
) -> list[Dict[str, Any]]:
    leader = common.DATA_ROOT / "007740" / "render" / "eval_img_0000.png"
    previous = (
        common.DATA_ROOT
        / common.previous_frame(frame)
        / "render"
        / "eval_img_0000.png"
    )
    comparisons = []
    for seed, rows in sorted(boundaries.items()):
        ordered = sorted(rows, key=lambda row: row.local_step)
        prior: Optional[common.Boundary] = None
        for boundary in ordered:
            output_dir = (
                frame_root(args, frame)
                / "comparisons"
                / f"seed-{seed}"
                / f"step-{boundary.local_step:09d}"
            )
            comparison_json = output_dir / "comparison.json"
            expected_render = boundary.render_dir / "eval_img_0000.png"
            if comparison_json.is_file():
                payload = json.loads(comparison_json.read_text(encoding="utf-8"))
                sources = payload.get("sources", {})
                target_rows = [
                    value
                    for label, value in sources.items()
                    if str(label).startswith("target ")
                ]
                if (
                    len(target_rows) != 1
                    or target_rows[0].get("source_sha256")
                    != common.sha256_file(expected_render)
                ):
                    raise common.InfrastructureError(
                        f"Existing visual comparison changed: {comparison_json}"
                    )
            else:
                payload = common.build_native_comparison(
                    frame=frame,
                    seed=seed,
                    step=boundary.local_step,
                    target_render=expected_render,
                    previous_accepted_render=previous,
                    leader_render=leader,
                    output_dir=output_dir,
                    previous_boundary_render=(
                        prior.render_dir / "eval_img_0000.png" if prior is not None else None
                    ),
                )
            comparisons.append(payload)
            prior = boundary
    common.atomic_json(
        frame_root(args, frame) / "comparison_index.json",
        {"schema_version": 1, "frame": frame, "comparisons": comparisons},
    )
    return comparisons


def ensure_visual_decision_template(
    path: Path,
    frame: str,
    boundaries: Iterable[common.Boundary],
) -> Dict[str, Dict[str, str]]:
    decisions = common.load_visual_decisions(path)
    changed = False
    for boundary in boundaries:
        key = common.visual_key(frame, boundary.seed, boundary.local_step)
        if key not in decisions:
            decisions[key] = {
                "verdict": "pending",
                "change_from_previous": (
                    "not_applicable"
                    if boundary.local_step == common.INTERVAL
                    else "pending"
                ),
                "note": "",
            }
            changed = True
    if changed:
        # "pending" is permitted only as a template value.  The parser rejects
        # it for change_from_previous, so normalize that field to not_applicable
        # until the reviewer records a decision.
        writable = {
            key: {
                **value,
                "change_from_previous": (
                    "not_applicable"
                    if value["change_from_previous"] == "pending"
                    else value["change_from_previous"]
                ),
            }
            for key, value in decisions.items()
        }
        common.atomic_json(path, writable)
        decisions = common.load_visual_decisions(path)
    return decisions


def pending_visual_keys(
    frame: str,
    boundaries: Iterable[common.Boundary],
    decisions: Mapping[str, Mapping[str, str]],
) -> list[str]:
    return [
        common.visual_key(frame, row.seed, row.local_step)
        for row in boundaries
        if common.decision_for(decisions, frame, row)["verdict"] == "pending"
    ]


def _run_ns_eval(
    args: argparse.Namespace,
    *,
    config_path: Path,
    validation_path: Path,
    render_dir: Path,
    log_path: Path,
) -> Dict[str, Any]:
    command = [
        str(args.venv / "bin" / "ns-eval"),
        "--load-config",
        str(config_path),
        "--output-path",
        str(validation_path),
        "--render-output-path",
        str(render_dir),
    ]
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
    if completed.returncode != 0 or not validation_path.is_file():
        raise common.InfrastructureError(f"Fresh snapshot ns-eval failed; see {log_path}")
    renders = sorted(render_dir.glob("eval_img_*.png"))
    if len(renders) != 3:
        raise common.InfrastructureError(
            f"Fresh snapshot ns-eval produced {len(renders)} renders, expected 3"
        )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    return {
        "command": command,
        "wall_seconds": time.monotonic() - started,
        "results": {
            name: float(payload["results"][name]) for name in ("psnr", "ssim", "lpips")
        },
        "renders": [str(path) for path in renders],
        "render_sha256": {path.name: common.sha256_file(path) for path in renders},
        "validation_sha256": common.sha256_file(validation_path),
    }


def _write_snapshot_metadata(
    *,
    frame: str,
    selected: common.Boundary,
    parent_info: Mapping[str, Any],
    snapshot: Path,
    source_config: Path,
    source_config_sha256: str,
    fresh_eval: Mapping[str, Any],
    final_render: Path,
    final_decision: Mapping[str, str],
    campaign_frame_root: Path,
    source_git: Mapping[str, Any],
) -> None:
    selection = {
        "schema_version": 2,
        "frame": frame,
        "seed": selected.seed,
        "parent_frame": parent_info["frame"],
        "parent_checkpoint": parent_info["checkpoint"],
        "selected_step": selected.local_step,
        "selection": "maximum_psnr_then_minimum_lpips_within_inclusive_0.07_db_after_visual_and_hard_gates",
        "metrics": dict(fresh_eval["results"]),
        "quality_tier": (
            "preferred"
            if (
                float(fresh_eval["results"]["psnr"]) >= common.PREFERRED_PSNR
                and float(fresh_eval["results"]["ssim"]) >= common.PREFERRED_SSIM
                and float(fresh_eval["results"]["lpips"]) <= common.PREFERRED_LPIPS
            )
            else "hard_minimum_after_confirmed_plateau"
        ),
        "preferred_target": {
            "psnr_min": common.PREFERRED_PSNR,
            "ssim_min": common.PREFERRED_SSIM,
            "lpips_max": common.PREFERRED_LPIPS,
        },
        "visual_gate": {
            "verdict": final_decision["verdict"],
            "crop": list(common.CROP_BOX),
            "note": final_decision.get("note", ""),
        },
        "render": str(final_render),
    }
    common.atomic_json(snapshot / "selection.json", selection)
    checkpoint = common.snapshot_checkpoint(snapshot)
    provenance = {
        "schema_version": 2,
        "frame": frame,
        "dataset": str((common.DATA_ROOT / frame).resolve()),
        "dataset_manifest": str(campaign_frame_root / "input_manifest.json"),
        "dataset_manifest_sha256": common.sha256_file(
            campaign_frame_root / "input_manifest.json"
        ),
        "parent_snapshot": parent_info["snapshot"],
        "parent_checkpoint": parent_info["checkpoint"],
        "parent_checkpoint_sha256": parent_info["checkpoint_sha256"],
        "checkpoint": str(checkpoint.relative_to(snapshot)),
        "checkpoint_sha256": common.sha256_file(checkpoint),
        "config": "config.yml",
        "config_sha256": common.sha256_file(snapshot / "config.yml"),
        "validation": "validation.json",
        "validation_sha256": common.sha256_file(snapshot / "validation.json"),
        "render_sha256": common.sha256_file(final_render),
        "source_checkpoint": str(selected.checkpoint),
        "source_checkpoint_sha256": selected.checkpoint_sha256,
        "source_config": str(source_config),
        "source_config_sha256": source_config_sha256,
        "training_source_commit": source_git["commit"],
        "training_source_fingerprint": source_git["source_fingerprint"],
        "fresh_eval": dict(fresh_eval),
    }
    common.atomic_json(snapshot / "provenance.json", provenance)


def _update_metrics(
    frame: str,
    selected: common.Boundary,
    parent_frame: str,
    metrics: Mapping[str, float],
    snapshot_checkpoint: Path,
) -> None:
    rows = common.read_metrics_rows()
    existing = next((row for row in rows if row["frame"] == frame), None)
    row = {
        "frame": frame,
        "seed": selected.seed,
        "parent_frame": parent_frame,
        "selected_step": selected.local_step,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "visual_gate": "pass",
        "checkpoint": str(snapshot_checkpoint),
        "checkpoint_sha256": common.sha256_file(snapshot_checkpoint),
    }
    if existing is not None:
        normalized = {name: str(row[name]) for name in common.METRICS_COLUMNS}
        if existing != normalized:
            raise common.InfrastructureError(f"Conflicting metrics row already exists for {frame}")
        return
    expected_previous = common.previous_frame(frame)
    if not rows or rows[-1]["frame"] != expected_previous:
        raise common.InfrastructureError(
            f"metrics.csv is not ready to append {frame}; last frame is "
            f"{rows[-1]['frame'] if rows else '<none>'}"
        )
    common.atomic_csv(common.METRICS_PATH, [*rows, row])


def promote_snapshot(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    parent_info: Mapping[str, Any],
    selected: common.Boundary,
    decisions: Mapping[str, Mapping[str, str]],
) -> bool:
    target_dataset = common.DATA_ROOT / frame
    snapshot = target_dataset / "snapshot"
    render_root = target_dataset / "render"
    frame_record = store.data["frames"][frame]
    promotion = frame_record.setdefault(
        "promotion",
        {
            "status": "preparing",
            "selected": common.boundary_payload(selected),
        },
    )
    if promotion.get("selected", {}).get("checkpoint_sha256") != selected.checkpoint_sha256:
        raise common.InfrastructureError(f"Promotion selection changed for {frame}")

    source_run_dir = selected.checkpoint.parent.parent
    source_config = source_run_dir / "config.yml"
    source_config_hash = common.sha256_file(source_config)
    if not snapshot.exists():
        stage = target_dataset / f".snapshot-stage-{os.getpid()}"
        if stage.exists():
            raise common.InfrastructureError(f"Snapshot staging collision: {stage}")
        checkpoint_target = (
            stage
            / "lookcloser"
            / "final"
            / "nerfstudio_models"
            / selected.checkpoint.name
        )
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected.checkpoint, checkpoint_target)
        if common.sha256_file(checkpoint_target) != selected.checkpoint_sha256:
            raise common.InfrastructureError("Promoted checkpoint copy hash mismatch")
        config = yaml.load(source_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
        final_checkpoint = (
            snapshot
            / "lookcloser"
            / "final"
            / "nerfstudio_models"
            / selected.checkpoint.name
        )
        config.output_dir = target_dataset
        config.experiment_name = "snapshot"
        config.timestamp = "final"
        config.pipeline.datamanager.dataparser.data = target_dataset
        config.data = None
        config.pipeline.datamanager.data = None
        config.load_dir = None
        config.load_step = None
        config.load_config = None
        config.load_checkpoint = final_checkpoint
        config.checkpoint_load_mode = "resume"
        config.load_optimizers = True
        config.load_scheduler = True
        config.max_num_iterations = selected.local_step + 1
        config.pipeline.model.eval_num_rays_per_chunk = 2048
        (stage / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
        stage.replace(snapshot)
        promotion["status"] = "snapshot_installed"
        promotion["snapshot"] = str(snapshot)
        store.flush()

    snapshot_checkpoint = common.snapshot_checkpoint(snapshot)
    if snapshot_checkpoint.name != selected.checkpoint.name:
        raise common.InfrastructureError(f"Installed snapshot checkpoint changed for {frame}")
    if common.sha256_file(snapshot_checkpoint) != selected.checkpoint_sha256:
        raise common.InfrastructureError(f"Installed snapshot checkpoint hash changed for {frame}")
    installed_config = yaml.load(
        (snapshot / "config.yml").read_text(encoding="utf-8"), Loader=yaml.Loader
    )
    if Path(installed_config.load_checkpoint).resolve() != snapshot_checkpoint.resolve():
        raise common.InfrastructureError("Snapshot config retained a temporary checkpoint path")
    if (
        Path(installed_config.pipeline.datamanager.dataparser.data).resolve()
        != target_dataset.resolve()
    ):
        raise common.InfrastructureError("Snapshot config retained the wrong dataparser path")

    validation_dir = frame_root(args, frame) / "final_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_path = snapshot / "validation.json"
    fresh_renders = validation_dir / "renders"
    fresh_record_path = validation_dir / "fresh_eval.json"
    if fresh_record_path.is_file() and validation_path.is_file():
        fresh_eval = json.loads(fresh_record_path.read_text(encoding="utf-8"))
        if fresh_eval["validation_sha256"] != common.sha256_file(validation_path):
            raise common.InfrastructureError("Fresh validation hash changed on resume")
    else:
        fresh_eval = _run_ns_eval(
            args,
            config_path=snapshot / "config.yml",
            validation_path=validation_path,
            render_dir=fresh_renders,
            log_path=validation_dir / "eval_stdout.log",
        )
        common.atomic_json(fresh_record_path, fresh_eval)
    scheduled = {
        "psnr": selected.psnr,
        "ssim": selected.ssim,
        "lpips": selected.lpips,
    }
    tolerance = {"psnr": 1e-4, "ssim": 1e-5, "lpips": 1e-4}
    drift = {
        name: float(fresh_eval["results"][name]) - scheduled[name] for name in scheduled
    }
    if any(abs(drift[name]) > tolerance[name] for name in drift):
        raise common.InfrastructureError(f"Fresh snapshot metrics drifted: {drift}")
    metrics = fresh_eval["results"]
    if not (
        metrics["psnr"] >= common.PSNR_MIN
        and metrics["ssim"] >= common.SSIM_MIN
        and metrics["lpips"] <= common.LPIPS_MAX
    ):
        raise common.QualityFailure(f"Fresh snapshot missed hard gates: {metrics}")

    final_comparison = validation_dir / "comparison"
    comparison_json = final_comparison / "comparison.json"
    if not comparison_json.is_file():
        common.build_native_comparison(
            frame=frame,
            seed=selected.seed,
            step=selected.local_step,
            target_render=fresh_renders / "eval_img_0000.png",
            previous_accepted_render=(
                common.DATA_ROOT
                / common.previous_frame(frame)
                / "render"
                / "eval_img_0000.png"
            ),
            leader_render=common.DATA_ROOT / "007740" / "render" / "eval_img_0000.png",
            output_dir=final_comparison,
            previous_boundary_render=selected.render_dir / "eval_img_0000.png",
        )
    final_key = common.final_visual_key(frame, selected.checkpoint_sha256)
    final_decision = decisions.get(
        final_key,
        {"verdict": "pending", "change_from_previous": "not_applicable", "note": ""},
    )
    if final_key not in decisions:
        writable = dict(decisions)
        writable[final_key] = dict(final_decision)
        common.atomic_json(args.visual_decisions, writable)
    if final_decision["verdict"] == "pending":
        promotion["status"] = "awaiting_final_visual"
        promotion["final_visual_key"] = final_key
        promotion["final_comparison"] = str(
            final_comparison / "native_comparison.png"
        )
        store.flush()
        return False
    if final_decision["verdict"] != "pass":
        raise common.QualityFailure(f"Fresh snapshot visual gate failed: {final_key}")

    render_root.mkdir(parents=True, exist_ok=True)
    final_render = render_root / "eval_img_0000.png"
    if final_render.exists():
        if common.sha256_file(final_render) != common.sha256_file(
            fresh_renders / "eval_img_0000.png"
        ):
            raise common.InfrastructureError(f"Final render collision for {frame}")
    else:
        temporary_render = render_root / ".eval_img_0000.png.tmp"
        shutil.copy2(fresh_renders / "eval_img_0000.png", temporary_render)
        temporary_render.replace(final_render)

    source_git = trajectory.git_preflight()
    _write_snapshot_metadata(
        frame=frame,
        selected=selected,
        parent_info=parent_info,
        snapshot=snapshot,
        source_config=source_config,
        source_config_sha256=source_config_hash,
        fresh_eval={**fresh_eval, "metric_drift": drift},
        final_render=final_render,
        final_decision=final_decision,
        campaign_frame_root=frame_root(args, frame),
        source_git=source_git,
    )
    _update_metrics(
        frame,
        selected,
        str(parent_info["frame"]),
        metrics,
        snapshot_checkpoint,
    )
    validated = common.validate_snapshot_files(snapshot, expected_frame=frame)
    expected_files = {
        "config.yml",
        "selection.json",
        "provenance.json",
        "validation.json",
        str(snapshot_checkpoint.relative_to(snapshot)),
    }
    actual_files = {
        str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise common.InfrastructureError(
            f"Final snapshot file set mismatch for {frame}: {sorted(actual_files)}"
        )
    renders = sorted(path.name for path in render_root.iterdir() if path.is_file())
    if renders != ["eval_img_0000.png"]:
        raise common.InfrastructureError(f"Final render file set mismatch for {frame}: {renders}")
    promotion.update(
        {
            "status": "complete",
            "validated_snapshot": validated,
            "final_render": str(final_render),
            "final_render_sha256": common.sha256_file(final_render),
        }
    )
    store.flush()
    return True


def prune_nonselected_checkpoints(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    selected: common.Boundary,
) -> None:
    root = frame_root(args, frame).resolve()
    manifest_path = root / "pruning.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            return
        entries = list(manifest["entries"])
        if manifest.get("selected_checkpoint_sha256") != selected.checkpoint_sha256:
            raise common.InfrastructureError(f"Pruning selection changed for {frame}")
    else:
        paths = sorted(
            (root / "runs").glob(
                "seed-*/authoritative/*/lookcloser/run/"
                "nerfstudio_models/step-*.ckpt"
            )
        )
        entries = []
        selected_resolved = selected.checkpoint.resolve()
        for path in paths:
            resolved = path.resolve()
            if resolved == selected_resolved:
                continue
            if not resolved.is_relative_to(root) or path.suffix != ".ckpt":
                raise common.InfrastructureError(f"Unsafe pruning target: {path}")
            entries.append(
                {
                    "path": str(path),
                    "step": common.checkpoint_step(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": common.sha256_file(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "frame": frame,
            "status": "planned",
            "authorized_policy": "delete nonselected intermediate checkpoints only after accepted promotion",
            "selected_checkpoint_retained": str(selected.checkpoint),
            "selected_checkpoint_sha256": selected.checkpoint_sha256,
            "entries": entries,
        }
        common.atomic_json(manifest_path, manifest)
    for entry in entries:
        path = Path(entry["path"])
        if path.is_file():
            if common.sha256_file(path) != entry["sha256"]:
                raise common.InfrastructureError(f"Checkpoint changed before pruning: {path}")
            path.unlink()
    manifest["status"] = "complete"
    manifest["completed_at"] = common.utc_now()
    manifest["bytes_removed"] = sum(int(entry["size_bytes"]) for entry in entries)
    common.atomic_json(manifest_path, manifest)
    store.data["frames"][frame]["pruning"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": common.sha256_file(manifest_path),
        "bytes_removed": manifest["bytes_removed"],
    }
    store.flush()


def validate_bootstrap() -> Dict[str, Any]:
    rows = common.read_metrics_rows()
    if [row["frame"] for row in rows[:2]] != ["007740", "007747"]:
        raise common.InfrastructureError("metrics.csv does not start with the two bootstrap frames")
    if len(rows) != 2:
        existing_targets = [row["frame"] for row in rows[2:]]
    else:
        existing_targets = []
    return {
        "007740": common.validate_snapshot_files(
            common.DATA_ROOT / "007740" / "snapshot", expected_frame="007740"
        ),
        "007747": common.validate_snapshot_files(
            common.DATA_ROOT / "007747" / "snapshot", expected_frame="007747"
        ),
        "existing_target_rows": existing_targets,
    }


def process_frame(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    parent_info: Mapping[str, Any],
) -> bool:
    target_dataset = common.DATA_ROOT / frame
    record = store.data["frames"].setdefault(
        frame,
        {
            "status": "initialized",
            "parent_frame": parent_info["frame"],
            "parent_snapshot": parent_info["snapshot"],
            "parent_checkpoint_sha256": parent_info["checkpoint_sha256"],
        },
    )
    if record["parent_frame"] != parent_info["frame"] or record[
        "parent_checkpoint_sha256"
    ] != parent_info["checkpoint_sha256"]:
        raise common.InfrastructureError(f"Parent changed on resume for {frame}")
    if record.get("status") == "accepted":
        common.validate_snapshot_files(target_dataset / "snapshot", expected_frame=frame)
        return True
    if record.get("status") == "pruning":
        selected = common.boundary_from_payload(record["selection"])
        prune_nonselected_checkpoints(args, store, frame=frame, selected=selected)
        record["status"] = "accepted"
        record["accepted_at"] = common.utc_now()
        store.flush()
        return True

    common.freeze_dataset_manifest(
        frame, target_dataset, frame_root(args, frame) / "input_manifest.json"
    )
    if "active_runs" not in record:
        if (target_dataset / "snapshot").exists() or (target_dataset / "render").exists():
            raise common.InfrastructureError(
                f"Untracked final artifact collision before training {frame}"
            )
        gpu = gpu_preflight(args, len(args.initial_seeds))
        seeds = args.initial_seeds[: int(gpu["selected"])]
        storage = storage_preflight(
            args, frame, Path(str(parent_info["checkpoint"])), len(seeds)
        )
        record["parallelism"] = gpu
        record["initial_seed_policy"] = {
            "seeds": list(seeds),
            "reason": "user-authorized wall-clock optimization",
            "fallback_seed": 42 if seeds == (43,) else None,
        }
        record["storage_preflight"] = storage
        record["active_runs"] = {
            str(seed): str(attempt_dir(args, frame, seed, 1)) for seed in seeds
        }
        record["attempt"] = 1
        record["status"] = "training"
        store.flush()
        try:
            run_wave(
                args,
                store,
                frame=frame,
                parent_snapshot=Path(str(parent_info["snapshot"])),
                seeds=seeds,
                active_runs=record["active_runs"],
                extend=False,
            )
        except RuntimeError as error:
            if str(error) != "THREE_JOB_OOM":
                raise
            record["three_job_oom"] = {
                "at": common.utc_now(),
                "evidence": store.data["waves"][-1],
            }
            record["attempt"] = 2
            record["active_runs"] = {
                str(seed): str(attempt_dir(args, frame, seed, 2)) for seed in (42, 43)
            }
            record["parallelism_fallback"] = {
                "selected": 2,
                "reason": "real three-job CUDA OOM",
            }
            store.flush()
            run_wave(
                args,
                store,
                frame=frame,
                parent_snapshot=Path(str(parent_info["snapshot"])),
                seeds=(42, 43),
                active_runs=record["active_runs"],
                extend=False,
            )
    else:
        incomplete = []
        for seed_text, path_text in record["active_runs"].items():
            summary_path = Path(path_text) / "summary.json"
            if not summary_path.is_file():
                incomplete.append(int(seed_text))
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if int(summary.get("latest_step", -1)) < common.INITIAL_TARGET_STEP:
                    incomplete.append(int(seed_text))
        if incomplete:
            if not record["active_runs"]:
                raise common.InfrastructureError(f"No initial trajectory exists for {frame}")
            run_wave(
                args,
                store,
                frame=frame,
                parent_snapshot=Path(str(parent_info["snapshot"])),
                seeds=tuple(incomplete),
                active_runs=record["active_runs"],
                extend=False,
            )

    boundaries_by_seed = _active_boundaries(record)
    comparisons = build_frame_comparisons(args, frame, boundaries_by_seed)
    all_boundaries = [row for rows in boundaries_by_seed.values() for row in rows]
    decisions = ensure_visual_decision_template(args.visual_decisions, frame, all_boundaries)
    pending = pending_visual_keys(frame, all_boundaries, decisions)
    if pending:
        record["status"] = "awaiting_boundary_visual"
        record["visual_review"] = {
            "pending_keys": pending,
            "comparison_index": str(frame_root(args, frame) / "comparison_index.json"),
            "comparison_count": len(comparisons),
        }
        store.flush()
        print(
            f"frame={frame} awaiting_visual count={len(pending)} "
            f"index={record['visual_review']['comparison_index']}",
            flush=True,
        )
        return False

    valid = [
        row
        for row in all_boundaries
        if common.boundary_is_valid(frame, row, decisions)
    ]
    if not valid:
        contenders = authorized_tail_seeds(
            args,
            common.hard_gate_bootstrap_seeds(frame, boundaries_by_seed, decisions),
        )
        if not contenders:
            raise common.QualityFailure(
                f"{frame} produced no hard-gate checkpoint and has no "
                "PSNR/SSIM/visual-pass trajectory eligible for another interval"
            )
        record["status"] = "tail_training"
        record.setdefault("tail_waves", []).append(
            {
                "at": common.utc_now(),
                "selected_before_wave": None,
                "contender_seeds": list(contenders),
                "reason": "bootstrap LPIPS toward hard gate; no checkpoint is accepted",
            }
        )
        store.flush()
        run_wave(
            args,
            store,
            frame=frame,
            parent_snapshot=Path(str(parent_info["snapshot"])),
            seeds=contenders,
            active_runs=record["active_runs"],
            extend=True,
        )
        boundaries_by_seed = _active_boundaries(record)
        build_frame_comparisons(args, frame, boundaries_by_seed)
        all_boundaries = [row for rows in boundaries_by_seed.values() for row in rows]
        ensure_visual_decision_template(args.visual_decisions, frame, all_boundaries)
        record["status"] = "awaiting_boundary_visual"
        store.flush()
        return False
    selected = common.select_boundary(valid)
    selected_seed_rows = boundaries_by_seed[selected.seed]
    if not common.plateau_confirmed(frame, selected_seed_rows, decisions):
        contenders = authorized_tail_seeds(args, common.contender_seeds(valid))
        if not contenders:
            raise common.QualityFailure(f"{frame} has no valid tail contender")
        record["status"] = "tail_training"
        record.setdefault("tail_waves", []).append(
            {
                "at": common.utc_now(),
                "selected_before_wave": common.boundary_payload(selected),
                "contender_seeds": list(contenders),
            }
        )
        store.flush()
        run_wave(
            args,
            store,
            frame=frame,
            parent_snapshot=Path(str(parent_info["snapshot"])),
            seeds=contenders,
            active_runs=record["active_runs"],
            extend=True,
        )
        # New comparisons require new explicit decisions, so return through the
        # same review gate rather than starting another interval blindly.
        boundaries_by_seed = _active_boundaries(record)
        build_frame_comparisons(args, frame, boundaries_by_seed)
        all_boundaries = [row for rows in boundaries_by_seed.values() for row in rows]
        ensure_visual_decision_template(args.visual_decisions, frame, all_boundaries)
        record["status"] = "awaiting_boundary_visual"
        store.flush()
        return False

    record["selection"] = common.boundary_payload(selected)
    record["quality_tier"] = (
        "preferred"
        if selected.preferred_pass
        else "hard_minimum_after_confirmed_plateau"
    )
    if not selected.preferred_pass:
        record["quality_fallback_reason"] = (
            "The checkpoint clears every hard and visual gate, and the selected "
            "trajectory confirmed two consecutive plateau intervals without "
            "reaching all preferred 007747-or-better targets."
        )
    record["status"] = "promoting"
    store.flush()
    if not promote_snapshot(
        args,
        store,
        frame=frame,
        parent_info=parent_info,
        selected=selected,
        decisions=common.load_visual_decisions(args.visual_decisions),
    ):
        return False
    record["status"] = "pruning"
    store.flush()
    prune_nonselected_checkpoints(
        args, store, frame=frame, selected=selected
    )
    record["status"] = "accepted"
    record["accepted_at"] = common.utc_now()
    store.flush()
    print(
        f"frame={frame} accepted seed={selected.seed} step={selected.local_step} "
        f"psnr={selected.psnr:.6f} ssim={selected.ssim:.6f} lpips={selected.lpips:.6f}",
        flush=True,
    )
    return True


def deterministic_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    commands = {}
    for frame in selected_frames(args):
        parent = common.previous_frame(frame)
        parent_snapshot = common.DATA_ROOT / parent / "snapshot"
        commands[frame] = {
            str(seed): _trajectory_command(
                args,
                frame=frame,
                parent_snapshot=parent_snapshot,
                seed=seed,
                output_dir=attempt_dir(args, frame, seed, 1),
                resume=False,
            )
            for seed in args.initial_seeds
        }
    return {
        "schema_version": 2,
        "campaign_root": str(args.campaign_root),
        "frames": list(selected_frames(args)),
        "commands": commands,
        "initial_seed_policy": {
            "seeds": list(args.initial_seeds),
            "reason": "user-authorized wall-clock optimization",
        },
        "selection": "visual and hard gates, max PSNR, inclusive 0.07 dB, min LPIPS",
        "tail_policy": "PSNR-window contender seeds, one interval per review wave",
    }


def run(args: argparse.Namespace) -> int:
    bootstrap = validate_bootstrap()
    if args.dry_run:
        print(json.dumps(deterministic_dry_run(args), indent=2, sort_keys=True))
        return 0
    if args.preflight_only:
        parent_name = common.previous_frame(args.start_frame)
        parent = (
            bootstrap[parent_name]
            if parent_name in bootstrap
            else common.validate_snapshot_files(
                common.DATA_ROOT / parent_name / "snapshot",
                expected_frame=parent_name,
            )
        )
        gpu = gpu_preflight(args, len(args.initial_seeds))
        dataset = common.compute_dataset_manifest(
            args.start_frame, common.DATA_ROOT / args.start_frame
        )
        storage = storage_preflight(
            args,
            args.start_frame,
            Path(str(parent["checkpoint"])),
            int(gpu["selected"]),
        )
        print(
            json.dumps(
                {
                    "bootstrap": bootstrap,
                    "gpu": gpu,
                    "dataset": dataset,
                    "storage": storage,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    args.campaign_root.mkdir(parents=True, exist_ok=True)
    store = CampaignStore(args.campaign_root / "campaign.json", resume=args.resume)
    store.data["campaign_root"] = str(args.campaign_root)
    store.data["frames_requested"] = list(selected_frames(args))
    store.data["visual_decisions"] = str(args.visual_decisions)
    store.data["bootstrap"] = bootstrap
    store.data["status"] = "running"
    store.flush()
    if not args.visual_decisions.exists():
        common.atomic_json(args.visual_decisions, {})

    parent_name = common.previous_frame(args.start_frame)
    parent_info: Mapping[str, Any] = (
        bootstrap[parent_name]
        if parent_name in bootstrap
        else common.validate_snapshot_files(
            common.DATA_ROOT / parent_name / "snapshot", expected_frame=parent_name
        )
    )
    for frame in selected_frames(args):
        expected_parent = common.previous_frame(frame)
        if str(parent_info["frame"]) != expected_parent:
            parent_info = common.validate_snapshot_files(
                common.DATA_ROOT / expected_parent / "snapshot",
                expected_frame=expected_parent,
            )
        if not process_frame(
            args, store, frame=frame, parent_info=parent_info
        ):
            store.data["status"] = "awaiting_visual_review"
            store.data["current_frame"] = frame
            store.flush()
            return QUALITY_EXIT
        parent_info = common.validate_snapshot_files(
            common.DATA_ROOT / frame / "snapshot", expected_frame=frame
        )
    store.data["status"] = "complete"
    store.data["completed_at"] = common.utc_now()
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
