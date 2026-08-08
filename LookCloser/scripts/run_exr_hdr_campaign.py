#!/usr/bin/env python3
"""Run the staged EXR/HDR loss, adaptive-frequency, and tuning campaign."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740")
DEFAULT_OUTPUT = Path("/mnt/data/lookcloser_exr_hdr_runs")
DEFAULT_CAMPAIGN_NAME = "exr_hdr_auto_frequency_v2_corrected_arm"
LOSS_CANDIDATES = (
    ("linear_l1", "linear_softplus"),
    ("rawnerf_weighted_l2", "linear_softplus"),
    ("linear_pq", "linear_softplus"),
    ("pq_l1", "pq_code"),
    ("eag_pq_dssim", "linear_softplus"),
)
MAP_METHODS = ("calibrated", "relative", "knee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN_NAME)
    parser.add_argument(
        "--phase",
        choices=("maps", "loss-screen", "map-screen", "tune", "final", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-steps-per-level", type=int, default=250)
    parser.add_argument("--screen-iterations", type=int, default=30377)
    parser.add_argument("--final-iterations", type=int, default=75941)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command: list[str], dry_run: bool) -> int:
    print("command=" + " ".join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def map_root(args: argparse.Namespace) -> Path:
    return args.data / "lookcloser_frequencies_exr_auto"


def build_maps(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    provenance = map_root(args) / "provenance.json"
    if provenance.is_file() and not args.force:
        manifest["maps"] = json.loads(provenance.read_text(encoding="utf-8"))
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "build_adaptive_exr_frequency_maps.py"),
        "--images-dir",
        str(args.data / "images"),
        "--out",
        str(map_root(args)),
        "--steps-per-level",
        str(args.map_steps_per_level),
        "--loss",
        "linear_pq",
        "--seed",
        str(args.seed),
    ]
    if args.force:
        command.append("--force")
    code = invoke(command, args.dry_run)
    if code:
        raise RuntimeError(f"Adaptive frequency-map builder failed with exit code {code}")
    manifest["maps"] = (
        json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else {"status": "dry-run"}
    )


def experiment_run_path(args: argparse.Namespace, tag: str) -> Path:
    return args.output_dir / args.campaign_name / "lookcloser" / tag


def base_train_command(
    args: argparse.Namespace,
    *,
    tag: str,
    loss: str,
    output: str,
    frequency_method: str,
    iterations: int,
    extra: tuple[str, ...] = (),
    early_reference: dict[str, float] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_lookcloser_quiet.py"),
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        args.campaign_name,
        "--timestamp",
        tag,
        "--seed",
        str(args.seed),
        "--frequency-map-dir",
        f"lookcloser_frequencies_exr_auto/{frequency_method}",
        "--reconstruction-loss-type",
        loss,
        "--rgb-output-parameterization",
        output,
        "--max-num-iterations",
        str(iterations),
        "--step-interval",
        "15188",
        "--save-interval",
        "15188",
        "--max-res",
        "8192",
        "--ray-sampling-mode",
        "adaptive",
        "--max-steps-per-ray",
        "1024",
        "--adaptive-coarse-step-size",
        "0.00625",
        "--corrected-arm-allocator",
        "--no-stop-on-no-improve",
        "--eval-checkpoint",
        "best",
        "--keep-all-checkpoints",
        "--no-update-summary",
        *extra,
    ]
    if early_reference is not None:
        command.extend(
            [
                "--early-reject-psnr-below",
                str(early_reference["psnr"] - 0.5),
                "--early-reject-ssim-below",
                str(early_reference["ssim"] - 0.02),
                "--early-reject-lpips-above",
                str(early_reference["lpips"] + 0.04),
            ]
        )
    return command


def load_result(args: argparse.Namespace, tag: str) -> dict[str, Any] | None:
    path = experiment_run_path(args, tag) / "run_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["summary_path"] = str(path)
    return summary


def run_candidate(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    stage: str,
    tag: str,
    loss: str,
    output: str,
    method: str,
    iterations: int,
    extra: tuple[str, ...] = (),
    early_reference: dict[str, float] | None = None,
) -> None:
    existing = load_result(args, tag)
    if existing is None or metrics(existing) is None or args.force:
        code = invoke(
            base_train_command(
                args,
                tag=tag,
                loss=loss,
                output=output,
                frequency_method=method,
                iterations=iterations,
                extra=extra,
                early_reference=early_reference,
            ),
            args.dry_run,
        )
        if code:
            raise RuntimeError(f"Training candidate {tag} failed with exit code {code}")
        existing = load_result(args, tag)
    manifest.setdefault("runs", {})[tag] = existing or {
        "status": "dry-run",
        "stage": stage,
        "loss": loss,
        "output": output,
        "frequency_method": method,
        "iterations": iterations,
        "extra": list(extra),
    }


def alias_candidate(
    manifest: dict[str, Any],
    *,
    tag: str,
    source_tag: str,
    stage: str,
    method: str,
    extra: tuple[str, ...] = (),
) -> bool:
    """Reuse an exactly equivalent measured run without spending GPU time again."""
    source = manifest.get("runs", {}).get(source_tag)
    if metrics(source) is None:
        return False
    alias = dict(source)
    alias.update(
        {
            "alias_of": source_tag,
            "stage": stage,
            "frequency_method": method,
            "extra": list(extra),
        }
    )
    manifest.setdefault("runs", {})[tag] = alias
    print(f"alias={tag} source={source_tag}", flush=True)
    return True


def metrics(run: dict[str, Any] | None) -> dict[str, float] | None:
    if not run:
        return None
    hdr = (run.get("eval") or {}).get("hdr") or {}
    values = hdr.get("aggregate") or {}
    if not all(key in values for key in ("psnr", "ssim", "lpips")):
        return None
    return {key: float(values[key]) for key in ("psnr", "ssim", "lpips")}


def trajectory(run: dict[str, Any] | None) -> list[dict[str, float]]:
    if not run or not run.get("summary_path"):
        return []
    metrics_path = Path(str(run["summary_path"])).parent / "metrics_compact.csv"
    if not metrics_path.is_file():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "step": float(row["step"]),
            "psnr": float(row["eval_all_psnr"]),
            "ssim": float(row["eval_all_ssim"]),
            "lpips": float(row["eval_all_lpips"]),
        }
        for row in rows
        if row.get("eval_all_psnr") and row.get("eval_all_ssim") and row.get("eval_all_lpips")
    ]


def select_best(tags: list[str], manifest: dict[str, Any], stage: str) -> str:
    measured = [(tag, metrics(manifest.get("runs", {}).get(tag))) for tag in tags]
    measured = [(tag, value) for tag, value in measured if value is not None]
    if not measured:
        if all((manifest.get("runs", {}).get(tag) or {}).get("status") == "dry-run" for tag in tags):
            return tags[0]
        raise RuntimeError("No complete measured candidates available for selection")
    trajectories = {
        tag: trajectory(manifest.get("runs", {}).get(tag)) for tag, _ in measured
    }
    if any(not rows for rows in trajectories.values()):
        raise RuntimeError(f"{stage} contains a candidate without an eval boundary")
    first_psnr = max(rows[0]["psnr"] for rows in trajectories.values())
    first_ssim = max(rows[0]["ssim"] for rows in trajectories.values())
    first_lpips = min(rows[0]["lpips"] for rows in trajectories.values())
    first_pass = {
        tag: (
            rows[0]["psnr"] >= first_psnr - 0.5
            and rows[0]["ssim"] >= first_ssim - 0.02
            and rows[0]["lpips"] <= first_lpips + 0.04
        )
        for tag, rows in trajectories.items()
    }
    admissible_tags = {tag for tag, passed in first_pass.items() if passed}
    first_admissible_tags = set(admissible_tags)
    incomplete = sorted(tag for tag in admissible_tags if len(trajectories[tag]) < 2)
    if incomplete:
        raise RuntimeError(f"{stage} candidates passed the first gate but lack a second boundary: {incomplete}")
    common_count = min((len(trajectories[tag]) for tag in admissible_tags), default=0)
    if common_count < 2:
        raise RuntimeError(f"{stage} needs two boundaries from every non-rejected candidate")
    gate_rows = []
    for index in range(common_count):
        if not admissible_tags:
            break
        active = {tag: trajectories[tag] for tag in admissible_tags}
        best_psnr_at_step = max(rows[index]["psnr"] for rows in active.values())
        best_ssim_at_step = max(rows[index]["ssim"] for rows in active.values())
        best_lpips_at_step = min(rows[index]["lpips"] for rows in active.values())
        row_gate = {
            "step": int(next(iter(active.values()))[index]["step"]),
            "reference": {
                "psnr": best_psnr_at_step,
                "ssim": best_ssim_at_step,
                "lpips": best_lpips_at_step,
            },
            "pass": {},
        }
        for tag, rows in active.items():
            row = rows[index]
            passed = (
                row["psnr"] >= best_psnr_at_step - 0.5
                and row["ssim"] >= best_ssim_at_step - 0.02
                and row["lpips"] <= best_lpips_at_step + 0.04
            )
            row_gate["pass"][tag] = passed
            if not passed:
                admissible_tags.discard(tag)
        gate_rows.append(row_gate)
    gate_rows[0]["pass"].update(first_pass)
    fallback_all = not admissible_tags
    if fallback_all:
        admissible_tags = first_admissible_tags
    manifest.setdefault("trajectory_gates", {})[stage] = {
        "minimum_required_points": 2,
        "observed_common_points": common_count,
        "tolerances": {"psnr_db": 0.5, "ssim": 0.02, "lpips": 0.04},
        "rows": gate_rows,
        "admissible": sorted(admissible_tags),
        "fallback_all_candidates": fallback_all,
        "early_rejected_after_first_boundary": sorted(tag for tag, passed in first_pass.items() if not passed),
    }
    measured = [(tag, value) for tag, value in measured if tag in admissible_tags]
    best_psnr = max(value["psnr"] for _, value in measured)
    tied = [(tag, value) for tag, value in measured if value["psnr"] >= best_psnr - 0.07]
    return min(tied, key=lambda item: (item[1]["lpips"], -item[1]["ssim"], -item[1]["psnr"]))[0]


def loss_screen(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    tags = []
    for loss, output in LOSS_CANDIDATES:
        tag = f"loss_{loss}_s{args.seed}"
        tags.append(tag)
        run_candidate(args, manifest, "loss-screen", tag, loss, output, "selected", args.screen_iterations)
    winner = select_best(tags, manifest, "loss-screen")
    manifest.setdefault("selection", {})["loss"] = winner
    return winner


def winner_recipe(tag: str) -> tuple[str, str]:
    loss = tag.removeprefix("loss_").rsplit("_s", 1)[0]
    return next(item for item in LOSS_CANDIDATES if item[0] == loss)


def map_screen(args: argparse.Namespace, manifest: dict[str, Any], loss_tag: str) -> str:
    loss, output = winner_recipe(loss_tag)
    reference = trajectory(manifest.get("runs", {}).get(loss_tag))[0]
    tags = []
    for method in MAP_METHODS:
        tag = f"map_{method}_{loss}_s{args.seed}"
        tags.append(tag)
        if not args.force and method == manifest.get("maps", {}).get("selected_method") and alias_candidate(
            manifest,
            tag=tag,
            source_tag=loss_tag,
            stage="map-screen",
            method=method,
        ):
            continue
        run_candidate(
            args,
            manifest,
            "map-screen",
            tag,
            loss,
            output,
            method,
            args.screen_iterations,
            early_reference=reference,
        )
    winner = select_best(tags, manifest, "map-screen")
    manifest.setdefault("selection", {})["map"] = winner
    return winner


def tune_candidates(loss: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if loss == "linear_l1":
        return tuple((f"beta_{value:g}", ("--hdr-softplus-beta", str(value))) for value in (0.5, 1.0, 2.0))
    if loss == "rawnerf_weighted_l2":
        return tuple((f"eps_{value:g}", ("--rawnerf-epsilon", str(value))) for value in (3e-4, 1e-3, 3e-3))
    if loss == "pq_l1":
        return tuple((f"temperature_{value:g}", ("--pq-code-temperature", str(value))) for value in (0.75, 1.0, 1.5))
    if loss == "eag_pq_dssim":
        return tuple((f"dssim_{value:g}", ("--eag-dssim-weight", str(value))) for value in (0.1, 0.2, 0.3))
    return tuple(
        (f"anchor_{value:g}", ("--pq-linear-anchor-weight", str(value))) for value in (0.0, 0.01, 0.05)
    )


def is_default_tuning(loss: str, extra: tuple[str, ...]) -> bool:
    defaults = {
        "linear_l1": ("--hdr-softplus-beta", 1.0),
        "rawnerf_weighted_l2": ("--rawnerf-epsilon", 1e-3),
        "pq_l1": ("--pq-code-temperature", 1.0),
        "eag_pq_dssim": ("--eag-dssim-weight", 0.2),
        "linear_pq": ("--pq-linear-anchor-weight", 0.0),
    }
    flag, value = defaults[loss]
    return len(extra) == 2 and extra[0] == flag and float(extra[1]) == value


def tune(args: argparse.Namespace, manifest: dict[str, Any], loss_tag: str, map_tag: str) -> str:
    loss, output = winner_recipe(loss_tag)
    method = map_tag.removeprefix("map_").removesuffix(f"_{loss}_s{args.seed}")
    reference = trajectory(manifest.get("runs", {}).get(map_tag))[0]
    tags = []
    for label, extra in tune_candidates(loss):
        tag = f"tune_{loss}_{method}_{label}_s{args.seed}"
        tags.append(tag)
        if not args.force and is_default_tuning(loss, extra) and alias_candidate(
            manifest,
            tag=tag,
            source_tag=map_tag,
            stage="tune",
            method=method,
            extra=extra,
        ):
            continue
        run_candidate(
            args,
            manifest,
            "tune",
            tag,
            loss,
            output,
            method,
            args.screen_iterations,
            extra,
            early_reference=reference,
        )
    winner = select_best(tags, manifest, "tune")
    manifest.setdefault("selection", {})["tuning"] = winner
    return winner


def final_run(args: argparse.Namespace, manifest: dict[str, Any], tune_tag: str, loss_tag: str, map_tag: str) -> None:
    loss, output = winner_recipe(loss_tag)
    method = map_tag.removeprefix("map_").removesuffix(f"_{loss}_s{args.seed}")
    tuning = next(
        extra for label, extra in tune_candidates(loss) if f"_{label}_s{args.seed}" in tune_tag
    )
    tag = f"final_{loss}_{method}_s{args.seed}"
    run_candidate(args, manifest, "final", tag, loss, output, method, args.final_iterations, tuning)
    manifest.setdefault("selection", {})["final"] = tag


def require_selection(manifest: dict[str, Any], key: str) -> str:
    value = manifest.get("selection", {}).get(key)
    if not value:
        raise RuntimeError(f"Campaign manifest has no {key!r} selection; run the preceding phase first")
    return str(value)


def main() -> int:
    args = parse_args()
    campaign_dir = args.output_dir / "campaigns" / args.campaign_name
    manifest_path = campaign_dir / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "schema": 1,
        "campaign": args.campaign_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "runs": {},
        "selection": {},
    }
    phases = ("maps", "loss-screen", "map-screen", "tune", "final") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "maps":
            build_maps(args, manifest)
        elif phase == "loss-screen":
            loss_screen(args, manifest)
        elif phase == "map-screen":
            map_screen(args, manifest, require_selection(manifest, "loss"))
        elif phase == "tune":
            tune(
                args,
                manifest,
                require_selection(manifest, "loss"),
                require_selection(manifest, "map"),
            )
        elif phase == "final":
            final_run(
                args,
                manifest,
                require_selection(manifest, "tuning"),
                require_selection(manifest, "loss"),
                require_selection(manifest, "map"),
            )
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
    print(f"campaign={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
