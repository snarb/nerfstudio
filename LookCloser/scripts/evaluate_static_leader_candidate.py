#!/usr/bin/env python3
"""Fresh-evaluate and record one scheduled static-leader checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType


DEFAULT_RUNNER = Path(
    "/home/brans/repos/nerfstudio_leader_stable_occ/LookCloser/scripts/run_lookcloser_quiet.py"
)
DEFAULT_DETAIL_SCORER = Path(
    "/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/scripts/score_static_detail_rois.py"
)
DEFAULT_DETAIL_REFERENCE = Path(
    "/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/experiments/"
    "static_archive_detail_reference.json"
)
EXPECTED_ARTIFACT_VIEWS = 3
EXPECTED_ARTIFACT_ROIS = 10
REQUIRED_DETAIL_CROPS = {
    "thin_pipe_eval1",
    "tangled_cable_holes_eval2",
    "fingers_eval2",
}
FROZEN_DETAIL_CROPS = REQUIRED_DETAIL_CROPS | {
    "stand_eval0",
    "stand_label_eval2",
}
OUTPUT_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def output_tag(value: str) -> str:
    """Accept a single bounded filename token, never a path or dotted suffix."""

    if OUTPUT_TAG_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "output tag must be a 1-64 character alphanumeric/underscore/hyphen token"
        )
    return value


def candidate_output_paths(run_dir: Path, step: int, tag: str) -> dict[str, Path]:
    """Return isolated evaluator outputs while preserving every historical default path."""

    output_tag(tag)
    stem = f"step-{step:09d}"
    return {
        "eval_json": run_dir / f"eval_{tag}_{stem}.json",
        "render_dir": run_dir / f"renders_{tag}_{stem}",
        "eval_log": run_dir / f"eval_{tag}_{stem}_stdout.log",
        "detail_dir": run_dir / f"detail_{tag}_{stem}",
        "summary": run_dir / f"{tag}_evaluation_{stem}.json",
    }


def artifact_infrastructure_errors(artifact: dict) -> list[str]:
    """Return completeness errors; quality scores are intentionally not checked here."""
    errors: list[str] = []
    roi = artifact.get("roi")
    checks = (
        (artifact.get("status") == "complete", "artifact status is not complete"),
        (
            artifact.get("artifact_views_scored") == EXPECTED_ARTIFACT_VIEWS,
            f"expected {EXPECTED_ARTIFACT_VIEWS} scored artifact views",
        ),
        (
            artifact.get("artifact_views_requested") == EXPECTED_ARTIFACT_VIEWS,
            f"expected {EXPECTED_ARTIFACT_VIEWS} requested artifact views",
        ),
        (isinstance(roi, dict), "ROI result is missing"),
    )
    errors.extend(message for passed, message in checks if not passed)
    if isinstance(roi, dict):
        roi_checks = (
            (roi.get("status") == "complete", "ROI status is not complete"),
            (roi.get("returncode") == 0, "ROI scorer did not exit successfully"),
            (
                roi.get("roi_count") == EXPECTED_ARTIFACT_ROIS,
                f"expected exactly {EXPECTED_ARTIFACT_ROIS} ROI scores",
            ),
        )
        errors.extend(message for passed, message in roi_checks if not passed)
    return errors


def require_completed_detail_result(
    returncode: int,
    detail_json: Path,
    expected_render_dir: Path | None = None,
) -> dict:
    """Load a complete detail result; exit 2 is a measured quality fail, not infrastructure failure."""
    if returncode not in (0, 2):
        raise RuntimeError(f"Detail scorer exited {returncode}")
    if not detail_json.is_file():
        raise RuntimeError(f"Detail scorer did not produce {detail_json}")
    try:
        result = json.loads(detail_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Detail scorer produced invalid JSON: {detail_json}") from exc
    comparison = result.get("reference_comparison")
    rows = comparison.get("rois") if isinstance(comparison, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Detail scorer did not produce a reference comparison")
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("crop"), str)
        and isinstance(row.get("pass"), bool)
        for row in rows
    ):
        raise RuntimeError("Detail scorer produced malformed crop rows")
    crop_names = [row["crop"] for row in rows]
    if len(crop_names) != len(set(crop_names)):
        raise RuntimeError("Detail scorer produced duplicate crop rows")
    crop_set = set(crop_names)
    if crop_set != FROZEN_DETAIL_CROPS:
        raise RuntimeError(
            "Detail scorer crop protocol mismatch: "
            f"expected={sorted(FROZEN_DETAIL_CROPS)} actual={sorted(crop_set)}"
        )
    if expected_render_dir is not None:
        recorded_render_dir = result.get("render_dir")
        if not isinstance(recorded_render_dir, str):
            raise RuntimeError("Detail scorer omitted its render directory")
        if Path(recorded_render_dir).resolve() != expected_render_dir.resolve():
            raise RuntimeError("Detail scorer result belongs to a different render directory")
    aggregate_pass = comparison.get("pass")
    if not isinstance(aggregate_pass, bool) or returncode != (0 if aggregate_pass else 2):
        raise RuntimeError("Detail scorer return code disagrees with its reference comparison")
    return result


def required_detail_gate(detail_result: dict) -> bool:
    """Apply the frozen micro-detail gate without requiring the stricter stand/label aggregate."""
    comparison = detail_result["reference_comparison"]
    by_crop = {row["crop"]: bool(row["pass"]) for row in comparison["rois"]}
    return all(by_crop[crop] for crop in REQUIRED_DETAIL_CROPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--historical-runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--detail-scorer", type=Path, default=DEFAULT_DETAIL_SCORER)
    parser.add_argument("--detail-reference", type=Path, default=DEFAULT_DETAIL_REFERENCE)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument(
        "--reuse-eval",
        action="store_true",
        help="Reuse a matching candidate eval JSON and its three existing renders.",
    )
    parser.add_argument(
        "--output-tag",
        type=output_tag,
        default="candidate",
        help="Filename token for isolated diagnostic outputs; the default preserves canonical paths.",
    )
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("historical_lookcloser_quiet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load historical runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_campaign_environment(campaign_path: Path) -> dict[str, object]:
    """Restore and verify the executable environment recorded by the controller."""
    manifest = json.loads(campaign_path.read_text(encoding="utf-8"))
    # Console-script execution does not guarantee that its sibling tools (notably
    # ninja for nerfacc JIT loading) are on PATH. Restore the active venv bin first.
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    python_paths: list[str] = []
    tcnn = manifest.get("tiny_cuda_nn") or {}
    overlay = tcnn.get("overlay")
    if overlay:
        overlay_path = Path(str(overlay))
        if not overlay_path.is_dir():
            raise RuntimeError(f"Recorded tiny-cuda-nn overlay does not exist: {overlay_path}")
        python_paths.append(str(overlay_path))
    historical_worktree = manifest.get("historical_worktree")
    if historical_worktree:
        worktree_path = Path(str(historical_worktree))
        if not worktree_path.is_dir():
            raise RuntimeError(f"Recorded historical worktree does not exist: {worktree_path}")
        python_paths.append(str(worktree_path))
    previous_pythonpath = os.environ.get("PYTHONPATH")
    if previous_pythonpath:
        python_paths.append(previous_pythonpath)
    if python_paths:
        os.environ["PYTHONPATH"] = ":".join(python_paths)

    for manifest_key, environment_key in (
        ("cuda_home", "CUDA_HOME"),
        ("torch_cuda_arch_list", "TORCH_CUDA_ARCH_LIST"),
        ("torch_extensions_dir", "TORCH_EXTENSIONS_DIR"),
    ):
        value = manifest.get(manifest_key)
        if value:
            os.environ[environment_key] = str(value)
    if os.environ.get("CUDA_HOME"):
        os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"

    expected_binding = tcnn.get("binding_sha256")
    if expected_binding:
        imported = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import tinycudann.modules; print(tinycudann.modules._C.__file__)",
            ],
            text=True,
            env=os.environ.copy(),
        ).strip()
        imported_path = Path(imported)
        actual_binding = sha256_file(imported_path)
        if actual_binding != expected_binding:
            raise RuntimeError(
                "Imported tiny-cuda-nn binding does not match campaign provenance: "
                f"{actual_binding} != {expected_binding} ({imported_path})"
            )
    return manifest


def main() -> int:
    args = parse_args()
    if not args.run_dir.is_dir():
        raise RuntimeError(f"Run directory does not exist: {args.run_dir}")
    if not args.checkpoint.is_file():
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.checkpoint.parent.parent != args.run_dir:
        raise RuntimeError("Checkpoint must belong to --run-dir")

    if args.campaign is not None:
        restore_campaign_environment(args.campaign)

    quiet = load_module(args.historical_runner)
    step = checkpoint_step(args.checkpoint)
    paths = candidate_output_paths(args.run_dir, step, args.output_tag)
    eval_config_parameters = inspect.signature(quiet.eval_config_for_step).parameters
    supports_candidate_eval_overrides = {
        "cache_train_rays",
        "filename_tag",
    }.issubset(eval_config_parameters)
    if args.output_tag != "candidate" and not supports_candidate_eval_overrides:
        raise RuntimeError("Historical runner does not support isolated tagged eval configs")
    eval_config_kwargs = {}
    if supports_candidate_eval_overrides:
        eval_config_kwargs = {
            "cache_train_rays": False,
            "filename_tag": None if args.output_tag == "candidate" else args.output_tag,
        }
    eval_config = quiet.eval_config_for_step(
        args.run_dir / "config.yml",
        args.checkpoint,
        args.eval_num_rays_per_chunk,
        **eval_config_kwargs,
    )
    eval_json = paths["eval_json"]
    render_dir = paths["render_dir"]
    eval_log = paths["eval_log"]
    command = [
        str(Path(sys.executable).with_name("ns-eval")),
        "--load-config",
        str(eval_config),
        "--output-path",
        str(eval_json),
        "--render-output-path",
        str(render_dir),
    ]
    if args.reuse_eval:
        if not eval_json.is_file():
            raise RuntimeError(f"Cannot reuse missing eval JSON: {eval_json}")
        eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
        recorded_checkpoint = Path(str(eval_payload.get("checkpoint", "")))
        if recorded_checkpoint.resolve() != args.checkpoint.resolve():
            raise RuntimeError(
                f"Reusable eval checkpoint mismatch: {recorded_checkpoint} != {args.checkpoint}"
            )
        missing_renders = [
            name
            for name in ("eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png")
            if not (render_dir / name).is_file()
        ]
        if missing_renders:
            raise RuntimeError(f"Cannot reuse eval; missing renders: {missing_renders}")
        eval_seconds = 0.0
    else:
        started = time.monotonic()
        with eval_log.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=os.environ.copy(),
            )
        eval_seconds = time.monotonic() - started
        eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
    eval_data = {
        "checkpoint": eval_payload["checkpoint"],
        "results": eval_payload["results"],
        "render_dir": str(render_dir),
        "eval_json": str(eval_json),
        "eval_log": str(eval_log),
        "eval_config": str(eval_config),
        "eval_seconds": eval_seconds,
    }

    artifact_args = argparse.Namespace(
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        artifact_score=True,
        artifact_roi_score=True,
        artifact_detector_preset="significant",
        artifact_render_names="eval_img_0000.png,eval_img_0001.png,eval_img_0002.png",
        artifact_render_name="eval_img_0000.png",
        artifact_roi_crop_names="all",
        artifact_roi_drop_border_components=0,
        artifact_crop_top=0,
        artifact_crop_bottom=0,
        artifact_crop_left=0,
        artifact_crop_right=0,
        artifact_gate_only=True,
    )
    artifact = quiet.run_artifact_detector(args.run_dir, eval_data, artifact_args)
    artifact_dir = Path(str(artifact["artifact_dir"]))
    roi = artifact.get("roi") or {}
    artifact_errors = artifact_infrastructure_errors(artifact)
    if artifact_errors:
        raise RuntimeError(
            "Artifact/ROI infrastructure did not produce the complete 3-view/10-ROI gate: "
            + "; ".join(artifact_errors)
        )

    detail_dir = paths["detail_dir"]
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_log = detail_dir / "detail_stdout.log"
    detail_json = detail_dir / "detail_roi_metrics.json"
    detail_json.unlink(missing_ok=True)
    detail_command = [
        sys.executable,
        str(args.detail_scorer),
        "--render-dir",
        str(render_dir),
        "--out-dir",
        str(detail_dir),
        "--reference",
        str(args.detail_reference),
    ]
    detail_started = time.monotonic()
    detail_process = subprocess.run(
        detail_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    detail_log.write_text(detail_process.stdout, encoding="utf-8")
    detail_result = require_completed_detail_result(
        detail_process.returncode,
        detail_json,
        expected_render_dir=render_dir,
    )
    detail_pass = required_detail_gate(detail_result)

    results = eval_payload["results"]
    numeric_pass = (
        float(results["psnr"]) >= 29.617964
        and float(results["ssim"]) >= 0.668450
        and float(results["lpips"]) <= 0.231135
    )
    automatic_pass = (
        int(artifact.get("artifact_count") or 0) == 0
        and int(artifact.get("artifact_views_scored") or 0) == EXPECTED_ARTIFACT_VIEWS
        and int(roi.get("roi_serious_count") or 0) == 0
        and int(roi.get("roi_count") or 0) == EXPECTED_ARTIFACT_ROIS
    )
    quality_pass = numeric_pass and automatic_pass and detail_pass
    summary = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "eval": eval_data,
        "artifact": artifact,
        "numeric_pass": numeric_pass,
        "automatic_pass": automatic_pass,
        "automatic_gate_complete": True,
        "detail_pass": detail_pass,
        "detail_gate_complete": True,
        "quality_pass": quality_pass,
        "detail": {
            "returncode": detail_process.returncode,
            "seconds": time.monotonic() - detail_started,
            "metrics": str(detail_json),
            "log": str(detail_log),
            "required_crops": sorted(REQUIRED_DETAIL_CROPS),
            "strict_all_five_pass": bool(detail_result["reference_comparison"]["pass"]),
        },
    }

    recorded = False
    record_status = "not_requested_or_gate_failed"
    if args.campaign is not None and quality_pass:
        campaign_payload = json.loads(args.campaign.read_text(encoding="utf-8"))
        existing = campaign_payload.get("accepted_candidate")
        existing_checkpoint = Path(str(existing.get("checkpoint"))) if isinstance(existing, dict) else None
        if existing_checkpoint is not None and existing_checkpoint.resolve() == args.checkpoint.resolve():
            recorded = True
            record_status = "idempotent_existing_candidate"
        else:
            recorder = Path(__file__).with_name("record_static_leader_candidate.py")
            subprocess.run(
                [
                    sys.executable,
                    str(recorder),
                    str(args.campaign),
                    "--eval-json",
                    str(eval_json),
                    "--render-dir",
                    str(render_dir),
                    "--artifact-dir",
                    str(artifact_dir),
                    "--detail-json",
                    str(detail_json),
                ],
                check=True,
            )
            recorded = True
            record_status = "recorded_new_candidate"

    summary["recorded"] = recorded
    summary["record_status"] = record_status
    summary_path = paths["summary"]
    write_json(summary_path, summary)

    print(
        f"candidate={args.checkpoint} psnr={results['psnr']:.6f} "
        f"ssim={results['ssim']:.6f} lpips={results['lpips']:.6f} "
        f"artifacts={artifact.get('artifact_count')} "
        f"roi_serious={roi.get('roi_serious_count')} "
        f"detail_pass={detail_pass} detail_returncode={detail_process.returncode} "
        f"recorded={recorded} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
