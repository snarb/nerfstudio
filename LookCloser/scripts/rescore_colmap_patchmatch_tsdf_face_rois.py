#!/usr/bin/env python3
"""Rescore a finalized PatchMatch-TSDF campaign prefix after a GT-only ROI correction.

This opt-in migration does not touch reconstruction artifacts.  It validates
the immutable campaign request and frozen scorer, stages every new score before
publishing any of them, preserves the superseded metric files, and rebuilds the
ordered CSV atomically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Sequence

from colmap_patchmatch_tsdf_campaign_common import (
    CSV_FIELDS,
    atomic_csv,
    atomic_json,
    canonical_sha256,
    load_json,
    robust_initial_thresholds,
    sha256,
    validate_hash_manifest,
)


CORRECTION_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,63}")
MUTATED_RELATIVE_PATHS = (
    Path("metrics.json"),
    Path("metrics/face_polygons.json"),
    Path("metrics/roi_rescore_receipt.json"),
    Path("visual/face_mask.png"),
    Path("visual/face_mask_overlay.png"),
    Path("result.json"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Atomic copy hash mismatch: {source} -> {destination}")
    os.replace(temporary, destination)


def published_prefix(output_root: Path, ordered: list[str]) -> list[str]:
    published = [frame_id for frame_id in ordered if (output_root / "frames" / frame_id).is_dir()]
    if published != ordered[: len(published)]:
        raise ValueError("Published frame inventory is not a contiguous ordered prefix")
    return published


def reject_live_campaign_controller(lock: Path) -> None:
    if not lock.is_file():
        return
    try:
        token = lock.read_text(encoding="utf-8").split()[0]
        pid = int(token.removeprefix("pid="))
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        lock.unlink(missing_ok=True)
        return
    raise RuntimeError(f"Campaign controller is active with pid={pid}: {lock}")


def metric_status(metrics: dict, thresholds: dict[str, float], previous: list[dict]) -> tuple[str, list[str]]:
    values = {key: float(metrics[key]) for key in ("face_psnr", "face_ssim", "face_lpips")}
    if not all(math.isfinite(value) for value in values.values()):
        return "fail_nonfinite", ["nonfinite"]
    if not previous:
        return "pass", []
    medians = {key: statistics.median(float(row[key]) for row in previous) for key in values}
    reasons = []
    if values["face_psnr"] < medians["face_psnr"] - float(thresholds["face_psnr"]):
        reasons.append("face_psnr_drop")
    if values["face_ssim"] < medians["face_ssim"] - float(thresholds["face_ssim"]):
        reasons.append("face_ssim_drop")
    if values["face_lpips"] > medians["face_lpips"] + float(thresholds["face_lpips"]):
        reasons.append("face_lpips_rise")
    return ("regression_flag" if reasons else "pass"), reasons


def rewrite_metric_paths(metrics: dict, final: Path) -> dict:
    result = dict(metrics)
    result["prediction"] = str(final / "render/eval_pred_0000.exr")
    result["ground_truth"] = str(final / "render/eval_gt_0000.exr")
    result["face_polygons"] = str(final / "metrics/face_polygons.json")
    result["review_crops"] = {
        key: str(final / "visual" / Path(value).name)
        for key, value in result["review_crops"].items()
    }
    return result


def updated_result(old: dict, metrics: dict, correction_id: str, polygon_sha256: str) -> dict:
    result = dict(old)
    for key in ("face_psnr", "face_ssim", "face_lpips", "metric_status"):
        result[key] = metrics[key]
    result["status"] = (
        "pass"
        if result["visual_status"] == "pass" and metrics["metric_status"] != "fail_nonfinite"
        else "fail"
    )
    result["metric_revision"] = {
        "correction_id": correction_id,
        "corrected_at": now(),
        "face_polygons_sha256": polygon_sha256,
        "reason": "replace static neck-contaminated ROI with manually verified heldout-GT-only face ROI",
    }
    return result


def validate_request(output_root: Path) -> tuple[dict, Path]:
    request = load_json(output_root / "campaign_request.json")
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    if request.get("request_sha256") != canonical_sha256(body):
        raise ValueError("campaign_request.json self-hash is invalid")
    scorer_row = next(
        (row for row in request["scripts"] if row["name"] == "score_colmap_patchmatch_tsdf_face.py"),
        None,
    )
    if scorer_row is None:
        raise ValueError("Immutable request does not inventory the face scorer")
    scorer = output_root / "config/code/score_colmap_patchmatch_tsdf_face.py"
    if not scorer.is_file() or sha256(scorer) != scorer_row["sha256"]:
        raise ValueError("Frozen campaign face scorer differs from the immutable request")
    return request, scorer


def copy_old_file(source: Path, destination: Path) -> dict[str, object] | None:
    if not source.is_file():
        return None
    atomic_copy(source, destination)
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)}


def rescore(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    polygons_dir = args.polygons_dir.expanduser().resolve()
    if not CORRECTION_ID_RE.fullmatch(args.correction_id):
        raise ValueError("--correction-id must be a short lowercase filesystem token")
    reject_live_campaign_controller(output_root / ".campaign_controller.lock")
    lock = output_root / ".campaign_roi_rescore.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"Another ROI rescore is active: {lock}") from error
    os.write(descriptor, f"pid={os.getpid()} started_at={now()}\n".encode())
    os.close(descriptor)
    try:
        request, scorer = validate_request(output_root)
        ordered = list(request["ordered_frame_ids"])
        frames = published_prefix(output_root, ordered)
        if args.frames:
            selected = [frame_id for frame_id in ordered if frame_id in set(args.frames)]
            if len(selected) != len(args.frames) or selected != frames:
                raise ValueError("ROI correction must cover the complete published ordered prefix")
        if not frames:
            raise ValueError("There are no finalized frames to rescore")

        correction_root = output_root / ".diagnostics" / f"roi_rescore_{args.correction_id}"
        stage = output_root / ".work" / f".roi_rescore_{args.correction_id}.tmp-{os.getpid()}"
        if correction_root.exists() or stage.exists():
            raise FileExistsError(correction_root if correction_root.exists() else stage)
        stage.mkdir(parents=True)

        raw_scores: dict[str, dict] = {}
        before: dict[str, dict] = {}
        for index, frame_id in enumerate(frames, 1):
            final = output_root / "frames" / frame_id
            validate_hash_manifest(final, load_json(final / "retained_manifest.json"))
            old_result = load_json(final / "result.json")
            if old_result.get("request_sha256") != request["request_sha256"]:
                raise ValueError(f"Published request hash mismatch for {frame_id}")
            polygon = polygons_dir / f"{frame_id}.json"
            if not polygon.is_file():
                raise FileNotFoundError(polygon)
            frame_stage = stage / frame_id
            subprocess.run(
                [
                    sys.executable,
                    str(scorer),
                    "--frame-id",
                    frame_id,
                    "--prediction",
                    str(final / "render/eval_pred_0000.exr"),
                    "--ground-truth",
                    str(final / "render/eval_gt_0000.exr"),
                    "--face-polygons",
                    str(polygon),
                    "--output-dir",
                    str(frame_stage),
                    "--device",
                    args.metric_device,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            metrics = rewrite_metric_paths(load_json(frame_stage / "metrics.json"), final)
            raw_scores[frame_id] = metrics
            before[frame_id] = {
                "metrics": {
                    key: old_result[key] for key in ("face_psnr", "face_ssim", "face_lpips", "metric_status")
                },
                "result_sha256": sha256(final / "result.json"),
                "metrics_sha256": sha256(final / "metrics.json"),
                "polygon_sha256": sha256(final / "metrics/face_polygons.json"),
            }
            print(f"staged={index}/{len(frames)} frame={frame_id}", flush=True)

        baseline_rows = [raw_scores[frame_id] for frame_id in ordered[:3]]
        thresholds = robust_initial_thresholds(baseline_rows)
        new_results: dict[str, dict] = {}
        accepted: list[dict] = []
        for frame_id in frames:
            metrics = raw_scores[frame_id]
            status, reasons = metric_status(metrics, thresholds, accepted[-5:])
            metrics["metric_status"] = status
            metrics["regression_reasons"] = reasons
            metrics["regression_reference_frame_ids"] = [row["frame_id"] for row in accepted[-5:]]
            atomic_json(stage / frame_id / "metrics.json", metrics)
            old_result = load_json(output_root / "frames" / frame_id / "result.json")
            polygon_sha = sha256(polygons_dir / f"{frame_id}.json")
            result = updated_result(old_result, metrics, args.correction_id, polygon_sha)
            atomic_json(stage / frame_id / "result.json", result)
            receipt = {
                "schema_version": 1,
                "correction_id": args.correction_id,
                "frame_id": frame_id,
                "request_sha256": request["request_sha256"],
                "frozen_scorer_sha256": sha256(scorer),
                "prediction_sha256": sha256(output_root / "frames" / frame_id / "render/eval_pred_0000.exr"),
                "ground_truth_sha256": sha256(output_root / "frames" / frame_id / "render/eval_gt_0000.exr"),
                "face_polygons_sha256": polygon_sha,
                "before": before[frame_id],
                "after": {
                    "face_psnr": metrics["face_psnr"],
                    "face_ssim": metrics["face_ssim"],
                    "face_lpips": metrics["face_lpips"],
                    "metric_status": status,
                },
            }
            atomic_json(stage / frame_id / "roi_rescore_receipt.json", receipt)
            new_results[frame_id] = result
            if result["visual_status"] == "pass" and status == "pass":
                accepted.append(result)

        correction_root.mkdir(parents=True)
        contact_sheet = None
        if args.preview_contact_sheet is not None:
            source_sheet = args.preview_contact_sheet.expanduser().resolve()
            if not source_sheet.is_file():
                raise FileNotFoundError(source_sheet)
            published_sheet = output_root / "contact_sheets" / f"{args.correction_id}_gt_roi_overlays.png"
            atomic_copy(source_sheet, published_sheet)
            atomic_copy(source_sheet, correction_root / "gt_roi_overlay_contact_sheet.png")
            contact_sheet = {
                "path": str(published_sheet),
                "sha256": sha256(published_sheet),
                "selection_surface": "heldout_ground_truth_only",
            }
        backup_rows = []
        for frame_id in frames:
            final = output_root / "frames" / frame_id
            backup = correction_root / "before" / frame_id
            for relative in MUTATED_RELATIVE_PATHS:
                row = copy_old_file(final / relative, backup / relative)
                if row is not None:
                    backup_rows.append({"frame_id": frame_id, **row})
            config_polygon = output_root / "config/face_polygons" / f"{frame_id}.json"
            row = copy_old_file(config_polygon, backup / "config_face_polygon.json")
            if row is not None:
                backup_rows.append({"frame_id": frame_id, **row})

        for frame_id in frames:
            final = output_root / "frames" / frame_id
            frame_stage = stage / frame_id
            polygon = polygons_dir / f"{frame_id}.json"
            atomic_copy(polygon, output_root / "config/face_polygons" / f"{frame_id}.json")
            atomic_copy(polygon, final / "metrics/face_polygons.json")
            for name in ("face_mask.png", "face_mask_overlay.png"):
                atomic_copy(frame_stage / name, final / "visual" / name)
            atomic_copy(frame_stage / "metrics.json", final / "metrics.json")
            atomic_copy(frame_stage / "roi_rescore_receipt.json", final / "metrics/roi_rescore_receipt.json")
            atomic_copy(frame_stage / "result.json", final / "result.json")

        csv_rows = [
            {key: new_results.get(frame_id, load_json(output_root / "frames" / frame_id / "result.json"))[key] for key in CSV_FIELDS}
            for frame_id in ordered
            if (output_root / "frames" / frame_id / "result.json").is_file()
        ]
        atomic_csv(output_root / "metrics.csv", csv_rows)
        manifest_path = output_root / "campaign_manifest.json"
        manifest = load_json(manifest_path)
        manifest["initial_baseline"] = {
            "frame_ids": ordered[:3],
            "metrics": [
                {
                    "frame_id": frame_id,
                    **{key: new_results[frame_id][key] for key in ("face_psnr", "face_ssim", "face_lpips")},
                }
                for frame_id in ordered[:3]
            ],
        }
        manifest["regression_thresholds"] = thresholds
        for frame_id in frames:
            state = dict(manifest["frame_states"][frame_id])
            state.update(
                {
                    "state": new_results[frame_id]["status"],
                    "metric_status": new_results[frame_id]["metric_status"],
                    "metric_correction_id": args.correction_id,
                    "updated_at": now(),
                }
            )
            manifest["frame_states"][frame_id] = state

        correction_manifest = {
            "schema_version": 1,
            "correction_id": args.correction_id,
            "completed_at": now(),
            "reason": args.reason,
            "request_sha256": request["request_sha256"],
            "frozen_scorer": str(scorer),
            "frozen_scorer_sha256": sha256(scorer),
            "migration_script": str(Path(__file__).resolve()),
            "migration_script_sha256": sha256(Path(__file__).resolve()),
            "frame_ids": frames,
            "frame_count": len(frames),
            "old_files": backup_rows,
            "new_metrics": [
                {key: new_results[frame_id][key] for key in ("frame_id", "face_psnr", "face_ssim", "face_lpips", "metric_status")}
                for frame_id in frames
            ],
            "regression_thresholds": thresholds,
            "geometry_or_render_modified": False,
            "selection_used_prediction": False,
            "gt_roi_overlay_contact_sheet": contact_sheet,
            "status": "complete",
        }
        atomic_json(correction_root / "correction_manifest.json", correction_manifest)
        manifest.setdefault("metric_protocol_corrections", []).append(
            {
                "correction_id": args.correction_id,
                "manifest": str(correction_root / "correction_manifest.json"),
                "manifest_sha256": sha256(correction_root / "correction_manifest.json"),
                "frame_count": len(frames),
                "reason": args.reason,
            }
        )
        manifest["updated_at"] = now()
        atomic_json(manifest_path, manifest)
        os.replace(stage, correction_root / "staged_new_scores")
        print(f"complete correction={args.correction_id} frames={len(frames)} csv_rows={len(csv_rows)}")
    finally:
        lock.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50"),
    )
    parser.add_argument("--polygons-dir", type=Path, required=True)
    parser.add_argument("--correction-id", required=True)
    parser.add_argument("--frames", nargs="*")
    parser.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preview-contact-sheet", type=Path)
    parser.add_argument(
        "--reason",
        default=(
            "The original static face polygon included a moving neck/background wedge despite "
            "declaring neck excluded; replace it with manually verified heldout-GT-only ROIs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    rescore(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
