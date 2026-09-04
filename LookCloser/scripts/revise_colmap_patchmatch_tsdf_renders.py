#!/usr/bin/env python3
"""Version and publish a uniform hard-texture render correction.

The immutable campaign request and remote PatchMatch/TSDF results remain
untouched.  ``prepare`` recreates the exact temporary JPEG texture inputs,
raycasts an already-published mesh, renders with the opt-in primary-colour
continuation rule, rescoring the same held-out-GT-only face polygon.  It stages
the complete published prefix before changing any result.  ``review`` records
per-frame LLM verdicts after the staged contact sheets have been inspected.
``publish`` replaces each frame directory by atomic renames, moves the entire
superseded directory into the correction archive, and rebuilds CSV/manifest
state atomically.  The ``extend-*`` actions apply the already-validated
correction to the next newly reconstructed frame before that frame is first
published.  Interrupted publication is resumable from per-frame state.
"""

from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from PIL import Image, ImageDraw

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


SCRIPT_DIR = Path(__file__).resolve().parent
CORRECTION_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,63}")
RENDER_RELATIVE_PATHS = (
    Path("render/eval_pred_0000.exr"),
    Path("render/eval_pred_0000.png"),
    Path("render/reprojection_audit.json"),
    Path("render/source_selection.png"),
)
VISUAL_NAMES = (
    "face_mask.png",
    "face_mask_overlay.png",
    "face_ear_hair_gt_pred.png",
    "ear_native_gt_pred.png",
    "lipstick_lips_hand_gt_pred.png",
    "actor_overview_gt_pred.png",
)
RENDER_ARGUMENTS = {
    "neighbors": 16,
    "aggregation_mode": "nearest-fill",
    "depth_log_tolerance": 0.01,
    "depth_hole_fill_max_area": 1000,
    "target_depth_component_min_area": 1000,
    "target_depth_component_max_log_jump": 0.0075,
    "nearest_fill_color_continuity": True,
    "nearest_fill_color_continuity_mode": "global",
    "nearest_fill_rank_penalty": 0.0,
    "primary_color_continuation": True,
    "primary_color_continuation_min_area": 20,
    "primary_color_continuation_max_area": 1000,
    "primary_color_continuation_min_median_l1": 0.1,
    "uses_masks": False,
    "uses_eval_rgb_for_prediction": False,
    "averages_sources": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copy hash mismatch: {source} -> {destination}")
    os.replace(temporary, destination)


def copy_tree_content(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        else:
            raise ValueError(f"Unsupported path in frame tree: {path}")


def reject_live_controller(root: Path) -> None:
    lock = root / ".campaign_controller.lock"
    if not lock.is_file():
        return
    try:
        pid = int(lock.read_text(encoding="utf-8").split()[0].removeprefix("pid="))
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        lock.unlink(missing_ok=True)
        return
    raise RuntimeError(f"Campaign controller is active with pid={pid}: {lock}")


class RevisionLock:
    def __init__(self, root: Path):
        self.path = root / ".campaign_render_revision.lock"

    def __enter__(self) -> None:
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise RuntimeError(f"Another render revision is active: {self.path}") from error
        os.write(descriptor, f"pid={os.getpid()} started_at={now()}\n".encode())
        os.close(descriptor)

    def __exit__(self, *_: object) -> None:
        self.path.unlink(missing_ok=True)


def validate_request(root: Path) -> tuple[dict, dict[str, Path]]:
    request = load_json(root / "campaign_request.json")
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    if request.get("request_sha256") != canonical_sha256(body):
        raise ValueError("campaign_request.json self-hash is invalid")
    inventory = {row["name"]: row for row in request["scripts"]}
    paths = {}
    for name in ("convert_exr_nerfstudio_to_jpeg.py", "score_colmap_patchmatch_tsdf_face.py"):
        path = root / "config/code" / name
        if name not in inventory or not path.is_file() or sha256(path) != inventory[name]["sha256"]:
            raise ValueError(f"Frozen campaign script hash mismatch: {name}")
        paths[name] = path
    return request, paths


def published_prefix(root: Path, ordered: list[str]) -> list[str]:
    frames = [frame_id for frame_id in ordered if (root / "frames" / frame_id).is_dir()]
    if frames != ordered[: len(frames)]:
        raise ValueError("Published frames are not a contiguous ordered prefix")
    return frames


def selected_prefix(root: Path, request: dict, values: list[str] | None) -> list[str]:
    frames = published_prefix(root, list(request["ordered_frame_ids"]))
    if values is not None and values != frames:
        raise ValueError("A render revision must cover the complete published ordered prefix")
    if not frames:
        raise ValueError("No published frames to revise")
    return frames


def revision_paths(root: Path, correction_id: str) -> tuple[Path, Path]:
    return (
        root / ".work" / f"render_revision_{correction_id}",
        root / ".diagnostics" / f"render_revision_{correction_id}",
    )


def extension_paths(root: Path, correction_id: str, frame_id: str) -> tuple[Path, Path]:
    return (
        root / ".work" / f"render_revision_{correction_id}_extensions" / frame_id,
        root / ".diagnostics" / f"render_revision_{correction_id}_extensions" / frame_id,
    )


def next_unpublished_frame(root: Path, request: dict, frame_id: str) -> None:
    ordered = list(request["ordered_frame_ids"])
    frames = published_prefix(root, ordered)
    if len(frames) >= len(ordered):
        raise ValueError("The campaign already published every requested frame")
    expected = ordered[len(frames)]
    if frame_id != expected:
        raise ValueError(f"Render revision extension must target next frame {expected}, got {frame_id}")
    retained = root / ".work" / "frames" / frame_id / "retained"
    if not retained.is_dir():
        raise FileNotFoundError(f"No reconstructed retained tree for {frame_id}: {retained}")
    validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))


def active_correction(root: Path, request: dict, correction_id: str) -> tuple[Path, dict]:
    correction = root / ".diagnostics" / f"render_revision_{correction_id}"
    manifest = load_json(correction / "correction_manifest.json")
    state = load_json(correction / "state.json")
    if manifest.get("status") != "complete" or state.get("status") != "complete":
        raise ValueError(f"Base render correction is not complete: {correction}")
    if manifest.get("correction_id") != correction_id:
        raise ValueError("Base render correction ID mismatch")
    if manifest.get("campaign_request_sha256") != request["request_sha256"]:
        raise ValueError("Base render correction belongs to another campaign request")
    if manifest.get("render_arguments") != RENDER_ARGUMENTS:
        raise ValueError("Active render arguments differ from the base correction")
    revision_request = manifest.get("revision_request", {})
    if revision_request.get("renderer_sha256") != sha256(SCRIPT_DIR / "render_mesh_image_blend.py"):
        raise ValueError("Renderer hash differs from the validated base correction")
    if revision_request.get("raycaster_sha256") != sha256(SCRIPT_DIR / "render_tsdf_mesh_depth.py"):
        raise ValueError("Raycaster hash differs from the validated base correction")
    return correction, manifest


def run_logged(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(command, check=True, text=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def verified_jpegs(frame: Path, converted: Path) -> None:
    rows = load_json(frame / "staging_manifest.json")["conversion_rows"]
    if len(rows) != 63:
        raise ValueError("Expected 63 retained staging conversion rows")
    for row in rows:
        relative = Path(str(row["frame_file_path"])).with_suffix(".jpg")
        image = converted / relative
        if not image.is_file() or sha256(image) != row["sha256"]:
            raise ValueError(f"Recreated JPEG mismatch: {image}")


def make_texture_dataset(frame: Path, converted: Path, output: Path) -> None:
    (output / "images").mkdir(parents=True)
    shutil.copyfile(frame / "texture/transforms.json", output / "transforms.json")
    subset = load_json(frame / "texture/angular_subset_manifest.json")
    names = [Path(row["file_path"]) for row in subset["selected_train_frames"]]
    names.append(Path("images/frame_eval_00001.jpg"))
    if len(names) != 17 or len(set(names)) != 17:
        raise ValueError("Texture dataset must contain 16 unique train images plus one eval image")
    for relative in names:
        source = converted / relative
        destination = output / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        os.link(source, destination)
    if sha256(output / "transforms.json") != subset["output_transforms_sha256"]:
        raise ValueError("Texture transforms hash differs from retained angular subset")


def finite_render(path: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (1080, 1920, 3) or not np.isfinite(image).all():
        raise ValueError(f"Invalid 1920x1080 finite render: {path}")


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


def prepared_files(stage: Path) -> list[Path]:
    return [
        *(stage / relative for relative in RENDER_RELATIVE_PATHS),
        stage / "metrics.json",
        *(stage / "visual" / name for name in VISUAL_NAMES),
    ]


def validate_prepared(stage: Path, manifest: dict) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError(f"Prepared manifest has no files: {stage}")
    for row in rows:
        path = stage / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Prepared file mismatch: {path}")
    finite_render(stage / "render/eval_pred_0000.exr")
    metrics = load_json(stage / "metrics.json")
    if not all(math.isfinite(float(metrics[key])) for key in ("face_psnr", "face_ssim", "face_lpips")):
        raise ValueError("Prepared face metric is non-finite")
    audit = load_json(stage / "render/reprojection_audit.json")
    if audit.get("uses_masks") is not False or audit.get("uses_eval_rgb_for_prediction") is not False:
        raise ValueError("Revised render used a mask or eval RGB")
    continuation = audit.get("nearest_fill_primary_color_continuation", {})
    if continuation.get("enabled") is not True:
        raise ValueError("Prepared render did not enable primary-colour continuation")


def prepare_frame(
    args: argparse.Namespace,
    request: dict,
    frozen: dict[str, Path],
    frame_id: str,
    work: Path,
) -> dict:
    final = args.output_root / "frames" / frame_id
    frame_work = work / "frames" / frame_id
    prepared = frame_work / "prepared"
    manifest_path = prepared / "prepared_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        validate_prepared(prepared, manifest)
        return manifest
    if frame_work.exists():
        quarantine = work / "quarantine" / f"{frame_id}_{int(datetime.now().timestamp())}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(frame_work, quarantine)
    scratch = frame_work / "scratch"
    converted = scratch / "jpeg_full"
    texture = scratch / "texture_data"
    mesh_depth = scratch / "mesh_depth"
    render = scratch / "render"
    score = scratch / "score"
    logs = frame_work / "logs"
    frame_work.mkdir(parents=True)
    validate_hash_manifest(final, load_json(final / "retained_manifest.json"))
    old_result = load_json(final / "result.json")
    if old_result.get("request_sha256") != request["request_sha256"]:
        raise ValueError(f"Request hash mismatch in published result {frame_id}")
    run_logged(
        [
            sys.executable,
            str(frozen["convert_exr_nerfstudio_to_jpeg.py"]),
            "--input", str(Path(request["source_root"]) / frame_id),
            "--output", str(converted),
            "--middle-gray", "0.18",
            "--exposure-mode", "per-image",
            "--exposure-percentile", "70",
            "--quality", "98",
            "--resume",
        ],
        logs / "convert.log",
    )
    verified_jpegs(final, converted)
    make_texture_dataset(final, converted, texture)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    run_logged(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_tsdf_mesh_depth.py"),
            "--data", str(texture),
            "--mesh", str(final / "mesh/colmap_patchmatch_tsdf.ply"),
            "--output-dir", str(mesh_depth),
            "--eval-mode", "filename",
            "--orientation-method", "up",
            "--center-method", "focus",
            "--auto-scale-poses",
            "--scale-factor", "1",
            "--scene-scale", "2",
            "--downscale-factor", "1",
        ],
        logs / "raycast.log",
        env=env,
    )
    run_logged(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_mesh_image_blend.py"),
            "--data", str(texture),
            "--mesh-depth-manifest", str(mesh_depth / "mesh_depth_manifest.json"),
            "--output-dir", str(render),
            "--neighbors", "16",
            "--aggregation-modes", "nearest-fill",
            "--blend-alphas", "1",
            "--depth-log-tolerance", "0.01",
            "--depth-hole-fill-max-area", "1000",
            "--depth-hole-fill-boundary-radius", "4",
            "--depth-hole-fill-max-relative-plane-rmse", "0.015",
            "--target-depth-component-min-area", "1000",
            "--target-depth-component-max-log-jump", "0.0075",
            "--nearest-fill-color-continuity",
            "--nearest-fill-color-continuity-mode", "global",
            "--nearest-fill-rank-penalty", "0",
            "--nearest-fill-primary-color-continuation",
            "--nearest-fill-primary-color-continuation-min-area", "20",
            "--nearest-fill-primary-color-continuation-max-area", "1000",
            "--nearest-fill-primary-color-continuation-min-median-l1", "0.1",
            "--eval-mode", "filename",
            "--orientation-method", "up",
            "--center-method", "focus",
            "--auto-scale-poses",
            "--scale-factor", "1",
            "--scene-scale", "2",
            "--downscale-factor", "1",
            "--device", "cuda",
        ],
        logs / "render.log",
        env=env,
    )
    new_variant = render / "nearest_fill16"
    if sha256(render / "eval_gt_0000.exr") != sha256(final / "render/eval_gt_0000.exr"):
        raise ValueError(f"Recreated held-out GT differs for {frame_id}")
    run_logged(
        [
            sys.executable,
            str(frozen["score_colmap_patchmatch_tsdf_face.py"]),
            "--frame-id", frame_id,
            "--prediction", str(new_variant / "eval_pred_0000.exr"),
            "--ground-truth", str(final / "render/eval_gt_0000.exr"),
            "--face-polygons", str(final / "metrics/face_polygons.json"),
            "--output-dir", str(score),
            "--device", args.metric_device,
        ],
        logs / "score.log",
        env=env,
    )
    prepared.mkdir(parents=True)
    for source, relative in (
        (new_variant / "eval_pred_0000.exr", Path("render/eval_pred_0000.exr")),
        (new_variant / "eval_pred_0000.png", Path("render/eval_pred_0000.png")),
        (new_variant / "source_selection.png", Path("render/source_selection.png")),
        (render / "reprojection_audit.json", Path("render/reprojection_audit.json")),
    ):
        atomic_copy(source, prepared / relative)
    for name in VISUAL_NAMES:
        atomic_copy(score / name, prepared / "visual" / name)
    metrics = rewrite_metric_paths(load_json(score / "metrics.json"), final)
    atomic_json(prepared / "metrics.json", metrics)
    rows = []
    for path in prepared_files(prepared):
        rows.append(
            {
                "path": str(path.relative_to(prepared)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    audit = load_json(prepared / "render/reprojection_audit.json")
    variant = next(row for row in audit["variants"] if row["name"] == "nearest_fill16")
    manifest = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "frame_id": frame_id,
        "prepared_at": now(),
        "request_sha256": request["request_sha256"],
        "mesh_sha256": sha256(final / "mesh/colmap_patchmatch_tsdf.ply"),
        "ground_truth_sha256": sha256(final / "render/eval_gt_0000.exr"),
        "face_polygons_sha256": sha256(final / "metrics/face_polygons.json"),
        "before": {
            "render_exr_sha256": sha256(final / "render/eval_pred_0000.exr"),
            "render_png_sha256": sha256(final / "render/eval_pred_0000.png"),
            "face_psnr": old_result["face_psnr"],
            "face_ssim": old_result["face_ssim"],
            "face_lpips": old_result["face_lpips"],
        },
        "after": {
            "render_exr_sha256": sha256(prepared / "render/eval_pred_0000.exr"),
            "render_png_sha256": sha256(prepared / "render/eval_pred_0000.png"),
            "face_psnr": metrics["face_psnr"],
            "face_ssim": metrics["face_ssim"],
            "face_lpips": metrics["face_lpips"],
            "primary_color_continuation": variant["primary_color_continuation"],
        },
        "files": rows,
    }
    atomic_json(manifest_path, manifest)
    validate_prepared(prepared, manifest)
    shutil.rmtree(scratch)
    print(f"prepared frame={frame_id}", flush=True)
    return manifest


def prepare_extension(args: argparse.Namespace) -> None:
    request, frozen = validate_request(args.output_root)
    active_correction(args.output_root, request, args.correction_id)
    next_unpublished_frame(args.output_root, request, args.frame_id)
    retained = args.output_root / ".work" / "frames" / args.frame_id / "retained"
    work, published = extension_paths(args.output_root, args.correction_id, args.frame_id)
    if published.exists():
        raise FileExistsError(published)
    polygon = args.output_root / "config" / "face_polygons" / f"{args.frame_id}.json"
    if not polygon.is_file():
        raise FileNotFoundError(f"Held-out-GT-only face polygon is required: {polygon}")
    request_payload = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "campaign_request_sha256": request["request_sha256"],
        "frame_id": args.frame_id,
        "render_arguments": RENDER_ARGUMENTS,
        "renderer_sha256": sha256(SCRIPT_DIR / "render_mesh_image_blend.py"),
        "raycaster_sha256": sha256(SCRIPT_DIR / "render_tsdf_mesh_depth.py"),
        "migration_script_sha256": sha256(Path(__file__).resolve()),
        "face_polygons_sha256": sha256(polygon),
    }
    request_payload["request_sha256"] = canonical_sha256(request_payload)
    request_path = work / "extension_request.json"
    prepared = work / "prepared"
    prepared_manifest = prepared / "prepared_manifest.json"
    if prepared_manifest.is_file():
        if load_json(request_path) != request_payload:
            raise ValueError("Refusing to resume: extension request or code hashes changed")
        manifest = load_json(prepared_manifest)
        validate_prepared(prepared, manifest)
        print(f"frame={args.frame_id} status=skipped-extension-prepared", flush=True)
        return
    if work.exists():
        if request_path.is_file():
            if load_json(request_path) != request_payload:
                raise ValueError("Refusing to resume: extension request or code hashes changed")
        elif any(work.iterdir()):
            raise RuntimeError(f"Cannot resume extension without {request_path}")
    else:
        work.mkdir(parents=True)
    atomic_json(request_path, request_payload)

    scratch = work / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    if prepared.exists():
        quarantine = work / f"quarantine_prepared_{int(datetime.now().timestamp())}"
        os.replace(prepared, quarantine)
    converted = scratch / "jpeg_full"
    texture = scratch / "texture_data"
    mesh_depth = scratch / "mesh_depth"
    render = scratch / "render"
    score = scratch / "score"
    logs = work / "logs"
    run_logged(
        [
            sys.executable,
            str(frozen["convert_exr_nerfstudio_to_jpeg.py"]),
            "--input", str(Path(request["source_root"]) / args.frame_id),
            "--output", str(converted),
            "--middle-gray", "0.18",
            "--exposure-mode", "per-image",
            "--exposure-percentile", "70",
            "--quality", "98",
            "--resume",
        ],
        logs / "convert.log",
    )
    verified_jpegs(retained, converted)
    make_texture_dataset(retained, converted, texture)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    run_logged(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_tsdf_mesh_depth.py"),
            "--data", str(texture),
            "--mesh", str(retained / "mesh/colmap_patchmatch_tsdf.ply"),
            "--output-dir", str(mesh_depth),
            "--eval-mode", "filename",
            "--orientation-method", "up",
            "--center-method", "focus",
            "--auto-scale-poses",
            "--scale-factor", "1",
            "--scene-scale", "2",
            "--downscale-factor", "1",
        ],
        logs / "raycast.log",
        env=env,
    )
    run_logged(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_mesh_image_blend.py"),
            "--data", str(texture),
            "--mesh-depth-manifest", str(mesh_depth / "mesh_depth_manifest.json"),
            "--output-dir", str(render),
            "--neighbors", "16",
            "--aggregation-modes", "nearest-fill",
            "--blend-alphas", "1",
            "--depth-log-tolerance", "0.01",
            "--depth-hole-fill-max-area", "1000",
            "--depth-hole-fill-boundary-radius", "4",
            "--depth-hole-fill-max-relative-plane-rmse", "0.015",
            "--target-depth-component-min-area", "1000",
            "--target-depth-component-max-log-jump", "0.0075",
            "--nearest-fill-color-continuity",
            "--nearest-fill-color-continuity-mode", "global",
            "--nearest-fill-rank-penalty", "0",
            "--nearest-fill-primary-color-continuation",
            "--nearest-fill-primary-color-continuation-min-area", "20",
            "--nearest-fill-primary-color-continuation-max-area", "1000",
            "--nearest-fill-primary-color-continuation-min-median-l1", "0.1",
            "--eval-mode", "filename",
            "--orientation-method", "up",
            "--center-method", "focus",
            "--auto-scale-poses",
            "--scale-factor", "1",
            "--scene-scale", "2",
            "--downscale-factor", "1",
            "--device", "cuda",
        ],
        logs / "render.log",
        env=env,
    )
    new_variant = render / "nearest_fill16"
    if sha256(render / "eval_gt_0000.exr") != sha256(retained / "render/eval_gt_0000.exr"):
        raise ValueError(f"Recreated held-out GT differs for {args.frame_id}")
    run_logged(
        [
            sys.executable,
            str(frozen["score_colmap_patchmatch_tsdf_face.py"]),
            "--frame-id", args.frame_id,
            "--prediction", str(new_variant / "eval_pred_0000.exr"),
            "--ground-truth", str(retained / "render/eval_gt_0000.exr"),
            "--face-polygons", str(polygon),
            "--output-dir", str(score),
            "--device", args.metric_device,
        ],
        logs / "score.log",
        env=env,
    )
    prepared.mkdir(parents=True)
    for source, relative in (
        (new_variant / "eval_pred_0000.exr", Path("render/eval_pred_0000.exr")),
        (new_variant / "eval_pred_0000.png", Path("render/eval_pred_0000.png")),
        (new_variant / "source_selection.png", Path("render/source_selection.png")),
        (render / "reprojection_audit.json", Path("render/reprojection_audit.json")),
    ):
        atomic_copy(source, prepared / relative)
    for name in VISUAL_NAMES:
        atomic_copy(score / name, prepared / "visual" / name)
    future_final = args.output_root / "frames" / args.frame_id
    metrics = rewrite_metric_paths(load_json(score / "metrics.json"), future_final)
    atomic_json(prepared / "metrics.json", metrics)
    rows = [
        {"path": str(path.relative_to(prepared)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in prepared_files(prepared)
    ]
    audit = load_json(prepared / "render/reprojection_audit.json")
    variant = next(row for row in audit["variants"] if row["name"] == "nearest_fill16")
    remote = load_json(retained / "remote_result.json")
    manifest = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "frame_id": args.frame_id,
        "prepared_at": now(),
        "request_sha256": request["request_sha256"],
        "extension_request_sha256": request_payload["request_sha256"],
        "mesh_sha256": sha256(retained / "mesh/colmap_patchmatch_tsdf.ply"),
        "ground_truth_sha256": sha256(retained / "render/eval_gt_0000.exr"),
        "face_polygons_sha256": sha256(polygon),
        "before": {
            "render_exr_sha256": sha256(retained / "render/eval_pred_0000.exr"),
            "render_png_sha256": sha256(retained / "render/eval_pred_0000.png"),
            "remote_render_exr_sha256": remote["render_exr_sha256"],
            "remote_render_png_sha256": remote["render_sha256"],
        },
        "after": {
            "render_exr_sha256": sha256(prepared / "render/eval_pred_0000.exr"),
            "render_png_sha256": sha256(prepared / "render/eval_pred_0000.png"),
            "face_psnr": metrics["face_psnr"],
            "face_ssim": metrics["face_ssim"],
            "face_lpips": metrics["face_lpips"],
            "primary_color_continuation": variant["primary_color_continuation"],
        },
        "files": rows,
    }
    atomic_json(prepared_manifest, manifest)
    validate_prepared(prepared, manifest)
    shutil.rmtree(scratch)
    print(f"frame={args.frame_id} status=extension-prepared", flush=True)


def review_extension(args: argparse.Namespace) -> None:
    request, _ = validate_request(args.output_root)
    active_correction(args.output_root, request, args.correction_id)
    next_unpublished_frame(args.output_root, request, args.frame_id)
    work, published = extension_paths(args.output_root, args.correction_id, args.frame_id)
    if published.exists():
        raise FileExistsError(published)
    prepared = work / "prepared"
    validate_prepared(prepared, load_json(prepared / "prepared_manifest.json"))
    if args.visual_status == "pass" and (args.ear_artifact or args.lipstick_artifact):
        raise ValueError("A visual pass cannot declare an artifact")
    crops = []
    for crop in ("face_ear_hair", "ear_native", "lipstick_lips_hand", "actor_overview"):
        path = prepared / "visual" / f"{crop}_gt_pred.png"
        crops.append({"crop": crop, "path": str(path), "sha256": sha256(path)})
    receipt = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "frame_id": args.frame_id,
        "reviewed_at": now(),
        "comparison": "heldout_gt_vs_revised_prediction",
        "reviewed_crops": crops,
        "background_ignored": True,
        "silhouette_holes_ignored": False,
        "visual_status": args.visual_status,
        "ear_artifact": args.ear_artifact,
        "lipstick_artifact": args.lipstick_artifact,
        "visual_notes": args.visual_notes,
    }
    atomic_json(work / "visual_review.json", receipt)
    print(f"frame={args.frame_id} status=extension-reviewed verdict={args.visual_status}", flush=True)


def labeled(image: Image.Image, frame_id: str) -> Image.Image:
    bar = 30
    result = Image.new("RGB", (image.width, image.height + bar), "white")
    result.paste(image, (0, bar))
    ImageDraw.Draw(result).text((8, 8), f"{frame_id}  held-out GT | revised prediction", fill="black")
    return result


def contact_sheets(work: Path, frames: list[str]) -> list[dict[str, object]]:
    output = work / "contact_sheets"
    output.mkdir(exist_ok=True)
    rows = []
    for start in range(0, len(frames), 10):
        batch = frames[start : start + 10]
        for crop in ("ear_native", "lipstick_lips_hand", "face_ear_hair", "actor_overview"):
            images = [
                labeled(
                    Image.open(work / "frames" / frame_id / "prepared/visual" / f"{crop}_gt_pred.png").convert("RGB"),
                    frame_id,
                )
                for frame_id in batch
            ]
            columns = 2
            width = max(image.width for image in images)
            height = max(image.height for image in images)
            sheet = Image.new(
                "RGB",
                (columns * width, math.ceil(len(images) / columns) * height),
                (32, 32, 32),
            )
            for index, image in enumerate(images):
                sheet.paste(image, ((index % columns) * width, (index // columns) * height))
            path = output / f"{batch[0]}_{batch[-1]}_{crop}_gt_pred.png"
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            sheet.save(temporary, format="PNG", compress_level=3)
            os.replace(temporary, path)
            rows.append(
                {"frames": batch, "crop": crop, "path": str(path), "sha256": sha256(path)}
            )
    return rows


def prepare(args: argparse.Namespace) -> None:
    request, frozen = validate_request(args.output_root)
    frames = selected_prefix(args.output_root, request, args.frames)
    work, correction = revision_paths(args.output_root, args.correction_id)
    if correction.exists():
        raise FileExistsError(correction)
    renderer = SCRIPT_DIR / "render_mesh_image_blend.py"
    raycaster = SCRIPT_DIR / "render_tsdf_mesh_depth.py"
    request_payload = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "campaign_request_sha256": request["request_sha256"],
        "frame_ids": frames,
        "render_arguments": RENDER_ARGUMENTS,
        "renderer": str(renderer),
        "renderer_sha256": sha256(renderer),
        "raycaster": str(raycaster),
        "raycaster_sha256": sha256(raycaster),
        "migration_script": str(Path(__file__).resolve()),
        "migration_script_sha256": sha256(Path(__file__).resolve()),
    }
    request_payload["request_sha256"] = canonical_sha256(request_payload)
    request_path = work / "revision_request.json"
    if request_path.is_file():
        if load_json(request_path) != request_payload:
            raise ValueError("Refusing to resume: render revision request or code hashes changed")
    else:
        if work.exists() and any(work.iterdir()):
            raise RuntimeError(f"Cannot resume render revision without {request_path}")
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(request_path, request_payload)
    manifests = []
    for frame_id in frames:
        manifests.append(prepare_frame(args, request, frozen, frame_id, work))
    sheets = contact_sheets(work, frames)
    payload = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_at": now(),
        "revision_request_sha256": request_payload["request_sha256"],
        "frame_ids": frames,
        "frame_count": len(frames),
        "frames": manifests,
        "contact_sheets": sheets,
    }
    atomic_json(work / "prepare_manifest.json", payload)
    print(f"complete status=prepared correction={args.correction_id} frames={len(frames)}", flush=True)


def review(args: argparse.Namespace) -> None:
    request, _ = validate_request(args.output_root)
    frames = selected_prefix(args.output_root, request, args.frames)
    work, correction = revision_paths(args.output_root, args.correction_id)
    if correction.exists():
        raise FileExistsError(correction)
    prepared = load_json(work / "prepare_manifest.json")
    if prepared.get("status") != "prepared" or prepared.get("frame_ids") != frames:
        raise ValueError("Prepared revision inventory mismatch")
    if args.visual_status == "pass" and (args.ear_artifact or args.lipstick_artifact):
        raise ValueError("A visual pass cannot declare an artifact")
    for frame_id in frames:
        relevant = [row for row in prepared["contact_sheets"] if frame_id in row["frames"]]
        if len(relevant) != 4 or any(sha256(Path(row["path"])) != row["sha256"] for row in relevant):
            raise ValueError(f"Contact-sheet inventory/hash mismatch for {frame_id}")
        receipt = {
            "schema_version": 1,
            "correction_id": args.correction_id,
            "frame_id": frame_id,
            "reviewed_at": now(),
            "comparison": "heldout_gt_vs_revised_prediction",
            "reviewed_crops": ["face_ear_hair", "ear_native", "lipstick_lips_hand", "actor_overview"],
            "contact_sheets": [{"path": row["path"], "sha256": row["sha256"]} for row in relevant],
            "background_ignored": True,
            "silhouette_holes_ignored": False,
            "visual_status": args.visual_status,
            "ear_artifact": args.ear_artifact,
            "lipstick_artifact": args.lipstick_artifact,
            "visual_notes": args.visual_notes,
        }
        atomic_json(work / "reviews" / f"{frame_id}.json", receipt)
    print(f"complete status=reviewed correction={args.correction_id} frames={len(frames)}", flush=True)


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


def refresh_retained_manifest(frame: Path) -> None:
    path = frame / "retained_manifest.json"
    manifest = load_json(path)
    by_path = {row["path"]: row for row in manifest["files"]}
    for relative in RENDER_RELATIVE_PATHS:
        token = str(relative)
        if token not in by_path:
            raise ValueError(f"Retained manifest lacks {token}")
        target = frame / relative
        by_path[token].update({"bytes": target.stat().st_size, "sha256": sha256(target)})
    atomic_json(path, manifest)
    validate_hash_manifest(frame, manifest)


def prior_accepted_results(root: Path, ordered: list[str], frame_id: str) -> list[dict]:
    rows = []
    for candidate in ordered[: ordered.index(frame_id)]:
        path = root / "frames" / candidate / "result.json"
        if path.is_file():
            row = load_json(path)
            if row.get("visual_status") == "pass" and row.get("metric_status") == "pass":
                rows.append(row)
    return rows[-5:]


def rebuild_campaign_csv(root: Path, ordered: list[str]) -> None:
    rows = []
    for frame_id in ordered:
        path = root / "frames" / frame_id / "result.json"
        if path.is_file():
            result = load_json(path)
            rows.append({key: result[key] for key in CSV_FIELDS})
    atomic_csv(root / "metrics.csv", rows)


def frozen_finalize_command(root: Path, request: dict, frame_id: str) -> list[str]:
    remote = request["remote"]
    return [
        sys.executable,
        str(root / "config/code/run_colmap_patchmatch_tsdf_campaign.py"),
        "finalize",
        "--source-root", request["source_root"],
        "--output-root", str(root),
        "--calibration-template", request["calibration_template_source"],
        "--remote-host", remote["host"],
        "--remote-scratch-root", remote["scratch_root"],
        "--remote-python", remote["python"],
        "--remote-colmap", remote["colmap"],
        "--gpu-index", str(remote["gpu_index"]),
        "--frames", frame_id,
    ]


def overlay_extension_tree(
    root: Path,
    frame_id: str,
    prepared: Path,
    review: dict,
    metrics: dict,
    destination: Path,
) -> None:
    for relative in RENDER_RELATIVE_PATHS:
        atomic_copy(prepared / relative, destination / relative)
    for name in VISUAL_NAMES:
        atomic_copy(prepared / "visual" / name, destination / "visual" / name)
    polygon = root / "config" / "face_polygons" / f"{frame_id}.json"
    atomic_copy(polygon, destination / "metrics/face_polygons.json")
    atomic_json(destination / "metrics.json", metrics)
    visual = {
        key: value for key, value in review.items()
        if key not in {"reviewed_crops"}
    }
    visual["reviewed_crops"] = ["face_ear_hair", "ear_native", "lipstick_lips_hand", "actor_overview"]
    visual["review_crop_hashes"] = {
        row["crop"]: row["sha256"] for row in review["reviewed_crops"]
    }
    atomic_json(destination / "visual_review.json", visual)
    refresh_retained_manifest(destination)


def publish_extension(args: argparse.Namespace) -> None:
    request, _ = validate_request(args.output_root)
    active_correction(args.output_root, request, args.correction_id)
    work, published = extension_paths(args.output_root, args.correction_id, args.frame_id)
    final = args.output_root / "frames" / args.frame_id
    if published.is_dir() and (published / "state.json").is_file():
        state = load_json(published / "state.json")
        if state.get("status") == "complete":
            result = load_json(final / "result.json")
            if result.get("render_revision", {}).get("correction_id") != args.correction_id:
                raise ValueError("Completed extension and published result disagree")
            rebuild_campaign_csv(args.output_root, list(request["ordered_frame_ids"]))
            print(f"frame={args.frame_id} status=skipped-extension-published", flush=True)
            return
    if not final.exists():
        next_unpublished_frame(args.output_root, request, args.frame_id)
    prepared = work / "prepared"
    prepared_manifest = load_json(prepared / "prepared_manifest.json")
    validate_prepared(prepared, prepared_manifest)
    extension_request = load_json(work / "extension_request.json")
    if extension_request.get("request_sha256") != prepared_manifest.get("extension_request_sha256"):
        raise ValueError("Prepared extension/request hash mismatch")
    review = load_json(work / "visual_review.json")
    if review.get("visual_status") not in {"pass", "fail"}:
        raise ValueError("A non-uncertain extension visual verdict is required")
    for row in review.get("reviewed_crops", []):
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Reviewed crop hash mismatch: {path}")

    manifest_path = args.output_root / "campaign_manifest.json"
    campaign_manifest = load_json(manifest_path)
    metrics = rewrite_metric_paths(load_json(prepared / "metrics.json"), final)
    previous = prior_accepted_results(args.output_root, list(request["ordered_frame_ids"]), args.frame_id)
    status, reasons = metric_status(metrics, campaign_manifest["regression_thresholds"], previous)
    metrics["metric_status"] = status
    metrics["regression_reasons"] = reasons
    metrics["regression_reference_frame_ids"] = [row["frame_id"] for row in previous]
    metrics["render_revision"] = {
        "correction_id": args.correction_id,
        "render_arguments": RENDER_ARGUMENTS,
        "extension": True,
    }

    retained = args.output_root / ".work" / "frames" / args.frame_id / "retained"
    staged_retained = retained.with_name(f".{retained.name}.render-revision-staged")
    before_retained = published / "before" / "retained"
    if not published.exists():
        published.mkdir(parents=True)
        atomic_json(
            published / "state.json",
            {
                "schema_version": 1,
                "status": "applying",
                "started_at": now(),
                "frame_id": args.frame_id,
                "correction_id": args.correction_id,
                "extension_request_sha256": extension_request["request_sha256"],
            },
        )
        for source in (
            work / "extension_request.json",
            prepared / "prepared_manifest.json",
            work / "visual_review.json",
        ):
            atomic_copy(source, published / source.name)

    if not final.exists() and not before_retained.exists():
        if staged_retained.exists():
            shutil.rmtree(staged_retained)
        copy_tree_content(retained, staged_retained)
        overlay_extension_tree(args.output_root, args.frame_id, prepared, review, metrics, staged_retained)
        validate_hash_manifest(staged_retained, load_json(staged_retained / "retained_manifest.json"))
        if sha256(staged_retained / "render/eval_pred_0000.png") != prepared_manifest["after"]["render_png_sha256"]:
            raise ValueError("Staged revised render differs from the prepared render")
        before_retained.parent.mkdir(parents=True, exist_ok=True)
        os.replace(retained, before_retained)
        os.replace(staged_retained, retained)
        atomic_json(
            published / "state.json",
            {
                "schema_version": 1,
                "status": "retained_revised",
                "updated_at": now(),
                "frame_id": args.frame_id,
                "correction_id": args.correction_id,
                "extension_request_sha256": extension_request["request_sha256"],
            },
        )
    elif not final.exists() and not retained.exists() and staged_retained.exists():
        os.replace(staged_retained, retained)
    if not final.exists():
        validate_hash_manifest(before_retained, load_json(before_retained / "retained_manifest.json"))
        validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
        if sha256(retained / "render/eval_pred_0000.png") != prepared_manifest["after"]["render_png_sha256"]:
            raise ValueError("Resumed retained tree does not contain the prepared revised render")
        run_logged(frozen_finalize_command(args.output_root, request, args.frame_id), published / "finalize.log")

    result = load_json(final / "result.json")
    if sha256(final / "render/eval_pred_0000.png") != prepared_manifest["after"]["render_png_sha256"]:
        raise ValueError("Published render differs from the prepared extension")
    result.update(
        {
            "render_sha256": prepared_manifest["after"]["render_png_sha256"],
            "face_psnr": metrics["face_psnr"],
            "face_ssim": metrics["face_ssim"],
            "face_lpips": metrics["face_lpips"],
            "metric_status": metrics["metric_status"],
            "visual_status": review["visual_status"],
            "ear_artifact": review["ear_artifact"],
            "lipstick_artifact": review["lipstick_artifact"],
            "visual_notes": review["visual_notes"],
        }
    )
    result["status"] = (
        "pass"
        if result["visual_status"] == "pass" and result["metric_status"] != "fail_nonfinite"
        else "fail"
    )
    result["render_revision"] = {
        "correction_id": args.correction_id,
        "published_at": now(),
        "reason": args.reason,
        "extension": True,
        "remote_render_preserved_in": str(before_retained / "render"),
        "renderer_sha256": sha256(SCRIPT_DIR / "render_mesh_image_blend.py"),
        "render_arguments": RENDER_ARGUMENTS,
        "before_render_sha256": prepared_manifest["before"]["render_png_sha256"],
        "after_render_sha256": prepared_manifest["after"]["render_png_sha256"],
    }
    atomic_json(final / "result.json", result)
    validate_hash_manifest(final, load_json(final / "retained_manifest.json"))
    if sha256(before_retained / "mesh/colmap_patchmatch_tsdf.ply") != result["mesh_sha256"]:
        raise ValueError("Mesh changed while applying render-only extension")

    receipt = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "frame_id": args.frame_id,
        "published_at": now(),
        "extension_request_sha256": extension_request["request_sha256"],
        "before_render_sha256": prepared_manifest["before"]["render_png_sha256"],
        "after_render_sha256": prepared_manifest["after"]["render_png_sha256"],
        "result_sha256": sha256(final / "result.json"),
        "mesh_unchanged": True,
        "eval_rgb_used_for_prediction": False,
        "masks_used": False,
        "source_colors_averaged": False,
    }
    atomic_json(published / "extension_receipt.json", receipt)
    extensions = campaign_manifest.setdefault("render_correction_extensions", [])
    extensions = [row for row in extensions if row.get("frame_id") != args.frame_id]
    extensions.append(
        {
            "correction_id": args.correction_id,
            "frame_id": args.frame_id,
            "receipt": str(published / "extension_receipt.json"),
            "receipt_sha256": sha256(published / "extension_receipt.json"),
        }
    )
    campaign_manifest["render_correction_extensions"] = extensions
    state = dict(campaign_manifest["frame_states"][args.frame_id])
    state.update(
        {
            "state": result["status"],
            "metric_status": result["metric_status"],
            "visual_status": result["visual_status"],
            "render_correction_id": args.correction_id,
            "updated_at": now(),
        }
    )
    campaign_manifest["frame_states"][args.frame_id] = state
    campaign_manifest["updated_at"] = now()
    atomic_json(manifest_path, campaign_manifest)
    rebuild_campaign_csv(args.output_root, list(request["ordered_frame_ids"]))
    atomic_json(
        published / "state.json",
        {
            "schema_version": 1,
            "status": "complete",
            "completed_at": now(),
            "frame_id": args.frame_id,
            "correction_id": args.correction_id,
            "extension_request_sha256": extension_request["request_sha256"],
            "extension_receipt_sha256": sha256(published / "extension_receipt.json"),
        },
    )
    if work.exists():
        os.replace(work, published / "prepared_and_reviewed")
    print(f"frame={args.frame_id} status=extension-published result={result['status']}", flush=True)


def update_frame_tree(
    args: argparse.Namespace,
    request: dict,
    frame_id: str,
    work: Path,
    correction: Path,
    metrics: dict,
    review_receipt: dict,
) -> dict:
    final = args.output_root / "frames" / frame_id
    before = correction / "before" / frame_id
    receipt_path = correction / "frame_receipts" / f"{frame_id}.json"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        result = load_json(final / "result.json")
        if result.get("render_revision", {}).get("correction_id") != args.correction_id:
            raise ValueError(f"Revision receipt/final mismatch for {frame_id}")
        return result
    if before.exists():
        raise RuntimeError(f"Interrupted frame transaction has backup but no receipt: {frame_id}")
    prepared = work / "frames" / frame_id / "prepared"
    validate_prepared(prepared, load_json(prepared / "prepared_manifest.json"))
    temporary = final.with_name(f".{frame_id}.render-revision.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    copy_tree_content(final, temporary)
    for relative in RENDER_RELATIVE_PATHS:
        atomic_copy(prepared / relative, temporary / relative)
    for name in VISUAL_NAMES:
        atomic_copy(prepared / "visual" / name, temporary / "visual" / name)
    atomic_json(temporary / "metrics.json", metrics)
    visual = dict(review_receipt)
    atomic_json(temporary / "visual_review.json", visual)
    old_result = load_json(final / "result.json")
    result = dict(old_result)
    for key in ("face_psnr", "face_ssim", "face_lpips", "metric_status"):
        result[key] = metrics[key]
    result.update(
        {
            "render_sha256": sha256(temporary / "render/eval_pred_0000.png"),
            "visual_status": visual["visual_status"],
            "ear_artifact": visual["ear_artifact"],
            "lipstick_artifact": visual["lipstick_artifact"],
            "visual_notes": visual["visual_notes"],
        }
    )
    result["status"] = (
        "pass"
        if result["visual_status"] == "pass" and result["metric_status"] != "fail_nonfinite"
        else "fail"
    )
    result["render_revision"] = {
        "correction_id": args.correction_id,
        "published_at": now(),
        "reason": args.reason,
        "remote_render_preserved_in": str(before / "render"),
        "renderer_sha256": sha256(SCRIPT_DIR / "render_mesh_image_blend.py"),
        "render_arguments": RENDER_ARGUMENTS,
        "before_render_sha256": old_result["render_sha256"],
        "after_render_sha256": result["render_sha256"],
    }
    refresh_retained_manifest(temporary)
    result["retained_manifest_sha256"] = sha256(temporary / "retained_manifest.json")
    atomic_json(temporary / "result.json", result)
    finite_render(temporary / "render/eval_pred_0000.exr")
    validate_hash_manifest(temporary, load_json(temporary / "retained_manifest.json"))
    before.parent.mkdir(parents=True, exist_ok=True)
    os.replace(final, before)
    try:
        os.replace(temporary, final)
    except Exception:
        os.replace(before, final)
        raise
    receipt = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "frame_id": frame_id,
        "published_at": now(),
        "before_result_sha256": sha256(before / "result.json"),
        "after_result_sha256": sha256(final / "result.json"),
        "before_render_sha256": sha256(before / "render/eval_pred_0000.png"),
        "after_render_sha256": sha256(final / "render/eval_pred_0000.png"),
        "mesh_unchanged": sha256(before / "mesh/colmap_patchmatch_tsdf.ply") == sha256(final / "mesh/colmap_patchmatch_tsdf.ply"),
    }
    atomic_json(receipt_path, receipt)
    return result


def publish(args: argparse.Namespace) -> None:
    request, _ = validate_request(args.output_root)
    frames = selected_prefix(args.output_root, request, args.frames)
    work, correction = revision_paths(args.output_root, args.correction_id)
    prepared = load_json(work / "prepare_manifest.json")
    revision_request = load_json(work / "revision_request.json")
    if prepared.get("status") != "prepared" or prepared.get("frame_ids") != frames:
        raise ValueError("Prepared revision inventory mismatch")
    reviews = {frame_id: load_json(work / "reviews" / f"{frame_id}.json") for frame_id in frames}
    if any(row.get("visual_status") not in {"pass", "fail"} for row in reviews.values()):
        raise ValueError("Every revised frame needs a non-uncertain visual verdict")
    if correction.exists():
        state = load_json(correction / "state.json")
        if state.get("status") == "complete":
            raise RuntimeError(f"Render revision is already complete: {correction}")
        if state.get("revision_request_sha256") != revision_request["request_sha256"]:
            raise ValueError("Existing correction state belongs to another request")
    else:
        correction.mkdir(parents=True)
        atomic_json(
            correction / "state.json",
            {
                "schema_version": 1,
                "status": "applying",
                "started_at": now(),
                "revision_request_sha256": revision_request["request_sha256"],
            },
        )
    raw_metrics = {
        frame_id: rewrite_metric_paths(
            load_json(work / "frames" / frame_id / "prepared/metrics.json"),
            args.output_root / "frames" / frame_id,
        )
        for frame_id in frames
    }
    thresholds = robust_initial_thresholds([raw_metrics[frame_id] for frame_id in frames[:3]])
    accepted: list[dict] = []
    results: dict[str, dict] = {}
    for frame_id in frames:
        metrics = raw_metrics[frame_id]
        status, reasons = metric_status(metrics, thresholds, accepted[-5:])
        metrics["metric_status"] = status
        metrics["regression_reasons"] = reasons
        metrics["regression_reference_frame_ids"] = [row["frame_id"] for row in accepted[-5:]]
        metrics["render_revision"] = {
            "correction_id": args.correction_id,
            "render_arguments": RENDER_ARGUMENTS,
        }
        result = update_frame_tree(
            args,
            request,
            frame_id,
            work,
            correction,
            metrics,
            reviews[frame_id],
        )
        results[frame_id] = result
        if result["visual_status"] == "pass" and status == "pass":
            accepted.append(result)
        print(f"published frame={frame_id}", flush=True)
    csv_rows = [{key: results[frame_id][key] for key in CSV_FIELDS} for frame_id in frames]
    atomic_csv(args.output_root / "metrics.csv", csv_rows)
    published_sheets = []
    for row in prepared["contact_sheets"]:
        source = Path(row["path"])
        destination = args.output_root / "contact_sheets" / f"{args.correction_id}_{source.name}"
        atomic_copy(source, destination)
        published_sheets.append({**row, "path": str(destination), "sha256": sha256(destination)})
    manifest_path = args.output_root / "campaign_manifest.json"
    manifest = load_json(manifest_path)
    manifest["initial_baseline"] = {
        "frame_ids": frames[:3],
        "metrics": [
            {
                "frame_id": frame_id,
                **{key: results[frame_id][key] for key in ("face_psnr", "face_ssim", "face_lpips")},
            }
            for frame_id in frames[:3]
        ],
    }
    manifest["regression_thresholds"] = thresholds
    for frame_id in frames:
        state = dict(manifest["frame_states"][frame_id])
        state.update(
            {
                "state": results[frame_id]["status"],
                "metric_status": results[frame_id]["metric_status"],
                "visual_status": results[frame_id]["visual_status"],
                "render_correction_id": args.correction_id,
                "updated_at": now(),
            }
        )
        manifest["frame_states"][frame_id] = state
    correction_manifest = {
        "schema_version": 1,
        "correction_id": args.correction_id,
        "status": "complete",
        "completed_at": now(),
        "reason": args.reason,
        "campaign_request_sha256": request["request_sha256"],
        "revision_request": revision_request,
        "frame_ids": frames,
        "frame_count": len(frames),
        "render_arguments": RENDER_ARGUMENTS,
        "geometry_modified": False,
        "mesh_hashes_unchanged": all(
            load_json(correction / "frame_receipts" / f"{frame_id}.json")["mesh_unchanged"]
            for frame_id in frames
        ),
        "eval_rgb_used_for_prediction": False,
        "masks_used": False,
        "source_colors_averaged": False,
        "contact_sheets": published_sheets,
        "frame_receipts": [
            load_json(correction / "frame_receipts" / f"{frame_id}.json") for frame_id in frames
        ],
        "metrics": [
            {key: results[frame_id][key] for key in ("frame_id", "face_psnr", "face_ssim", "face_lpips", "metric_status", "visual_status")}
            for frame_id in frames
        ],
        "regression_thresholds": thresholds,
    }
    atomic_json(correction / "correction_manifest.json", correction_manifest)
    manifest.setdefault("render_corrections", []).append(
        {
            "correction_id": args.correction_id,
            "manifest": str(correction / "correction_manifest.json"),
            "manifest_sha256": sha256(correction / "correction_manifest.json"),
            "frame_count": len(frames),
            "reason": args.reason,
        }
    )
    manifest["updated_at"] = now()
    atomic_json(manifest_path, manifest)
    atomic_json(
        correction / "state.json",
        {
            "schema_version": 1,
            "status": "complete",
            "completed_at": now(),
            "revision_request_sha256": revision_request["request_sha256"],
            "correction_manifest_sha256": sha256(correction / "correction_manifest.json"),
        },
    )
    os.replace(work, correction / "prepared_and_reviewed")
    print(f"complete status=published correction={args.correction_id} frames={len(frames)}", flush=True)


def parse_bool(value: str) -> bool:
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50"),
    )
    parser.add_argument("--correction-id", required=True)
    parser.add_argument("--frames", nargs="*")


def add_extension_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50"),
    )
    parser.add_argument("--correction-id", required=True)
    parser.add_argument("--frame-id", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser("prepare")
    add_common(prepare_parser)
    prepare_parser.add_argument("--gpu-index", default="0")
    prepare_parser.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="cuda")
    review_parser = actions.add_parser("review")
    add_common(review_parser)
    review_parser.add_argument("--visual-status", choices=("pass", "fail"), required=True)
    review_parser.add_argument("--ear-artifact", type=parse_bool, required=True)
    review_parser.add_argument("--lipstick-artifact", type=parse_bool, required=True)
    review_parser.add_argument("--visual-notes", required=True)
    publish_parser = actions.add_parser("publish")
    add_common(publish_parser)
    publish_parser.add_argument(
        "--reason",
        default=(
            "Uniform hard-source correction after a three-frame canary localized recurrent lipstick "
            "mosaic to small, photometrically discontinuous fallback visibility components."
        ),
    )
    extension_prepare_parser = actions.add_parser("extend-prepare")
    add_extension_common(extension_prepare_parser)
    extension_prepare_parser.add_argument("--gpu-index", default="0")
    extension_prepare_parser.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="cuda")
    extension_review_parser = actions.add_parser("extend-review")
    add_extension_common(extension_review_parser)
    extension_review_parser.add_argument("--visual-status", choices=("pass", "fail"), required=True)
    extension_review_parser.add_argument("--ear-artifact", type=parse_bool, required=True)
    extension_review_parser.add_argument("--lipstick-artifact", type=parse_bool, required=True)
    extension_review_parser.add_argument("--visual-notes", required=True)
    extension_publish_parser = actions.add_parser("extend-publish")
    add_extension_common(extension_publish_parser)
    extension_publish_parser.add_argument(
        "--reason",
        default=(
            "Apply the campaign-wide hard-source continuity correction to a newly reconstructed "
            "frame before its first publication."
        ),
    )
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    if not CORRECTION_ID_RE.fullmatch(args.correction_id):
        parser.error("--correction-id must be a short lowercase filesystem token")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reject_live_controller(args.output_root)
    with RevisionLock(args.output_root):
        if args.action == "prepare":
            prepare(args)
        elif args.action == "review":
            review(args)
        elif args.action == "publish":
            publish(args)
        elif args.action == "extend-prepare":
            prepare_extension(args)
        elif args.action == "extend-review":
            review_extension(args)
        else:
            publish_extension(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
