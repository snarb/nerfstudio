#!/usr/bin/env python3
"""Run and validate one frozen PatchMatch-TSDF frame on the pinned GPU host."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256, validate_hash_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--colmap-bin", type=Path, default=Path("/usr/local/bin/colmap"))
    parser.add_argument("--gpu-index", default="0")
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.workspace = args.workspace.expanduser().resolve()
    args.colmap_bin = args.colmap_bin.expanduser().resolve()
    if not (len(args.frame_id) == 6 and args.frame_id.isdigit()):
        parser.error("--frame-id must be six digits")
    if not args.data.is_dir() or not args.colmap_bin.is_file():
        parser.error("--data and --colmap-bin must exist")
    return args


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate_rgb(path: Path, expected_shape: tuple[int, int]) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or tuple(image.shape[:2]) != expected_shape:
        raise ValueError(f"Invalid {expected_shape[1]}x{expected_shape[0]} RGB image: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite RGB image: {path}")


def compact_log(source: Path, destination: Path, limit: int = 256 << 10) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        payload = stream.read()
    destination.write_bytes(payload)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_pipeline(frame_id: str, data: Path, output: Path) -> dict:
    transforms = load_json(data / "transforms.json")
    if len(transforms.get("train_filenames", [])) != 62 or len(transforms.get("val_filenames", [])) != 1:
        raise ValueError("Remote staged dataset must have an explicit 62/1 split")
    geometric = sorted((output / "dense" / "stereo" / "depth_maps").glob("**/*.geometric.bin"))
    if len(geometric) != 62:
        raise ValueError(f"Expected 62 geometric depth maps, got {len(geometric)}")
    depth_payload = load_json(output / "depth_dataset" / "transforms.json")["colmap_mvs_depth"]
    rows = depth_payload.get("depth_maps")
    if depth_payload.get("train_depth_count") != 62 or not isinstance(rows, list) or len(rows) != 62:
        raise ValueError("Imported depth metadata must contain 62 train rows")
    shapes = {tuple(row.get("shape", [])) for row in rows}
    coverages = [float(row["coverage"]) for row in rows]
    if shapes != {(1080, 1920)}:
        raise ValueError(f"Expected only 1080x1920 depth maps, got {sorted(shapes)}")
    if not all(math.isfinite(value) and value > 0.0 for value in coverages):
        raise ValueError("Every geometric depth map must have finite positive coverage")
    if not math.isclose(float(depth_payload["coverage_mean"]), float(np.mean(coverages)), abs_tol=1e-12):
        raise ValueError("Depth coverage mean disagrees with per-camera rows")
    if not math.isclose(float(depth_payload["coverage_min"]), min(coverages), abs_tol=1e-12):
        raise ValueError("Depth coverage minimum disagrees with per-camera rows")

    mesh = output / "colmap_patchmatch_tsdf.ply"
    mesh_meta = load_json(mesh.with_suffix(".json"))
    vertices = int(mesh_meta.get("vertices", 0))
    triangles = int(mesh_meta.get("triangles", 0))
    components = int(mesh_meta.get("connected_components", 0))
    if vertices <= 0 or triangles <= 0 or components <= 0 or sha256(mesh) != mesh_meta.get("output_sha256"):
        raise ValueError("Mesh is empty or its metadata/hash is invalid")

    render_root = output / "render"
    variant = render_root / "nearest_fill16"
    prediction_exr = variant / "eval_pred_0000.exr"
    prediction_png = variant / "eval_pred_0000.png"
    ground_truth = render_root / "eval_gt_0000.exr"
    validate_rgb(prediction_exr, (1080, 1920))
    validate_rgb(prediction_png, (1080, 1920))
    validate_rgb(ground_truth, (1080, 1920))
    pipeline = load_json(output / "pipeline_manifest.json")
    if pipeline.get("uses_eval_images_for_geometry") is not False or pipeline.get("uses_masks") is not False:
        raise ValueError("Pipeline receipt does not prove held-out/mask exclusion")
    if "5509fffe" not in str(pipeline.get("colmap", {}).get("build")):
        raise ValueError("Pipeline receipt does not identify the verified COLMAP commit")
    angular = load_json(output / "texture_subset" / "angular_subset_manifest.json")
    if angular.get("selected_train_count") != 16 or angular.get("selection_uses_image_pixels") is not False:
        raise ValueError("Texture subset must contain 16 calibration-only angular cameras")
    reprojection = load_json(render_root / "reprojection_audit.json")
    if reprojection.get("metrics") not in ({}, None):
        raise ValueError("Remote renderer must not compute legacy/full-frame metrics")
    return {
        "schema_version": 1,
        "frame_id": frame_id,
        "depth_map_count": len(geometric),
        "depth_shape": [1080, 1920],
        "depth_coverage_mean": float(depth_payload["coverage_mean"]),
        "depth_coverage_min": float(depth_payload["coverage_min"]),
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "mesh_components": components,
        "mesh_component_triangles": mesh_meta["component_triangles"],
        "mesh_sha256": sha256(mesh),
        "render_sha256": sha256(prediction_png),
        "render_exr_sha256": sha256(prediction_exr),
        "ground_truth_sha256": sha256(ground_truth),
        "texture_camera_count": int(angular["selected_train_count"]),
        "validation_status": "pass",
    }


def publish_retained(data: Path, output: Path, workspace: Path, result: dict) -> Path:
    retained = workspace / "retained"
    if retained.exists():
        manifest = load_json(retained / "retained_manifest.json")
        validate_hash_manifest(retained, manifest)
        return retained
    stage = workspace / f".retained.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    mappings = {
        output / "colmap_patchmatch_tsdf.ply": stage / "mesh" / "colmap_patchmatch_tsdf.ply",
        output / "colmap_patchmatch_tsdf.json": stage / "mesh" / "colmap_patchmatch_tsdf.json",
        output / "pipeline_request.json": stage / "pipeline_request.json",
        output / "pipeline_manifest.json": stage / "pipeline_manifest.json",
        output / "depth_dataset" / "transforms.json": stage / "depth_qc.json",
        output / "texture_subset" / "angular_subset_manifest.json": stage / "texture" / "angular_subset_manifest.json",
        output / "texture_subset" / "transforms.json": stage / "texture" / "transforms.json",
        output / "render" / "eval_gt_0000.exr": stage / "render" / "eval_gt_0000.exr",
        output / "render" / "nearest_fill16" / "eval_pred_0000.exr": stage / "render" / "eval_pred_0000.exr",
        output / "render" / "nearest_fill16" / "eval_pred_0000.png": stage / "render" / "eval_pred_0000.png",
        output / "render" / "nearest_fill16" / "source_selection.png": stage / "render" / "source_selection.png",
        output / "render" / "reprojection_audit.json": stage / "render" / "reprojection_audit.json",
        data / "transforms.json": stage / "staged_transforms.json",
        data / "staging_manifest.json": stage / "staging_manifest.json",
    }
    for source, destination in mappings.items():
        copy_file(source, destination)
    for log in sorted((output / "logs").glob("*.log")):
        compact_log(log, stage / "logs" / f"{log.stem}.tail.log")
    staging = load_json(data / "staging_manifest.json")
    target_rows = [
        row for row in staging["conversion_rows"]
        if row.get("physical_camera") == "F004_B005_1210O9"
    ]
    if len(target_rows) != 1:
        raise ValueError("Staging audit has no unique held-out conversion row")
    gt_audit = {
        "schema_version": 1,
        "heldout_used_for_prediction": False,
        "physical_camera": "F004_B005_1210O9",
        "source_exr": target_rows[0]["input"],
        "source_exr_sha256": target_rows[0]["source_sha256"],
        "temporary_jpeg_sha256": target_rows[0]["sha256"],
        "retained_display_ground_truth": "render/eval_gt_0000.exr",
        "retained_display_ground_truth_sha256": result["ground_truth_sha256"],
    }
    atomic_json(stage / "render" / "eval_ground_truth.json", gt_audit)
    atomic_json(stage / "remote_result.json", result)
    files = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(stage / "retained_manifest.json", {"schema_version": 1, "files": files})
    os.replace(stage, retained)
    return retained


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace.mkdir(parents=True, exist_ok=True)
    retained = args.workspace / "retained"
    if retained.is_dir() and (retained / "retained_manifest.json").is_file():
        validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
        print(f"complete status=reused frame={args.frame_id} retained={retained}", flush=True)
        return 0
    output = args.workspace / "pipeline"
    command = [
        sys.executable, str(Path(__file__).resolve().parent / "run_colmap_patchmatch_tsdf.py"),
        "--data", str(args.data), "--output-dir", str(output),
        "--colmap-bin", str(args.colmap_bin), "--gpu-index", args.gpu_index,
    ]
    if output.exists():
        command.append("--resume")
    print(f"frame={args.frame_id} status=running", flush=True)
    subprocess.run(command, check=True)
    result = validate_pipeline(args.frame_id, args.data, output)
    retained = publish_retained(args.data, output, args.workspace, result)
    validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
    print(f"frame={args.frame_id} status=complete retained={retained}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
