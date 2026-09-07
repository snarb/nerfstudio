#!/usr/bin/env python3
"""Run the frozen PatchMatch -> TSDF geometry stages without a discarded eval render."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256, validate_hash_manifest
from run_colmap_patchmatch_tsdf import runtime_env, train_count, validate_colmap_build


SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--colmap-bin", type=Path, required=True)
    parser.add_argument("--gpu-index", default="0")
    args = parser.parse_args(argv)
    for name in ("data", "workspace", "colmap_bin"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if len(args.frame_id) != 6 or not args.frame_id.isdigit():
        parser.error("--frame-id must be six digits")
    if not args.data.is_dir() or not args.colmap_bin.is_file():
        parser.error("--data and --colmap-bin must exist")
    return args


def run_stage(
    name: str, command: list[str], log: Path, env: dict[str, str], complete: bool, resume: bool,
) -> dict[str, Any]:
    if complete:
        if not resume:
            raise RuntimeError(f"Stage output exists without resume: {name}")
        return {"stage": name, "status": "reused", "seconds": 0.0}
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    return {"stage": name, "status": "complete", "seconds": time.monotonic() - start}


def commands(data: Path, output: Path, colmap: Path, gpu_index: str) -> list[tuple[str, list[str], Path, bool]]:
    model = output / "fixed_model"
    dense = output / "dense"
    depth_data = output / "depth_dataset"
    mesh = output / "colmap_patchmatch_tsdf.ply"
    logs = output / "logs"
    python = sys.executable
    return [
        (
            "export-fixed-model",
            [python, str(SCRIPT_DIR / "export_nerfstudio_colmap_model.py"), "--data", str(data), "--output-model", str(model), "--split", "train"],
            logs / "export_fixed_model.log", (model / "export_manifest.json").is_file(),
        ),
        (
            "undistort",
            [str(colmap), "image_undistorter", "--image_path", str(data), "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1920", "--copy_policy", "soft-link"],
            logs / "image_undistorter.log", (dense / "sparse/images.bin").is_file(),
        ),
        (
            "patch-config",
            [python, str(SCRIPT_DIR / "build_colmap_patch_match_config.py"), "--data", str(data), "--output", str(dense / "stereo/patch-match.cfg"), "--source-count", "12", "--split", "train"],
            logs / "patch_config.log", (dense / "stereo/patch-match.cfg").is_file(),
        ),
        (
            "patchmatch-photometric",
            [str(colmap), "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "1920", "--PatchMatchStereo.gpu_index", gpu_index, "--PatchMatchStereo.depth_min", "4.5", "--PatchMatchStereo.depth_max", "20.0", "--PatchMatchStereo.num_iterations", "3", "--PatchMatchStereo.geom_consistency", "0", "--PatchMatchStereo.filter", "0", "--PatchMatchStereo.write_consistency_graph", "1"],
            logs / "patch_match_photometric.log",
            len(list((dense / "stereo/depth_maps").glob("**/*.photometric.bin"))) == 62,
        ),
        (
            "patchmatch-geometric",
            [str(colmap), "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "1920", "--PatchMatchStereo.gpu_index", gpu_index, "--PatchMatchStereo.depth_min", "4.5", "--PatchMatchStereo.depth_max", "20.0", "--PatchMatchStereo.num_iterations", "3", "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.geom_consistency_max_cost", "6.0", "--PatchMatchStereo.filter", "1", "--PatchMatchStereo.filter_min_ncc", "0.1", "--PatchMatchStereo.filter_min_triangulation_angle", "1.0", "--PatchMatchStereo.filter_min_num_consistent", "2", "--PatchMatchStereo.filter_geom_consistency_max_cost", "2.0", "--PatchMatchStereo.write_consistency_graph", "1"],
            logs / "patch_match_geometric.log",
            len(list((dense / "stereo/depth_maps").glob("**/*.geometric.bin"))) == 62,
        ),
        (
            "import-depth",
            [python, str(SCRIPT_DIR / "import_colmap_mvs_depth_dataset.py"), "--data", str(data), "--depth-maps", str(dense / "stereo/depth_maps"), "--output", str(depth_data), "--input-type", "geometric", "--colmap-model", str(dense / "sparse"), "--undistorted-images", str(dense / "images")],
            logs / "import_depth.log", (depth_data / "transforms.json").is_file(),
        ),
        (
            "fuse-tsdf",
            [python, str(SCRIPT_DIR / "fuse_depth_tsdf_mesh.py"), "--data", str(depth_data), "--output", str(mesh), "--eval-mode", "filename", "--orientation-method", "up", "--center-method", "focus", "--auto-scale-poses", "--scale-factor", "1", "--scene-scale", "2", "--downscale-factor", "1", "--depth-unit-scale-factor", "1", "--voxel-length", "0.0005", "--sdf-trunc", "0.004", "--depth-trunc", "4.0", "--backend", "tensor", "--device", "CUDA:0", "--tensor-weight-threshold", "2.0", "--crop-aabb", "-0.15", "-0.15", "-0.15", "0.15", "0.15", "0.15", "--min-component-triangles", "100", "--min-component-fraction", "0.002"],
            logs / "fuse_tsdf.log", mesh.is_file() and mesh.with_suffix(".json").is_file(),
        ),
    ]


def validate_geometry(frame_id: str, data: Path, output: Path, colmap_build: str) -> dict[str, Any]:
    split = load_json(data / "transforms.json")
    if len(split.get("train_filenames", [])) != 62 or len(split.get("val_filenames", [])) != 1:
        raise ValueError("Geometry worker input must retain the explicit 62/1 split")
    geometric = sorted((output / "dense/stereo/depth_maps").glob("**/*.geometric.bin"))
    if len(geometric) != 62:
        raise ValueError(f"Expected 62 geometric maps, got {len(geometric)}")
    depth = load_json(output / "depth_dataset/transforms.json")["colmap_mvs_depth"]
    rows = depth.get("depth_maps", [])
    coverages = [float(row["coverage"]) for row in rows]
    if (
        depth.get("train_depth_count") != 62
        or len(rows) != 62
        or {tuple(row.get("shape", [])) for row in rows} != {(1080, 1920)}
        or not all(math.isfinite(value) and value > 0 for value in coverages)
        or not math.isclose(float(depth["coverage_mean"]), float(np.mean(coverages)), abs_tol=1e-12)
        or not math.isclose(float(depth["coverage_min"]), min(coverages), abs_tol=1e-12)
    ):
        raise ValueError("Full-resolution geometric depth validation failed")
    mesh = output / "colmap_patchmatch_tsdf.ply"
    metadata = load_json(mesh.with_suffix(".json"))
    result = {
        "schema_version": 1, "frame_id": frame_id,
        "geometry_only": True, "heldout_rgb_read": False,
        "colmap_build": colmap_build,
        "depth_map_count": 62, "depth_shape": [1080, 1920],
        "depth_coverage_mean": float(depth["coverage_mean"]),
        "depth_coverage_min": float(depth["coverage_min"]),
        "mesh_vertices": int(metadata.get("vertices", 0)),
        "mesh_triangles": int(metadata.get("triangles", 0)),
        "mesh_components": int(metadata.get("connected_components", 0)),
        "mesh_component_triangles": metadata.get("component_triangles"),
        "mesh_sha256": sha256(mesh), "mesh_metadata_sha256": sha256(mesh.with_suffix(".json")),
        "validation_status": "pass",
    }
    if (
        result["mesh_vertices"] <= 0 or result["mesh_triangles"] <= 0
        or result["mesh_components"] != 1
        or metadata.get("output_sha256") != result["mesh_sha256"]
    ):
        raise ValueError("TSDF mesh validation failed")
    return result


def publish(data: Path, output: Path, workspace: Path, result: dict[str, Any]) -> Path:
    retained = workspace / "retained"
    if retained.is_dir():
        validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
        return retained
    stage = workspace / f".retained.tmp-{os.getpid()}"
    stage.mkdir(parents=True)
    mappings = {
        output / "colmap_patchmatch_tsdf.ply": stage / "mesh/colmap_patchmatch_tsdf.ply",
        output / "colmap_patchmatch_tsdf.json": stage / "mesh/colmap_patchmatch_tsdf.json",
        output / "pipeline_request.json": stage / "pipeline_request.json",
        output / "pipeline_manifest.json": stage / "pipeline_manifest.json",
        output / "depth_dataset/transforms.json": stage / "depth_qc.json",
        data / "transforms.json": stage / "staged_transforms.json",
        data / "staging_manifest.json": stage / "staging_manifest.json",
    }
    for source, destination in mappings.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for log in sorted((output / "logs").glob("*.log")):
        destination = stage / "logs" / f"{log.stem}.tail.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - (256 << 10)))
            destination.write_bytes(stream.read())
    atomic_json(stage / "remote_result.json", result)
    files = [
        {"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(stage.rglob("*")) if path.is_file()
    ]
    atomic_json(stage / "retained_manifest.json", {"schema_version": 1, "files": files})
    validate_hash_manifest(stage, load_json(stage / "retained_manifest.json"))
    os.replace(stage, retained)
    return retained


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.workspace.mkdir(parents=True, exist_ok=True)
    retained = args.workspace / "retained"
    if retained.is_dir():
        validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
        print(f"frame={args.frame_id} status=reused retained={retained}", flush=True)
        return 0
    output = args.workspace / "pipeline"
    resume = output.exists()
    output.mkdir(parents=True, exist_ok=True)
    if train_count(args.data) != 62:
        raise ValueError("Frozen geometry recipe requires exactly 62 train cameras")
    environment = runtime_env(args.colmap_bin)
    probe = subprocess.run([str(args.colmap_bin), "-h"], check=True, text=True, capture_output=True, env=environment)
    colmap_build = validate_colmap_build(probe.stdout + probe.stderr, allow_unverified=False)
    stage_commands = commands(args.data, output, args.colmap_bin, args.gpu_index)
    request = {
        "schema_version": 1, "method": "fixed_pose_colmap_patchmatch_tsdf_geometry_only",
        "data": str(args.data), "source_transforms_sha256": sha256(args.data / "transforms.json"),
        "uses_eval_images_for_geometry": False, "uses_masks": False,
        "colmap_build": colmap_build,
        "commands": [{"stage": name, "command": command} for name, command, _, _ in stage_commands],
    }
    request_path = output / "pipeline_request.json"
    if request_path.is_file() and load_json(request_path) != request:
        raise RuntimeError("Refusing to resume changed geometry-only inputs/commands")
    atomic_json(request_path, request)
    stages = [run_stage(name, command, log, environment, complete, resume) for name, command, log, complete in stage_commands]
    result = validate_geometry(args.frame_id, args.data, output, colmap_build)
    atomic_json(output / "pipeline_manifest.json", {
        **request, "stages": stages, "mesh_sha256": result["mesh_sha256"],
        "mesh_metadata_sha256": result["mesh_metadata_sha256"],
        "depth_coverage_mean": result["depth_coverage_mean"],
        "depth_coverage_min": result["depth_coverage_min"],
    })
    retained = publish(args.data, output, args.workspace, result)
    print(f"frame={args.frame_id} status=complete retained={retained}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
