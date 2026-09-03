#!/usr/bin/env python3
"""Run fixed-pose COLMAP PatchMatch -> TSDF -> hard texture rendering.

The recipe is reusable across synchronized frames from one calibrated rig.  It
uses only train cameras for geometry, never consumes masks, never averages RGB
sources, and keeps completed expensive stages reusable.  Eval RGB is read only by
the final renderer after its prediction has already been constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VERIFIED_COLMAP_MARKERS = (
    "COLMAP 3.13.0.dev0",
    "Commit 5509fffe",
    "with CUDA",
)


def add_boolean_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str | None = None,
) -> None:
    """Backport ``BooleanOptionalAction`` for the project's Python 3.8 environment."""

    destination = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=destination, action="store_true", help=help)
    group.add_argument(f"--no-{name.lstrip('-')}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def parse_aabb(value: str) -> list[float]:
    try:
        result = [float(token) for token in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("AABB must be six comma-separated floats") from error
    if len(result) != 6 or not all(result[index + 3] > result[index] for index in range(3)):
        raise argparse.ArgumentTypeError("AABB must be min_x,min_y,min_z,max_x,max_y,max_z")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JPG Nerfstudio dataset with explicit splits.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--texture-data",
        type=Path,
        default=None,
        help=(
            "Optional prebuilt calibrated subset used only for hard texture lookup. "
            "By default an angular subset is built from --data."
        ),
    )
    parser.add_argument("--texture-camera-count", type=int, default=16)
    parser.add_argument("--colmap-bin", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--texture-neighbors", type=int, default=16)
    parser.add_argument("--patchmatch-iterations", type=int, default=3)
    parser.add_argument("--depth-min", type=float, default=4.5)
    parser.add_argument("--depth-max", type=float, default=20.0)
    parser.add_argument("--filter-min-ncc", type=float, default=0.1)
    parser.add_argument(
        "--geom-consistency-max-cost",
        type=float,
        default=6.0,
        help="Full-resolution geometric consistency gate; 6 px preserves the 3 px gate used at half resolution.",
    )
    parser.add_argument("--filter-min-triangulation-angle", type=float, default=1.0)
    parser.add_argument("--filter-min-num-consistent", type=int, default=2)
    parser.add_argument(
        "--filter-geom-consistency-max-cost",
        type=float,
        default=2.0,
        help="Full-resolution final consistency gate; 2 px preserves the 1 px half-resolution gate.",
    )
    parser.add_argument("--voxel-length", type=float, default=0.0005)
    parser.add_argument("--sdf-trunc", type=float, default=0.004)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument("--tsdf-backend", choices=("legacy", "tensor"), default="tensor")
    parser.add_argument("--tsdf-device", default="CUDA:0")
    parser.add_argument("--tensor-weight-threshold", type=float, default=2.0)
    parser.add_argument("--crop-aabb", type=parse_aabb, default=parse_aabb("-0.15,-0.15,-0.15,0.15,0.15,0.15"))
    parser.add_argument("--min-component-triangles", type=int, default=100)
    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=0.002,
        help="Remove disconnected TSDF islands below this fraction of the largest component.",
    )
    parser.add_argument("--depth-log-tolerance", type=float, default=0.01)
    parser.add_argument("--depth-hole-fill-max-area", type=int, default=1000)
    parser.add_argument("--target-depth-component-min-area", type=int, default=0)
    parser.add_argument("--target-depth-component-max-log-jump", type=float, default=0.0075)
    add_boolean_argument(parser, "--nearest-fill-color-continuity", default=False)
    parser.add_argument(
        "--nearest-fill-color-continuity-mode", choices=("pixel", "global"), default="pixel"
    )
    parser.add_argument("--nearest-fill-rank-penalty", type=float, default=0.0)
    parser.add_argument("--metric-surface-depth-manifest", type=Path, default=None)
    parser.add_argument("--roi-boxes-json", type=Path, default=None)
    add_boolean_argument(parser, "--score-metrics", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument(
        "--allow-unverified-colmap-build",
        action="store_true",
        help=(
            "Allow a CUDA COLMAP build other than the verified 3.13.0.dev0 commit 5509fffe. "
            "COLMAP 4.1.1 and one packaged 3.13 build produced invalid depth on this rig."
        ),
    )
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.texture_data = None if args.texture_data is None else args.texture_data.expanduser().resolve()
    if args.colmap_bin is None:
        resolved = shutil.which("colmap")
        if resolved is None:
            parser.error("COLMAP is not on PATH; pass --colmap-bin")
        args.colmap_bin = Path(resolved).resolve()
    else:
        args.colmap_bin = args.colmap_bin.expanduser().resolve()
    if not args.data.is_dir():
        parser.error("--data does not exist")
    if args.texture_data is not None and not args.texture_data.is_dir():
        parser.error("--texture-data does not exist")
    if not args.colmap_bin.is_file():
        parser.error(f"COLMAP binary does not exist: {args.colmap_bin}")
    positive = (
        args.image_size,
        args.source_count,
        args.texture_neighbors,
        args.texture_camera_count,
        args.patchmatch_iterations,
        args.depth_min,
        args.depth_max,
        args.filter_min_num_consistent,
        args.geom_consistency_max_cost,
        args.filter_geom_consistency_max_cost,
        args.voxel_length,
        args.sdf_trunc,
        args.depth_trunc,
        args.depth_log_tolerance,
        args.tensor_weight_threshold,
    )
    if any(float(value) <= 0 for value in positive) or args.depth_max <= args.depth_min:
        parser.error("Resolution/count/geometry parameters must be positive and depth_max > depth_min")
    if args.sdf_trunc < args.voxel_length:
        parser.error("--sdf-trunc must be at least one voxel")
    if (
        args.min_component_triangles < 0
        or args.depth_hole_fill_max_area < 0
        or args.target_depth_component_min_area < 0
    ):
        parser.error("component and hole-fill thresholds must be non-negative")
    if (
        not np.isfinite(args.target_depth_component_max_log_jump)
        or args.target_depth_component_max_log_jump <= 0.0
        or not np.isfinite(args.nearest_fill_rank_penalty)
        or args.nearest_fill_rank_penalty < 0.0
    ):
        parser.error("target depth jump must be positive and nearest-fill rank penalty non-negative")
    if not np.isfinite(args.min_component_fraction) or not 0.0 <= args.min_component_fraction <= 1.0:
        parser.error("--min-component-fraction must be finite and between zero and one")
    if args.score_metrics:
        if args.metric_surface_depth_manifest is None or args.roi_boxes_json is None:
            parser.error("--score-metrics requires --metric-surface-depth-manifest and --roi-boxes-json")
    if args.metric_surface_depth_manifest is not None:
        args.metric_surface_depth_manifest = args.metric_surface_depth_manifest.expanduser().resolve()
    if args.roi_boxes_json is not None:
        args.roi_boxes_json = args.roi_boxes_json.expanduser().resolve()
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_colmap_build(probe: str, *, allow_unverified: bool) -> str:
    """Fail closed on unvalidated PatchMatch binaries.

    Dense depth changed materially across the tested COLMAP binaries even with
    identical cameras and options, so a generic CUDA check is not sufficient.
    """

    if "with CUDA" not in probe:
        raise RuntimeError("COLMAP PatchMatch requires a CUDA build; probe did not report 'with CUDA'")
    missing = [marker for marker in VERIFIED_COLMAP_MARKERS if marker not in probe]
    if missing and not allow_unverified:
        raise RuntimeError(
            "Unverified COLMAP PatchMatch build. Expected "
            f"{', '.join(VERIFIED_COLMAP_MARKERS)}; pass --allow-unverified-colmap-build only after a depth-map canary."
        )
    return "\n".join(line.strip() for line in probe.splitlines()[:2])


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train_count(data: Path) -> int:
    payload = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    train = payload.get("train_filenames")
    if not isinstance(frames, list) or not isinstance(train, list) or not train:
        raise ValueError("Pipeline requires frames and an explicit non-empty train_filenames split")
    if any(isinstance(frame, dict) and "mask_path" in frame for frame in frames):
        raise ValueError("COLMAP PatchMatch-TSDF pipeline forbids image/person masks")
    return len(train)


def runtime_env(colmap_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    prefix_lib = colmap_bin.parent.parent / "lib"
    if prefix_lib.is_dir():
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = str(prefix_lib) + (f":{current}" if current else "")
    return env


def run_stage(
    name: str,
    command: list[str],
    *,
    log: Path,
    env: dict[str, str],
    complete: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, object]:
    printable = " ".join(subprocess.list2cmdline([token]) for token in command)
    if complete and resume:
        print(f"stage={name} status=skipped", flush=True)
        return {"name": name, "status": "skipped", "command": command, "log": str(log)}
    if dry_run:
        print(f"stage={name} status=dry-run command={printable}", flush=True)
        return {"name": name, "status": "dry-run", "command": command, "log": str(log)}
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"stage={name} status=running", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    elapsed = time.monotonic() - started
    print(f"stage={name} status=complete seconds={elapsed:.1f}", flush=True)
    return {
        "name": name,
        "status": "complete",
        "seconds": elapsed,
        "command": command,
        "log": str(log),
    }


def validate_normalization(mesh_receipt: Path, depth_manifest: Path) -> None:
    mesh = json.loads(mesh_receipt.read_text(encoding="utf-8"))
    depth = json.loads(depth_manifest.read_text(encoding="utf-8"))
    if not np.isclose(float(mesh["dataparser_scale"]), float(depth["dataparser_scale"]), rtol=1e-7, atol=1e-9):
        raise ValueError("Fusion and texture datasets resolve different dataparser scales")
    if not np.allclose(
        np.asarray(mesh["dataparser_transform"]),
        np.asarray(depth["dataparser_transform"]),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError("Fusion and texture datasets resolve different dataparser transforms")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_existed = args.output_dir.exists()
    if output_existed and not args.resume and not args.dry_run:
        raise FileExistsError(f"Output directory exists; pass --resume: {args.output_dir}")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    count = train_count(args.data)
    if count < 2:
        raise ValueError("PatchMatch requires at least two train cameras")
    source_count = min(args.source_count, count - 1)
    texture_data = args.texture_data or (args.output_dir / "texture_subset")
    texture_count = train_count(texture_data) if args.texture_data is not None else min(args.texture_camera_count, count)
    texture_neighbors = min(args.texture_neighbors, texture_count)
    python = sys.executable
    colmap = str(args.colmap_bin)
    env = runtime_env(args.colmap_bin)
    colmap_build = "not-probed-dry-run"
    if not args.dry_run:
        probe = subprocess.run([colmap, "-h"], check=True, text=True, capture_output=True, env=env)
        colmap_build = validate_colmap_build(
            probe.stdout + probe.stderr,
            allow_unverified=args.allow_unverified_colmap_build,
        )

    model = args.output_dir / "fixed_model"
    dense = args.output_dir / "dense"
    depth_data = args.output_dir / "depth_dataset"
    mesh = args.output_dir / "colmap_patchmatch_tsdf.ply"
    mesh_depth = args.output_dir / "mesh_depth"
    render = args.output_dir / "render"
    logs = args.output_dir / "logs"
    commands: list[tuple[str, list[str], Path, bool]] = [
        *(
            []
            if args.texture_data is not None
            else [
                (
                    "texture-subset",
                    [python, str(SCRIPT_DIR / "build_angular_camera_subset.py"), "--input", str(args.data), "--output", str(texture_data), "--train-count", str(texture_count), "--strategy", "angular"],
                    logs / "texture_subset.log",
                    (texture_data / "transforms.json").is_file(),
                )
            ]
        ),
        (
            "export-fixed-model",
            [python, str(SCRIPT_DIR / "export_nerfstudio_colmap_model.py"), "--data", str(args.data), "--output-model", str(model), "--split", "train"],
            logs / "export_fixed_model.log",
            (model / "export_manifest.json").is_file(),
        ),
        (
            "undistort",
            [colmap, "image_undistorter", "--image_path", str(args.data), "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", str(args.image_size), "--copy_policy", "soft-link"],
            logs / "image_undistorter.log",
            (dense / "sparse" / "images.bin").is_file(),
        ),
        (
            "patch-config",
            [python, str(SCRIPT_DIR / "build_colmap_patch_match_config.py"), "--data", str(args.data), "--output", str(dense / "stereo" / "patch-match.cfg"), "--source-count", str(source_count), "--split", "train"],
            logs / "patch_config.log",
            (dense / "stereo" / "patch-match.cfg").is_file(),
        ),
        (
            "patchmatch-photometric",
            [colmap, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", str(args.image_size), "--PatchMatchStereo.gpu_index", str(args.gpu_index), "--PatchMatchStereo.depth_min", str(args.depth_min), "--PatchMatchStereo.depth_max", str(args.depth_max), "--PatchMatchStereo.num_iterations", str(args.patchmatch_iterations), "--PatchMatchStereo.geom_consistency", "0", "--PatchMatchStereo.filter", "0", "--PatchMatchStereo.write_consistency_graph", "1"],
            logs / "patch_match_photometric.log",
            len(list((dense / "stereo" / "depth_maps").glob("**/*.photometric.bin"))) == count,
        ),
        (
            "patchmatch-geometric",
            [colmap, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", str(args.image_size), "--PatchMatchStereo.gpu_index", str(args.gpu_index), "--PatchMatchStereo.depth_min", str(args.depth_min), "--PatchMatchStereo.depth_max", str(args.depth_max), "--PatchMatchStereo.num_iterations", str(args.patchmatch_iterations), "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.geom_consistency_max_cost", str(args.geom_consistency_max_cost), "--PatchMatchStereo.filter", "1", "--PatchMatchStereo.filter_min_ncc", str(args.filter_min_ncc), "--PatchMatchStereo.filter_min_triangulation_angle", str(args.filter_min_triangulation_angle), "--PatchMatchStereo.filter_min_num_consistent", str(args.filter_min_num_consistent), "--PatchMatchStereo.filter_geom_consistency_max_cost", str(args.filter_geom_consistency_max_cost), "--PatchMatchStereo.write_consistency_graph", "1"],
            logs / "patch_match_geometric.log",
            len(list((dense / "stereo" / "depth_maps").glob("**/*.geometric.bin"))) == count,
        ),
        (
            "import-depth",
            [python, str(SCRIPT_DIR / "import_colmap_mvs_depth_dataset.py"), "--data", str(args.data), "--depth-maps", str(dense / "stereo" / "depth_maps"), "--output", str(depth_data), "--input-type", "geometric", "--colmap-model", str(dense / "sparse"), "--undistorted-images", str(dense / "images")],
            logs / "import_depth.log",
            (depth_data / "transforms.json").is_file(),
        ),
        (
            "fuse-tsdf",
            [python, str(SCRIPT_DIR / "fuse_depth_tsdf_mesh.py"), "--data", str(depth_data), "--output", str(mesh), "--eval-mode", "filename", "--orientation-method", "up", "--center-method", "focus", "--auto-scale-poses", "--scale-factor", "1", "--scene-scale", "2", "--downscale-factor", "1", "--depth-unit-scale-factor", "1", "--voxel-length", str(args.voxel_length), "--sdf-trunc", str(args.sdf_trunc), "--depth-trunc", str(args.depth_trunc), "--backend", args.tsdf_backend, "--device", args.tsdf_device, "--tensor-weight-threshold", str(args.tensor_weight_threshold), "--crop-aabb", *[str(value) for value in args.crop_aabb], "--min-component-triangles", str(args.min_component_triangles), "--min-component-fraction", str(args.min_component_fraction)],
            logs / "fuse_tsdf.log",
            mesh.is_file() and mesh.with_suffix(".json").is_file(),
        ),
        (
            "raycast-mesh",
            [python, str(SCRIPT_DIR / "render_tsdf_mesh_depth.py"), "--data", str(texture_data), "--mesh", str(mesh), "--output-dir", str(mesh_depth), "--eval-mode", "filename", "--orientation-method", "up", "--center-method", "focus", "--auto-scale-poses", "--scale-factor", "1", "--scene-scale", "2", "--downscale-factor", "1"],
            logs / "raycast_mesh.log",
            (mesh_depth / "mesh_depth_manifest.json").is_file(),
        ),
    ]
    render_command = [
        python,
        str(SCRIPT_DIR / "render_mesh_image_blend.py"),
        "--data", str(texture_data),
        "--mesh-depth-manifest", str(mesh_depth / "mesh_depth_manifest.json"),
        "--output-dir", str(render),
        "--neighbors", str(texture_neighbors),
        "--aggregation-modes", "nearest-fill",
        "--blend-alphas", "1",
        "--depth-log-tolerance", str(args.depth_log_tolerance),
        "--depth-hole-fill-max-area", str(args.depth_hole_fill_max_area),
        "--depth-hole-fill-boundary-radius", "4",
        "--depth-hole-fill-max-relative-plane-rmse", "0.015",
        "--target-depth-component-min-area", str(args.target_depth_component_min_area),
        "--target-depth-component-max-log-jump", str(args.target_depth_component_max_log_jump),
        "--nearest-fill-color-continuity-mode", args.nearest_fill_color_continuity_mode,
        "--nearest-fill-rank-penalty", str(args.nearest_fill_rank_penalty),
        "--eval-mode", "filename",
        "--orientation-method", "up",
        "--center-method", "focus",
        "--auto-scale-poses",
        "--scale-factor", "1",
        "--scene-scale", "2",
        "--downscale-factor", "1",
        "--device", "cuda",
    ]
    if args.nearest_fill_color_continuity:
        render_command.append("--nearest-fill-color-continuity")
    if args.score_metrics:
        render_command += [
            "--metric-surface-depth-manifest", str(args.metric_surface_depth_manifest),
            "--score-metrics",
            "--metric-regions", "surface-roi",
            "--roi-boxes-json", str(args.roi_boxes_json),
        ]
    commands.append(
        (
            "hard-texture-render",
            render_command,
            logs / "hard_texture_render.log",
            (render / f"nearest_fill{texture_neighbors}" / "eval_pred_0000.png").is_file(),
        )
    )

    request = {
        "schema_version": 1,
        "data": str(args.data),
        "texture_data": str(texture_data),
        "texture_subset_strategy": None if args.texture_data is not None else "angular",
        "texture_camera_count": texture_count,
        "source_transforms_sha256": sha256(args.data / "transforms.json"),
        "texture_source_transforms_sha256": sha256(
            (args.texture_data or args.data) / "transforms.json"
        ),
        "colmap_binary": colmap,
        "colmap_build": colmap_build,
        "commands": [{"stage": name, "command": command} for name, command, _, _ in commands],
    }
    request_path = args.output_dir / "pipeline_request.json"
    if not args.dry_run:
        if output_existed:
            if not request_path.is_file():
                raise RuntimeError(f"Cannot safely resume without {request_path}")
            previous_request = json.loads(request_path.read_text(encoding="utf-8"))
            if previous_request != request:
                raise RuntimeError("Refusing to resume: inputs, COLMAP build, or stage parameters changed")
        else:
            write_json(request_path, request)

    stages: list[dict[str, object]] = []
    for name, command, log, complete in commands:
        stages.append(
            run_stage(
                name,
                command,
                log=log,
                env=env,
                complete=complete,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        )
        if name == "raycast-mesh" and not args.dry_run:
            validate_normalization(mesh.with_suffix(".json"), mesh_depth / "mesh_depth_manifest.json")
    if args.dry_run:
        return 0
    final_render = render / f"nearest_fill{texture_neighbors}" / "eval_pred_0000.png"
    receipt = {
        "schema_version": 1,
        "method": "fixed_pose_colmap_patchmatch_tsdf_hard_nearest_fill",
        "data": str(args.data),
        "texture_data": str(texture_data),
        "source_transforms_sha256": sha256(args.data / "transforms.json"),
        "uses_eval_images_for_geometry": False,
        "uses_masks": False,
        "rgb_aggregation": (
            "hard_nearest_fill_global_color_order_no_average"
            if args.nearest_fill_color_continuity
            and args.nearest_fill_color_continuity_mode == "global"
            else "hard_nearest_fill_no_average"
        ),
        "target_depth_component_filter": {
            "min_area": args.target_depth_component_min_area,
            "max_log_jump": args.target_depth_component_max_log_jump,
        },
        "nearest_fill_color_continuity": args.nearest_fill_color_continuity,
        "nearest_fill_color_continuity_mode": args.nearest_fill_color_continuity_mode,
        "nearest_fill_rank_penalty": args.nearest_fill_rank_penalty,
        "colmap": {
            "binary": colmap,
            "build": colmap_build,
            "image_size": args.image_size,
            "source_count": source_count,
            "iterations": args.patchmatch_iterations,
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "filter_min_ncc": args.filter_min_ncc,
            "geom_consistency_max_cost": args.geom_consistency_max_cost,
            "filter_min_triangulation_angle": args.filter_min_triangulation_angle,
            "filter_min_num_consistent": args.filter_min_num_consistent,
            "filter_geom_consistency_max_cost": args.filter_geom_consistency_max_cost,
        },
        "tsdf": {
            "voxel_length": args.voxel_length,
            "sdf_trunc": args.sdf_trunc,
            "depth_trunc": args.depth_trunc,
            "backend": args.tsdf_backend,
            "device": args.tsdf_device if args.tsdf_backend == "tensor" else "CPU",
            "tensor_weight_threshold": args.tensor_weight_threshold if args.tsdf_backend == "tensor" else None,
            "crop_aabb": args.crop_aabb,
            "min_component_triangles": args.min_component_triangles,
            "min_component_fraction": args.min_component_fraction,
        },
        "texture_neighbors": texture_neighbors,
        "stages": stages,
        "mesh": str(mesh),
        "mesh_sha256": sha256(mesh),
        "mesh_metadata": str(mesh.with_suffix(".json")),
        "mesh_metadata_sha256": sha256(mesh.with_suffix(".json")),
        "depth_metadata": json.loads((depth_data / "transforms.json").read_text(encoding="utf-8"))[
            "colmap_mvs_depth"
        ],
        "final_render": str(final_render),
        "final_render_sha256": sha256(final_render),
    }
    write_json(args.output_dir / "pipeline_manifest.json", receipt)
    print(f"complete render={final_render}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
