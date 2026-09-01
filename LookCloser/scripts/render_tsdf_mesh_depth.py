#!/usr/bin/env python3
"""Render first-hit camera-z depth and optional vertex colour from a TSDF mesh."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig


def nerfstudio_c2w_to_opencv_extrinsic(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert Nerfstudio/OpenGL c2w into OpenCV world-to-camera coordinates."""

    source = np.asarray(camera_to_world, dtype=np.float64)
    if source.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3] = source
    elif source.shape == (4, 4):
        homogeneous = source.copy()
    else:
        raise ValueError(f"Expected camera_to_world shape (3,4) or (4,4), got {source.shape}")
    if not np.isfinite(homogeneous).all():
        raise ValueError("camera_to_world contains non-finite values")
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(homogeneous @ opengl_to_opencv)


def sha256(path: Path) -> str:
    """Return a stable content hash without depending on another diagnostic script."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def barycentric_vertex_colors(
    *,
    primitive_ids: np.ndarray,
    primitive_uvs: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> np.ndarray:
    """Interpolate legacy Open3D vertex colours for raycasting hits."""

    result = np.zeros((*primitive_ids.shape, 3), dtype=np.float32)
    hit = primitive_ids != np.iinfo(np.uint32).max
    if not hit.any() or len(vertex_colors) == 0:
        return result
    ids = primitive_ids[hit].astype(np.int64, copy=False)
    uv = primitive_uvs[hit]
    weights = np.stack((1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]), axis=-1)
    result[hit] = np.sum(vertex_colors[triangles[ids]] * weights[..., None], axis=1)
    return np.clip(result, 0.0, 1.0)


def apply_plane_fallback(
    *,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    mesh_t_hit: np.ndarray,
    plane: np.ndarray,
    replace_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill mesh misses and collapse near-wall fragments onto one dominant plane."""

    coefficients = np.asarray(plane, dtype=np.float64)
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("plane must contain four finite coefficients")
    norm = np.linalg.norm(coefficients[:3])
    if norm <= 1e-12:
        raise ValueError("plane normal must be non-zero")
    coefficients = coefficients / norm
    denominator = np.sum(ray_directions * coefficients[:3], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        plane_t = -(np.sum(ray_origins * coefficients[:3], axis=-1) + coefficients[3]) / denominator
    plane_valid = np.isfinite(plane_t) & (plane_t > 0) & (np.abs(denominator) > 1e-8)
    mesh_valid = np.isfinite(mesh_t_hit) & (mesh_t_hit > 0)
    mesh_points = ray_origins + np.where(mesh_valid, mesh_t_hit, 0.0)[..., None] * ray_directions
    mesh_plane_distance = np.abs(np.sum(mesh_points * coefficients[:3], axis=-1) + coefficients[3])
    replace = plane_valid & (~mesh_valid | (mesh_plane_distance <= replace_band) | (mesh_t_hit > plane_t))
    combined_t = np.where(replace, plane_t, mesh_t_hit)
    return combined_t, replace, plane_valid


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), action="append", default=None)
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--auto-scale-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument(
        "--portable-manifest-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store dataset-relative image paths and manifest-relative depth/mesh paths. "
            "Disable only for legacy absolute-path receipts."
        ),
    )
    parser.add_argument(
        "--mesh-coordinate-scale",
        type=float,
        default=1.0,
        help="Multiply mesh vertices before raycasting; normally one when fusion/render parser settings match.",
    )
    parser.add_argument("--write-color-png", action=argparse.BooleanOptionalAction, default=True)
    plane = parser.add_mutually_exclusive_group()
    plane.add_argument("--fallback-plane", type=float, nargs=4, metavar=("A", "B", "C", "D"))
    plane.add_argument(
        "--fit-dominant-plane",
        action="store_true",
        help="Fit a geometric RANSAC plane to mesh vertices and use it behind foreground surfaces.",
    )
    parser.add_argument("--plane-distance-threshold", type=float, default=0.008)
    parser.add_argument(
        "--plane-replace-band",
        type=float,
        default=0.05,
        help="Replace fragmented mesh hits this close to the dominant plane; foreground remains untouched.",
    )
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.mesh = args.mesh.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.split = args.split or ["train", "val"]
    if not args.data.is_dir() or not args.mesh.is_file():
        parser.error("--data and --mesh must exist")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if len(set(args.split)) != len(args.split):
        parser.error("--split values must be unique")
    if not math.isfinite(args.mesh_coordinate_scale) or args.mesh_coordinate_scale <= 0:
        parser.error("--mesh-coordinate-scale must be finite and positive")
    if args.plane_distance_threshold <= 0 or args.plane_replace_band < 0:
        parser.error("plane distance threshold must be positive and replacement band non-negative")
    return args


def manifest_reference(path: Path, *, anchor: Path, portable: bool) -> str:
    """Return a portable path relative to the directory that owns a manifest."""

    path = path.resolve()
    if not portable:
        return str(path)
    return os.path.relpath(path, anchor.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for TSDF mesh raycasting") from error

    legacy = o3d.io.read_triangle_mesh(str(args.mesh))
    if len(legacy.triangles) == 0:
        raise ValueError(f"Mesh contains no triangles: {args.mesh}")
    if args.mesh_coordinate_scale != 1.0:
        legacy.scale(float(args.mesh_coordinate_scale), center=(0.0, 0.0, 0.0))
    triangles = np.asarray(legacy.triangles, dtype=np.int64)
    vertex_colors = np.asarray(legacy.vertex_colors, dtype=np.float32)
    fallback_plane = None if args.fallback_plane is None else np.asarray(args.fallback_plane, dtype=np.float64)
    plane_inlier_fraction = None
    if args.fit_dominant_plane:
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(legacy.vertices)))
        o3d.utility.random.seed(42)
        fitted, inliers = points.segment_plane(
            distance_threshold=float(args.plane_distance_threshold),
            ransac_n=3,
            num_iterations=5000,
            probability=0.999,
        )
        fallback_plane = np.asarray(fitted, dtype=np.float64)
        plane_inlier_fraction = len(inliers) / max(len(legacy.vertices), 1)
        print(
            f"dominant_plane={fallback_plane.tolist()} inlier_fraction={plane_inlier_fraction:.6f}",
            flush=True,
        )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    seen_stems: set[str] = set()
    for split in args.split:
        parser_config = NerfstudioDataParserConfig(
            data=args.data,
            eval_mode=args.eval_mode,
            eval_interval=args.eval_interval,
            orientation_method=args.orientation_method,
            center_method=args.center_method,
            auto_scale_poses=args.auto_scale_poses,
            scale_factor=args.scale_factor,
            scene_scale=args.scene_scale,
            downscale_factor=args.downscale_factor,
            load_3D_points=False,
        )
        outputs = parser_config.setup().get_dataparser_outputs(split=split)
        if outputs.mask_filenames is not None:
            raise ValueError("TSDF diagnostic forbids image/person masks")
        cameras = outputs.cameras.to("cpu")
        for image_index, image_path in enumerate(outputs.image_filenames):
            stem = image_path.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            width = int(cameras.width[image_index].item())
            height = int(cameras.height[image_index].item())
            intrinsic = np.asarray(
                [
                    [float(cameras.fx[image_index].item()), 0.0, float(cameras.cx[image_index].item())],
                    [0.0, float(cameras.fy[image_index].item()), float(cameras.cy[image_index].item())],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            extrinsic = nerfstudio_c2w_to_opencv_extrinsic(cameras.camera_to_worlds[image_index].numpy()).astype(
                np.float32
            )
            rays = scene.create_rays_pinhole(
                intrinsic_matrix=o3d.core.Tensor(intrinsic),
                extrinsic_matrix=o3d.core.Tensor(extrinsic),
                width_px=width,
                height_px=height,
            )
            result = scene.cast_rays(rays)
            rays_array = rays.numpy()
            t_hit = result["t_hit"].numpy()
            plane_replaced = np.zeros(t_hit.shape, dtype=bool)
            plane_valid = np.zeros(t_hit.shape, dtype=bool)
            if fallback_plane is not None:
                t_hit, plane_replaced, plane_valid = apply_plane_fallback(
                    ray_origins=rays_array[..., :3],
                    ray_directions=rays_array[..., 3:],
                    mesh_t_hit=t_hit,
                    plane=fallback_plane,
                    replace_band=float(args.plane_replace_band),
                )
            hit = np.isfinite(t_hit)
            points = rays_array[..., :3] + np.where(hit, t_hit, 0.0)[..., None] * rays_array[..., 3:]
            camera_points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            normalized_depth = np.where(hit & (camera_points[..., 2] > 0), camera_points[..., 2], 0.0)
            # Dataset loaders will multiply saved depth by dataparser_scale.
            saved_depth = (normalized_depth / float(outputs.dataparser_scale)).astype(np.float32)
            depth_path = args.output_dir / f"{stem}.npy.gz"
            with gzip.open(depth_path, "wb", compresslevel=6) as stream:
                np.save(stream, saved_depth, allow_pickle=False)
            if args.write_color_png:
                color = barycentric_vertex_colors(
                    primitive_ids=result["primitive_ids"].numpy(),
                    primitive_uvs=result["primitive_uvs"].numpy(),
                    triangles=triangles,
                    vertex_colors=vertex_colors,
                )
                color[plane_replaced] = np.asarray([0.55, 0.28, 0.16], dtype=np.float32)
                Image.fromarray(np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8)).save(
                    args.output_dir / f"{stem}.mesh_color.png"
                )
            positive = saved_depth > 0
            rows.append(
                {
                    "split": split,
                    "image_index": image_index,
                    "image": (
                        str(image_path.resolve().relative_to(args.data))
                        if args.portable_manifest_paths and image_path.resolve().is_relative_to(args.data)
                        else manifest_reference(
                            image_path,
                            anchor=args.output_dir,
                            portable=args.portable_manifest_paths,
                        )
                    ),
                    "depth": manifest_reference(
                        depth_path,
                        anchor=args.output_dir,
                        portable=args.portable_manifest_paths,
                    ),
                    "valid_pixel_fraction": float(positive.mean()),
                    "depth_median": float(np.median(saved_depth[positive])) if positive.any() else None,
                    "plane_valid_fraction": float(plane_valid.mean()),
                    "plane_replaced_fraction": float(plane_replaced.mean()),
                }
            )
            print(f"rendered={len(rows)} split={split} stem={stem} valid={positive.mean():.6f}", flush=True)

    manifest = {
        "schema_version": 1,
        "method": "open3d_tsdf_mesh_first_hit_camera_z",
        "data": str(args.data),
        "mesh": manifest_reference(
            args.mesh,
            anchor=args.output_dir,
            portable=args.portable_manifest_paths,
        ),
        "mesh_sha256": sha256(args.mesh),
        "splits": args.split,
        "masks": False,
        "parameters": {
            "eval_mode": args.eval_mode,
            "eval_interval": args.eval_interval,
            "orientation_method": args.orientation_method,
            "center_method": args.center_method,
            "auto_scale_poses": args.auto_scale_poses,
            "scale_factor": args.scale_factor,
            "scene_scale": args.scene_scale,
            "downscale_factor": args.downscale_factor,
            "portable_manifest_paths": args.portable_manifest_paths,
            "mesh_coordinate_scale": args.mesh_coordinate_scale,
            "fallback_plane": None if fallback_plane is None else fallback_plane.tolist(),
            "fit_dominant_plane": args.fit_dominant_plane,
            "plane_distance_threshold": args.plane_distance_threshold,
            "plane_replace_band": args.plane_replace_band,
            "plane_inlier_fraction": plane_inlier_fraction,
        },
        "images": rows,
    }
    (args.output_dir / "mesh_depth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"complete images={len(rows)} output={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
