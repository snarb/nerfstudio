#!/usr/bin/env python3
"""Render an eval camera by calibrated train-image reprojection through one mesh.

This is a geometry/appearance causal gate, not a learned image refiner.  It
never reads eval RGB while constructing predictions and uses no image/person
mask.  The default admits pixels only when the same continuous mesh is visible
in both the target and source cameras.  An explicit hard-colour continuation
mode can reuse the nearest train camera's 3D reprojection across small,
photometrically discontinuous visibility gaps; it never averages sources.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig


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


def load_exr_image(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.ndim != 3 or encoded.shape[-1] not in (3, 4):
        raise ValueError(f"Failed to decode RGB(A) OpenEXR: {path}")
    order = (2, 1, 0) if encoded.shape[-1] == 3 else (2, 1, 0, 3)
    return np.ascontiguousarray(encoded[..., order], dtype=np.float32)


def write_exr_image(path: Path, image: np.ndarray | torch.Tensor) -> None:
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"OpenEXR output must have RGB(A) channels, got {array.shape}")
    order = (2, 1, 0) if array.shape[-1] == 3 else (2, 1, 0, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(array[..., order])):
        raise OSError(f"Failed to write OpenEXR: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mesh-depth-manifest", type=Path, required=True)
    parser.add_argument(
        "--metric-surface-depth-manifest",
        type=Path,
        default=None,
        help=(
            "Optional independent mesh-depth manifest whose eval first-hit support defines surface metrics. "
            "Rendering and visibility still use --mesh-depth-manifest. The default preserves legacy behavior."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-prediction-exr", type=Path, default=None)
    parser.add_argument("--ground-truth-exr", type=Path, default=None)
    parser.add_argument(
        "--skip-ground-truth-copy", action="store_true",
        help="Render synthetic cameras without opening target RGB; incompatible with metric scoring.",
    )
    parser.add_argument("--neighbors", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--primary-angular-camera-count", type=int, default=0)
    parser.add_argument("--seam-cut-rank-penalty", type=float, default=0.0001)
    parser.add_argument("--camera-color-calibration", type=Path, default=None)
    parser.add_argument("--camera-color-model", choices=("ingest", "exposure", "rgb", "spatial"), default="rgb")
    parser.add_argument("--overlap-exposure-grid", type=int, nargs=2, default=None)
    parser.add_argument("--write-source-warp-audit", action="store_true")
    parser.add_argument(
        "--aggregation-modes",
        nargs="+",
        choices=("weighted", "nearest-fill", "best-view", "seam-cut"),
        default=("weighted",),
        help=(
            "weighted preserves the legacy colour average; nearest-fill takes the globally nearest valid source "
            "and uses later sources only for holes; best-view makes a hard per-pixel choice from geometric "
            "confidence, target-view angular similarity, projected resolution, and image-border margin."
        ),
    )
    parser.add_argument(
        "--blend-alphas",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Geometric source-colour blend strengths; one fully replaces supported base pixels.",
    )
    parser.add_argument("--depth-log-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--source-observed-depth-data", type=Path, default=None,
        help="Optional calibrated train depth dataset used to veto foreground occluders absent from the mesh.",
    )
    parser.add_argument("--source-observed-depth-log-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--depth-hole-fill-max-area",
        type=int,
        default=0,
        help=(
            "Fill enclosed mesh-depth holes up to this many pixels by a locally fitted camera-z plane. "
            "Zero preserves the historical renderer exactly."
        ),
    )
    parser.add_argument("--depth-hole-fill-boundary-radius", type=int, default=4)
    parser.add_argument("--depth-hole-fill-max-relative-plane-rmse", type=float, default=0.015)
    parser.add_argument(
        "--target-depth-component-min-area",
        type=int,
        default=0,
        help=(
            "Remove target-view mesh-depth components smaller than this many pixels after cutting "
            "connections across large relative depth jumps. Zero preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--target-depth-component-max-log-jump",
        type=float,
        default=0.0075,
        help="Maximum adjacent |delta log(depth)| used by the optional target-view component filter.",
    )
    add_boolean_argument(
        parser,
        "--nearest-fill-color-continuity",
        default=False,
        help=(
            "For nearest-fill fallback pixels, hard-select the valid train reprojection closest in RGB "
            "to the nearest valid first-source pixel; no source colours are averaged."
        ),
    )
    parser.add_argument(
        "--nearest-fill-color-continuity-mode",
        choices=("pixel", "global"),
        default="pixel",
        help=(
            "pixel hard-selects a fallback at every disoccluded pixel; global reorders whole fallback "
            "cameras by their median overlap colour error to the first source."
        ),
    )
    parser.add_argument(
        "--nearest-fill-rank-penalty",
        type=float,
        default=0.0,
        help="Non-negative per-rank penalty added to fallback RGB distance.",
    )
    add_boolean_argument(
        parser,
        "--nearest-fill-primary-color-continuation",
        default=False,
        help=(
            "For small connected nearest-fill visibility gaps with a sharp colour discontinuity, "
            "hard-select the nearest train camera's RGB at the same 3D reprojection. This is disabled "
            "by default and never averages sources or reads eval RGB."
        ),
    )
    parser.add_argument("--nearest-fill-primary-color-continuation-min-area", type=int, default=20)
    parser.add_argument("--nearest-fill-primary-color-continuation-max-area", type=int, default=1000)
    parser.add_argument(
        "--nearest-fill-primary-color-continuation-min-median-l1",
        type=float,
        default=0.1,
    )
    parser.add_argument("--best-view-angle-power", type=float, default=4.0)
    parser.add_argument("--best-view-border-margin", type=float, default=64.0)
    parser.add_argument(
        "--camera-distance-power",
        type=float,
        default=2.0,
        help=(
            "Exponent p in the source-camera weight 1 / distance**p. "
            "Larger values keep the nearest source sharper while allowing farther sources to fill visibility holes."
        ),
    )
    parser.add_argument(
        "--detail-transfer-sigmas",
        type=float,
        nargs="*",
        default=(),
        help=(
            "Optional Gaussian sigmas for replacing only the base high-frequency band with the "
            "geometry-aligned source band while retaining base low-frequency colour."
        ),
    )
    parser.add_argument(
        "--detail-transfer-strengths",
        type=float,
        nargs="+",
        default=(1.0,),
    )
    add_boolean_argument(
        parser,
        "--score-metrics",
        default=False,
        help="Score display-domain PSNR/SSIM/LPIPS after every prediction has been constructed.",
    )
    parser.add_argument(
        "--metric-regions",
        nargs="+",
        choices=("full", "roi", "surface", "surface-roi", "reprojected", "reprojected-roi"),
        default=None,
        help=(
            "Regions to score. The legacy default is full plus roi when --roi-boxes-json is set. "
            "surface uses the target mesh first-hit support; surface-roi intersects it with the ROI. "
            "reprojected variants use only target-surface pixels visible in at least one selected source."
        ),
    )
    parser.add_argument(
        "--roi-boxes-json",
        type=Path,
        default=None,
        help="Optional diagnostic JSON containing exactly one boxes_xyxy entry.",
    )
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--center-method", default="focus")
    add_boolean_argument(parser, "--auto-scale-poses", default=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.mesh_depth_manifest = args.mesh_depth_manifest.expanduser().resolve()
    if args.metric_surface_depth_manifest is not None:
        args.metric_surface_depth_manifest = args.metric_surface_depth_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.base_prediction_exr is not None:
        args.base_prediction_exr = args.base_prediction_exr.expanduser().resolve()
    if args.ground_truth_exr is not None:
        args.ground_truth_exr = args.ground_truth_exr.expanduser().resolve()
    if args.roi_boxes_json is not None:
        args.roi_boxes_json = args.roi_boxes_json.expanduser().resolve()
    if not args.data.is_dir() or not args.mesh_depth_manifest.is_file():
        parser.error("--data and --mesh-depth-manifest must exist")
    if args.metric_surface_depth_manifest is not None and not args.metric_surface_depth_manifest.is_file():
        parser.error("--metric-surface-depth-manifest must exist")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.neighbors or min(args.neighbors) <= 0:
        parser.error("--neighbors must contain positive counts")
    if not args.blend_alphas or any(not 0.0 <= value <= 1.0 for value in args.blend_alphas):
        parser.error("--blend-alphas must contain values in [0, 1]")
    if args.depth_log_tolerance <= 0.0:
        parser.error("--depth-log-tolerance must be positive")
    if args.primary_angular_camera_count < 0:
        parser.error("--primary-angular-camera-count must be non-negative")
    if args.overlap_exposure_grid is not None and min(args.overlap_exposure_grid)<2:
        parser.error("Overlap exposure grid must have at least two nodes per axis")
    if args.overlap_exposure_grid is not None and args.camera_color_calibration is None:
        parser.error("Overlap exposure correction requires an audited camera calibration and ingest curve")
    if not math.isfinite(args.seam_cut_rank_penalty) or args.seam_cut_rank_penalty < 0:
        parser.error("--seam-cut-rank-penalty must be finite and non-negative")
    if args.skip_ground_truth_copy and (args.score_metrics or args.ground_truth_exr is not None):
        parser.error("--skip-ground-truth-copy forbids ground truth and metric scoring")
    if args.depth_hole_fill_max_area < 0:
        parser.error("--depth-hole-fill-max-area must be non-negative")
    if args.depth_hole_fill_boundary_radius <= 0:
        parser.error("--depth-hole-fill-boundary-radius must be positive")
    if args.depth_hole_fill_max_relative_plane_rmse <= 0.0:
        parser.error("--depth-hole-fill-max-relative-plane-rmse must be positive")
    if args.target_depth_component_min_area < 0:
        parser.error("--target-depth-component-min-area must be non-negative")
    if not math.isfinite(args.target_depth_component_max_log_jump) or args.target_depth_component_max_log_jump <= 0.0:
        parser.error("--target-depth-component-max-log-jump must be positive")
    if not math.isfinite(args.nearest_fill_rank_penalty) or args.nearest_fill_rank_penalty < 0.0:
        parser.error("--nearest-fill-rank-penalty must be non-negative")
    if args.nearest_fill_primary_color_continuation_min_area <= 0:
        parser.error("--nearest-fill-primary-color-continuation-min-area must be positive")
    if (
        args.nearest_fill_primary_color_continuation_max_area
        < args.nearest_fill_primary_color_continuation_min_area
    ):
        parser.error(
            "--nearest-fill-primary-color-continuation-max-area must be at least the minimum area"
        )
    if (
        not math.isfinite(args.nearest_fill_primary_color_continuation_min_median_l1)
        or args.nearest_fill_primary_color_continuation_min_median_l1 < 0.0
    ):
        parser.error("--nearest-fill-primary-color-continuation-min-median-l1 must be non-negative")
    if args.best_view_angle_power < 0.0:
        parser.error("--best-view-angle-power must be non-negative")
    if args.best_view_border_margin <= 0.0:
        parser.error("--best-view-border-margin must be positive")
    if args.camera_distance_power < 0.0:
        parser.error("--camera-distance-power must be non-negative")
    if any(value <= 0.0 for value in args.detail_transfer_sigmas):
        parser.error("--detail-transfer-sigmas must be positive")
    if any(value < 0.0 for value in args.detail_transfer_strengths):
        parser.error("--detail-transfer-strengths must be non-negative")
    if args.roi_boxes_json is not None and not args.roi_boxes_json.is_file():
        parser.error("--roi-boxes-json must exist")
    if args.roi_boxes_json is not None and not args.score_metrics:
        parser.error("--roi-boxes-json requires --score-metrics")
    if args.metric_regions is not None and "roi" in args.metric_regions and args.roi_boxes_json is None:
        parser.error("--metric-regions roi requires --roi-boxes-json")
    if args.metric_regions is not None and "surface-roi" in args.metric_regions and args.roi_boxes_json is None:
        parser.error("--metric-regions surface-roi requires --roi-boxes-json")
    if args.metric_regions is not None and "reprojected-roi" in args.metric_regions and args.roi_boxes_json is None:
        parser.error("--metric-regions reprojected-roi requires --roi-boxes-json")
    return args


def load_depth(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        depth = np.load(stream, allow_pickle=False)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError(f"Invalid non-negative depth: {path}")
    return depth.astype(np.float32, copy=False)


def filter_small_target_depth_components(
    depth: np.ndarray,
    *,
    min_area: int,
    max_log_jump: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove small depth-coherent target-view lobes without an image/person mask.

    Ordinary 2D connected components cannot separate a thin background lobe
    attached to the subject by one depth-discontinuous bridge.  This graph
    connects eight-neighbour pixels only when their camera-z values agree in
    log space, then removes only components below ``min_area``.
    """

    source = np.asarray(depth, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("depth must be a finite non-negative 2D array")
    stats: dict[str, object] = {
        "enabled": min_area > 0,
        "min_area": min_area,
        "max_log_jump": max_log_jump,
        "components_before": 0,
        "components_removed": 0,
        "pixels_removed": 0,
        "largest_removed_components": [],
    }
    if min_area <= 0:
        return source, stats

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    valid = source > 0.0
    valid_count = int(valid.sum())
    if valid_count == 0:
        return source, stats
    height, width = source.shape
    identifiers = np.full(source.shape, -1, dtype=np.int32)
    identifiers[valid] = np.arange(valid_count, dtype=np.int32)
    log_depth = np.zeros_like(source)
    log_depth[valid] = np.log(source[valid])
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for delta_y, delta_x in ((0, 1), (1, 0), (1, 1), (1, -1)):
        x0 = max(0, -delta_x)
        x1 = min(width, width - delta_x)
        y1 = height - delta_y
        adjacent = valid[:y1, x0:x1] & valid[delta_y:, x0 + delta_x:x1 + delta_x]
        adjacent &= (
            np.abs(
                log_depth[:y1, x0:x1]
                - log_depth[delta_y:, x0 + delta_x:x1 + delta_x]
            )
            <= max_log_jump
        )
        first = identifiers[:y1, x0:x1][adjacent]
        second = identifiers[delta_y:, x0 + delta_x:x1 + delta_x][adjacent]
        rows.extend((first, second))
        columns.extend((second, first))
    graph = coo_matrix(
        (
            np.ones(sum(len(item) for item in rows), dtype=np.uint8),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(valid_count, valid_count),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=component_count)
    remove_nodes = sizes[labels] < min_area
    yy, xx = np.nonzero(valid)
    filtered = source.copy()
    filtered[yy[remove_nodes], xx[remove_nodes]] = 0.0
    removed_rows = []
    for component in np.flatnonzero(sizes < min_area):
        selected = labels == component
        removed_rows.append(
            {
                "area": int(sizes[component]),
                "bbox_xyxy": [
                    int(xx[selected].min()),
                    int(yy[selected].min()),
                    int(xx[selected].max() + 1),
                    int(yy[selected].max() + 1),
                ],
                "median_depth": float(np.median(source[valid][selected])),
            }
        )
    stats.update(
        {
            "components_before": int(component_count),
            "components_removed": len(removed_rows),
            "pixels_removed": int(remove_nodes.sum()),
            "largest_removed_components": sorted(
                removed_rows, key=lambda row: int(row["area"]), reverse=True
            )[:16],
        }
    )
    return filtered, stats


def fill_small_consistent_depth_holes(
    depth: np.ndarray,
    *,
    max_area: int,
    boundary_radius: int,
    max_relative_plane_rmse: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fill small enclosed camera-z holes only when their boundary supports one local plane."""

    source = np.asarray(depth, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0.0):
        raise ValueError("depth must be a finite non-negative 2D array")
    stats: dict[str, object] = {
        "enabled": max_area > 0,
        "max_area": max_area,
        "boundary_radius": boundary_radius,
        "max_relative_plane_rmse": max_relative_plane_rmse,
        "candidate_holes": 0,
        "filled_holes": 0,
        "filled_pixels": 0,
    }
    if max_area <= 0:
        return source, stats

    from scipy import ndimage

    valid = source > 0.0
    enclosed = ndimage.binary_fill_holes(valid) & ~valid
    labels, count = ndimage.label(enclosed)
    stats["candidate_holes"] = int(count)
    filled = source.copy()
    accepted: list[dict[str, object]] = []
    for label_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == label_id
        area = int(component.sum())
        if area == 0 or area > max_area:
            continue
        y0 = max(component_slice[0].start - boundary_radius, 0)
        y1 = min(component_slice[0].stop + boundary_radius, source.shape[0])
        x0 = max(component_slice[1].start - boundary_radius, 0)
        x1 = min(component_slice[1].stop + boundary_radius, source.shape[1])
        local_component = labels[y0:y1, x0:x1] == label_id
        local_valid = valid[y0:y1, x0:x1]
        boundary = ndimage.binary_dilation(local_component, iterations=boundary_radius) & local_valid
        boundary_y, boundary_x = np.nonzero(boundary)
        if boundary_y.size < 12:
            continue
        boundary_z = source[y0:y1, x0:x1][boundary].astype(np.float64, copy=False)
        centre_x = float(boundary_x.mean())
        centre_y = float(boundary_y.mean())
        coordinate_scale = float(max(np.ptp(boundary_x), np.ptp(boundary_y), 1))
        design = np.stack(
            (
                (boundary_x - centre_x) / coordinate_scale,
                (boundary_y - centre_y) / coordinate_scale,
                np.ones_like(boundary_x),
            ),
            axis=-1,
        )
        coefficients, *_ = np.linalg.lstsq(design, boundary_z, rcond=None)
        residual = boundary_z - design @ coefficients
        median_depth = float(np.median(boundary_z))
        relative_rmse = float(np.sqrt(np.mean(np.square(residual))) / max(median_depth, 1e-8))
        if not np.isfinite(relative_rmse) or relative_rmse > max_relative_plane_rmse:
            continue
        hole_y, hole_x = np.nonzero(local_component)
        hole_design = np.stack(
            (
                (hole_x - centre_x) / coordinate_scale,
                (hole_y - centre_y) / coordinate_scale,
                np.ones_like(hole_x),
            ),
            axis=-1,
        )
        predicted = hole_design @ coefficients
        if not np.isfinite(predicted).all() or np.any(predicted <= 0.0):
            continue
        local_filled = filled[y0:y1, x0:x1]
        local_filled[local_component] = predicted.astype(np.float32, copy=False)
        stats["filled_holes"] = int(stats["filled_holes"]) + 1
        stats["filled_pixels"] = int(stats["filled_pixels"]) + area
        accepted.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(component_slice[1].start),
                    int(component_slice[0].start),
                    int(component_slice[1].stop),
                    int(component_slice[0].stop),
                ],
                "relative_plane_rmse": relative_rmse,
            }
        )
    stats["largest_filled_holes"] = sorted(accepted, key=lambda row: int(row["area"]), reverse=True)[:16]
    return filled, stats


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() == ".exr":
        array = load_exr_image(path)[..., :3].astype(np.float32, copy=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).to(device)


def resolve_manifest_path(value: str, data: Path, manifest_path: Path) -> Path:
    """Resolve portable dataset-relative manifests without depending on cwd."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (data / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Prefer the dataset contract in the error that the eventual loader emits.
    return candidates[0].resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_parameters(cameras, index: int, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    c2w = cameras.camera_to_worlds[index].to(device=device, dtype=torch.float32)
    intrinsics = {
        "fx": float(cameras.fx[index].item()),
        "fy": float(cameras.fy[index].item()),
        "cx": float(cameras.cx[index].item()),
        "cy": float(cameras.cy[index].item()),
        "width": int(cameras.width[index].item()),
        "height": int(cameras.height[index].item()),
    }
    return c2w, intrinsics


def project_target_to_source(
    target_depth: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsics: dict[str, float],
    source_c2w: torch.Tensor,
    source_intrinsics: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project positive OpenCV camera-z depth between Nerfstudio cameras."""

    world = target_depth_to_world(target_depth, target_c2w, target_intrinsics)
    source_gl = (world - source_c2w[:, 3]) @ source_c2w[:, :3]
    source_z = -source_gl[..., 2]
    safe_z = source_z.clamp_min(1e-8)
    source_u = source_intrinsics["fx"] * source_gl[..., 0] / safe_z + source_intrinsics["cx"]
    source_v = source_intrinsics["fy"] * (-source_gl[..., 1]) / safe_z + source_intrinsics["cy"]
    return source_u, source_v, source_z


def target_depth_to_world(
    target_depth: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsics: dict[str, float],
) -> torch.Tensor:
    """Unproject positive OpenCV camera-z depth into Nerfstudio world space."""

    height, width = target_depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=target_depth.device),
        torch.arange(width, dtype=torch.float32, device=target_depth.device),
        indexing="ij",
    )
    z = target_depth
    x = (xx - target_intrinsics["cx"]) * z / target_intrinsics["fx"]
    y = (yy - target_intrinsics["cy"]) * z / target_intrinsics["fy"]
    target_gl = torch.stack((x, -y, -z), dim=-1)
    return target_gl @ target_c2w[:, :3].T + target_c2w[:, 3]


def best_view_score(
    *,
    valid: torch.Tensor,
    depth_confidence: torch.Tensor,
    target_world: torch.Tensor,
    target_camera_center: torch.Tensor,
    source_camera_center: torch.Tensor,
    target_depth: torch.Tensor,
    projected_depth: torch.Tensor,
    source_u: torch.Tensor,
    source_v: torch.Tensor,
    target_intrinsics: dict[str, float],
    source_intrinsics: dict[str, float],
    angle_power: float,
    border_margin: float,
) -> torch.Tensor:
    """Score source quality per target pixel without averaging source colours."""

    target_direction = F.normalize(target_camera_center - target_world, dim=-1, eps=1e-8)
    source_direction = F.normalize(source_camera_center - target_world, dim=-1, eps=1e-8)
    angular_similarity = (target_direction * source_direction).sum(dim=-1).clamp(0.0, 1.0)
    angular_score = angular_similarity.pow(angle_power)
    target_pixels_per_world = (
        math.sqrt(target_intrinsics["fx"] * target_intrinsics["fy"]) / target_depth.clamp_min(1e-8)
    )
    source_pixels_per_world = (
        math.sqrt(source_intrinsics["fx"] * source_intrinsics["fy"]) / projected_depth.clamp_min(1e-8)
    )
    resolution_score = (source_pixels_per_world / target_pixels_per_world).clamp(0.0, 1.0)
    border_distance = torch.minimum(
        torch.minimum(source_u, source_intrinsics["width"] - 1.0 - source_u),
        torch.minimum(source_v, source_intrinsics["height"] - 1.0 - source_v),
    )
    border_score = border_distance.div(border_margin).clamp(0.0, 1.0)
    score = depth_confidence * angular_score * resolution_score * (0.25 + 0.75 * border_score)
    return torch.where(valid, score, torch.full_like(score, -torch.inf))


def aggregate_warped_sources(
    warped: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    weighted_scores: list[torch.Tensor],
    best_scores: list[torch.Tensor],
    *,
    mode: str,
    nearest_fill_color_continuity: bool = False,
    nearest_fill_color_continuity_mode: str = "pixel",
    nearest_fill_rank_penalty: float = 0.0,
    seam_cut_rank_penalty: float = 0.0001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine source warps and return RGB, valid mask, and selected source rank."""

    if not warped or not (len(warped) == len(valid_masks) == len(weighted_scores) == len(best_scores)):
        raise ValueError("source lists must be non-empty and have matching lengths")
    rgb_stack = torch.stack(warped)
    valid_stack = torch.stack(valid_masks)
    valid = valid_stack.any(dim=0)
    if mode == "weighted":
        score_stack = torch.stack(weighted_scores)
        weight_sum = score_stack.sum(dim=0)
        rgb = (rgb_stack * score_stack.unsqueeze(1)).sum(dim=0) / weight_sum.unsqueeze(0).clamp_min(1e-8)
        selected = torch.argmax(score_stack, dim=0)
    elif mode == "nearest-fill":
        selected = torch.argmax(valid_stack.to(dtype=torch.int64), dim=0)
        first_valid = valid_stack[0]
        if nearest_fill_color_continuity and len(warped) > 1 and bool(first_valid.any()):
            if nearest_fill_color_continuity_mode not in {"pixel", "global"}:
                raise ValueError("nearest-fill continuity mode must be pixel or global")
            from scipy import ndimage

            _, nearest_indices = ndimage.distance_transform_edt(
                ~first_valid.detach().cpu().numpy(), return_indices=True
            )
            nearest_y = torch.from_numpy(nearest_indices[0]).to(device=rgb_stack.device)
            nearest_x = torch.from_numpy(nearest_indices[1]).to(device=rgb_stack.device)
            reference = rgb_stack[0][:, nearest_y, nearest_x]
            color_distance = (rgb_stack - reference.unsqueeze(0)).abs().mean(dim=1)
            rank_penalty = (
                torch.arange(len(warped), device=rgb_stack.device, dtype=color_distance.dtype)
                .view(-1, 1, 1)
                .mul(nearest_fill_rank_penalty)
            )
            if nearest_fill_color_continuity_mode == "pixel":
                fallback_cost = torch.where(
                    valid_stack,
                    color_distance + rank_penalty,
                    torch.full_like(color_distance, torch.inf),
                )
                continuity_selected = torch.argmin(fallback_cost, dim=0)
            else:
                global_costs = [torch.zeros((), device=rgb_stack.device, dtype=rgb_stack.dtype)]
                for rank in range(1, len(warped)):
                    overlap = first_valid & valid_stack[rank]
                    overlap_error = (
                        torch.median(color_distance[rank][overlap])
                        if bool(overlap.any())
                        else torch.full((), torch.inf, device=rgb_stack.device, dtype=rgb_stack.dtype)
                    )
                    global_costs.append(overlap_error + nearest_fill_rank_penalty * rank)
                source_order = torch.argsort(torch.stack(global_costs))
                ordered_valid = valid_stack[source_order]
                selected_order = torch.argmax(ordered_valid.to(dtype=torch.int64), dim=0)
                continuity_selected = source_order[selected_order]
            selected = torch.where(first_valid, torch.zeros_like(selected), continuity_selected)
        gather_index = selected.unsqueeze(0).unsqueeze(0).expand(1, rgb_stack.shape[1], *selected.shape)
        rgb = torch.gather(rgb_stack, dim=0, index=gather_index)[0]
    elif mode == "seam-cut":
        from hard_texture_seam_cut import optimize_source_labels

        labels, seam_stats = optimize_source_labels(
            rgb_stack.permute(0, 2, 3, 1).detach().cpu().numpy(),
            valid_stack.detach().cpu().numpy(),
            rank_penalty=seam_cut_rank_penalty,
        )
        print("seam_cut=" + json.dumps(seam_stats), flush=True)
        selected = torch.from_numpy(labels).to(device=rgb_stack.device, dtype=torch.long).clamp_min(0)
        gather_index = selected.unsqueeze(0).unsqueeze(0).expand(1, rgb_stack.shape[1], *selected.shape)
        rgb = torch.gather(rgb_stack, dim=0, index=gather_index)[0]
    elif mode == "best-view":
        score_stack = torch.stack(best_scores)
        selected = torch.argmax(score_stack, dim=0)
        gather_index = selected.unsqueeze(0).unsqueeze(0).expand(1, rgb_stack.shape[1], *selected.shape)
        rgb = torch.gather(rgb_stack, dim=0, index=gather_index)[0]
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    selected = torch.where(valid, selected, torch.full_like(selected, -1))
    return rgb, valid.unsqueeze(0), selected


def continue_primary_color_across_small_fallback_components(
    rgb: torch.Tensor,
    selection: torch.Tensor,
    primary_warped: torch.Tensor,
    primary_valid: torch.Tensor,
    primary_projectable: torch.Tensor,
    *,
    min_area: int,
    max_area: int,
    min_median_l1: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Replace only sharp, small fallback islands with hard primary-camera RGB.

    The component decision uses train reprojections and geometry-validity maps
    only.  ``primary_projectable`` deliberately excludes the source-depth
    consistency test: selected pixels retain their target-depth 3D projection,
    but take colour from the closest train camera to prevent a small visibility
    gap from becoming a multi-camera texture shard.
    """

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("rgb must be CHW with three channels")
    if primary_warped.shape != rgb.shape:
        raise ValueError("primary_warped must match rgb")
    if any(mask.shape != selection.shape for mask in (primary_valid, primary_projectable)):
        raise ValueError("selection and primary masks must have matching shapes")
    if min_area <= 0 or max_area < min_area:
        raise ValueError("component area bounds are invalid")
    if not math.isfinite(min_median_l1) or min_median_l1 < 0.0:
        raise ValueError("min_median_l1 must be finite and non-negative")

    from scipy import ndimage

    fallback = (selection >= 0) & ~primary_valid
    stats: dict[str, object] = {
        "enabled": True,
        "min_area": min_area,
        "max_area": max_area,
        "min_median_l1": min_median_l1,
        "candidate_components": 0,
        "selected_components": 0,
        "selected_component_pixels": 0,
        "replaced_pixels": 0,
        "components": [],
    }
    if not bool(fallback.any()) or not bool(primary_valid.any()):
        return rgb, selection, stats

    primary_valid_cpu = primary_valid.detach().cpu().numpy()
    _, nearest_indices = ndimage.distance_transform_edt(
        ~primary_valid_cpu, return_indices=True
    )
    nearest_y = torch.from_numpy(nearest_indices[0]).to(device=rgb.device)
    nearest_x = torch.from_numpy(nearest_indices[1]).to(device=rgb.device)
    reference = primary_warped[:, nearest_y, nearest_x]
    discontinuity = (rgb - reference).abs().mean(dim=0)
    labels, component_count = ndimage.label(
        fallback.detach().cpu().numpy(), np.ones((3, 3), dtype=np.uint8)
    )
    stats["candidate_components"] = int(component_count)
    replacement = torch.zeros_like(primary_valid)
    rows: list[dict[str, object]] = []
    for component_id in range(1, component_count + 1):
        yy, xx = np.where(labels == component_id)
        area = int(len(xx))
        if not min_area <= area <= max_area:
            continue
        component = torch.from_numpy(labels == component_id).to(device=rgb.device)
        median_l1 = float(torch.median(discontinuity[component]).item())
        if median_l1 <= min_median_l1:
            continue
        replaceable = component & primary_projectable
        replaced = int(replaceable.sum().item())
        if replaced == 0:
            continue
        replacement |= replaceable
        rows.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max()) + 1,
                    int(yy.max()) + 1,
                ],
                "median_nearest_primary_l1": median_l1,
                "replaced_pixels": replaced,
            }
        )

    if bool(replacement.any()):
        rgb = torch.where(replacement.unsqueeze(0), primary_warped, rgb)
        selection = torch.where(replacement, torch.zeros_like(selection), selection)
    stats["selected_components"] = len(rows)
    stats["selected_component_pixels"] = sum(int(row["area"]) for row in rows)
    stats["replaced_pixels"] = int(replacement.sum().item())
    stats["components"] = sorted(rows, key=lambda row: int(row["area"]), reverse=True)
    return rgb, selection, stats


def source_selection_fractions(selection: torch.Tensor, count: int) -> list[float]:
    """Summarize hard/maximum-contribution source ranks over covered pixels."""

    valid = selection >= 0
    denominator = int(valid.sum().item())
    if denominator == 0:
        return [0.0] * count
    return [float(((selection == rank).sum() / denominator).item()) for rank in range(count)]


def grid_sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    grid = torch.stack(
        (2.0 * u / max(width - 1, 1) - 1.0, 2.0 * v / max(height - 1, 1) - 1.0),
        dim=-1,
    ).unsqueeze(0)
    return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def normalized_gaussian_blur(image: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur CHW data inside a support mask without leaking zero across its boundary."""

    if image.ndim != 3 or valid.shape != (1, *image.shape[-2:]):
        raise ValueError("image must be CHW and valid must be 1HW")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    channels = image.shape[0]
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)

    def separable(value: torch.Tensor, groups: int) -> torch.Tensor:
        result = F.conv2d(
            value.unsqueeze(0),
            horizontal.expand(groups, 1, 1, -1),
            padding=(0, radius),
            groups=groups,
        )
        result = F.conv2d(
            result,
            vertical.expand(groups, 1, -1, 1),
            padding=(radius, 0),
            groups=groups,
        )
        return result[0]

    support = valid.to(dtype=image.dtype)
    numerator = separable(image * support, channels)
    denominator = separable(support, 1)
    return numerator / denominator.clamp_min(1e-6)


def surface_detail_transfer(
    *,
    base: torch.Tensor,
    source: torch.Tensor,
    valid: torch.Tensor,
    sigma: float,
    strength: float,
) -> torch.Tensor:
    """Replace the supported base high-pass band with aligned source detail."""

    base_low = normalized_gaussian_blur(base, valid, sigma)
    source_low = normalized_gaussian_blur(source, valid, sigma)
    replacement = base + strength * ((source - source_low) - (base - base_low))
    return torch.where(valid, replacement, base)


def load_single_roi(path: Path) -> tuple[tuple[int, int, int, int], str]:
    """Load the one eval ROI accepted by this single-target renderer."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_boxes = payload.get("boxes_xyxy")
    if raw_boxes is None and isinstance(payload.get("boxes"), list):
        raw_boxes = [row.get("xyxy") if isinstance(row, dict) else None for row in payload["boxes"]]
    if not isinstance(raw_boxes, list) or len(raw_boxes) != 1:
        raise ValueError(f"Expected exactly one ROI box in {path}")
    raw = raw_boxes[0]
    if not isinstance(raw, list) or len(raw) != 4 or not all(isinstance(value, int) for value in raw):
        raise ValueError(f"ROI must contain four integer xyxy coordinates: {raw!r}")
    x0, y0, x1, y1 = raw
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI box: {raw!r}")
    return (x0, y0, x1, y1), str(payload.get("name", "roi"))


def display_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    lpips_model,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, float]:
    """Measure display-domain metrics without affecting prediction construction."""

    from torchmetrics.functional.image import structural_similarity_index_measure

    if prediction.shape != ground_truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and ground_truth must be matching CHW tensors")
    pred = prediction.float().clamp(0.0, 1.0).unsqueeze(0)
    gt = ground_truth.float().clamp(0.0, 1.0).unsqueeze(0)

    def measure(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float]:
        mse = torch.mean((left - right).square())
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(left, right, data_range=1.0)
        lpips = lpips_model(left, right)
        return float(psnr.item()), float(ssim.item()), float(lpips.item())

    psnr, ssim, lpips = measure(pred, gt)
    result = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    if roi is not None:
        x0, y0, x1, y1 = roi
        height, width = pred.shape[-2:]
        if x1 > width or y1 > height:
            raise ValueError(f"ROI {roi} exceeds image width/height {(width, height)}")
        roi_psnr, roi_ssim, roi_lpips = measure(pred[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1])
        result.update({"roi_psnr": roi_psnr, "roi_ssim": roi_ssim, "roi_lpips": roi_lpips})
    return result


def masked_display_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    mask: torch.Tensor,
    lpips_model,
) -> dict[str, float | list[int]]:
    """Measure one geometry-defined region without scoring pixels outside it.

    PSNR is computed from exactly the selected RGB samples. SSIM and LPIPS need
    rectangular images, so both inputs are cropped to the tight mask bounds and
    receive the same black value outside the mask. This keeps held-out RGB out
    of prediction construction while preventing the room from affecting the
    actor-only comparison.
    """

    from torchmetrics.functional.image import structural_similarity_index_measure

    if prediction.shape != ground_truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and ground_truth must be matching CHW tensors")
    if mask.shape != prediction.shape[-2:]:
        raise ValueError("mask must match prediction height and width")
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("metric mask must select at least one pixel")
    yy, xx = torch.where(mask)
    x0, x1 = int(xx.min().item()), int(xx.max().item()) + 1
    y0, y1 = int(yy.min().item()), int(yy.max().item()) + 1
    pred = prediction.float().clamp(0.0, 1.0)
    gt = ground_truth.float().clamp(0.0, 1.0)
    selected_error = (pred[:, mask] - gt[:, mask]).square().mean()
    psnr = -10.0 * torch.log10(selected_error.clamp_min(1e-12))
    crop_mask = mask[y0:y1, x0:x1].unsqueeze(0)
    pred_crop = torch.where(crop_mask, pred[:, y0:y1, x0:x1], 0.0).unsqueeze(0)
    gt_crop = torch.where(crop_mask, gt[:, y0:y1, x0:x1], 0.0).unsqueeze(0)
    ssim = structural_similarity_index_measure(pred_crop, gt_crop, data_range=1.0)
    lpips = lpips_model(pred_crop, gt_crop)
    return {
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpips.item()),
        "pixel_fraction": float(mask.float().mean().item()),
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def score_written_variants(
    output_dir: Path,
    variants: list[dict[str, object]],
    ground_truth: Path,
    *,
    device: torch.device,
    roi_boxes_json: Path | None,
    surface_mask: torch.Tensor | None = None,
    metric_regions: list[str] | None = None,
    variant_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    """Score renderer outputs lazily so the optional path cannot affect legacy runs."""

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    roi = None
    roi_name = None
    if roi_boxes_json is not None:
        roi, roi_name = load_single_roi(roi_boxes_json)
    regions = metric_regions
    if regions is None:
        regions = ["full"] + (["roi"] if roi is not None else [])
    if any(region.startswith("surface") for region in regions) and surface_mask is None:
        raise ValueError("surface metric regions require a target surface mask")
    if any(region.startswith("reprojected") for region in regions) and variant_masks is None:
        raise ValueError("reprojected metric regions require per-variant visibility masks")
    gt = load_rgb(ground_truth, device)
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    by_variant: dict[str, object] = {}
    with torch.inference_mode():
        for row in variants:
            name = str(row["name"])
            pred = load_rgb(output_dir / name / "eval_pred_0000.exr", device)
            aggregate: dict[str, object] = {}
            if "full" in regions or "roi" in regions:
                legacy = display_metrics(
                    pred,
                    gt,
                    lpips_model=lpips_model,
                    roi=roi if "roi" in regions else None,
                )
                if "full" in regions:
                    aggregate.update({key: legacy[key] for key in ("psnr", "ssim", "lpips")})
                if "roi" in regions:
                    aggregate.update({key: value for key, value in legacy.items() if key.startswith("roi_")})
            if "surface" in regions:
                assert surface_mask is not None
                aggregate["surface"] = masked_display_metrics(
                    pred, gt, mask=surface_mask, lpips_model=lpips_model
                )
            if "surface-roi" in regions:
                assert surface_mask is not None and roi is not None
                x0, y0, x1, y1 = roi
                roi_mask = torch.zeros_like(surface_mask, dtype=torch.bool)
                roi_mask[y0:y1, x0:x1] = True
                aggregate["surface_roi"] = masked_display_metrics(
                    pred, gt, mask=surface_mask & roi_mask, lpips_model=lpips_model
                )
            if "reprojected" in regions:
                assert variant_masks is not None
                aggregate["reprojected"] = masked_display_metrics(
                    pred, gt, mask=variant_masks[name], lpips_model=lpips_model
                )
            if "reprojected-roi" in regions:
                assert variant_masks is not None and roi is not None
                x0, y0, x1, y1 = roi
                roi_mask = torch.zeros_like(variant_masks[name], dtype=torch.bool)
                roi_mask[y0:y1, x0:x1] = True
                aggregate["reprojected_roi"] = masked_display_metrics(
                    pred, gt, mask=variant_masks[name] & roi_mask, lpips_model=lpips_model
                )
            receipt = {
                "schema_version": 1,
                "metric_domain": "display-referred RGB clamped to [0, 1]",
                "metric_regions": regions,
                "roi_name": roi_name,
                "roi_bbox_xyxy": None if roi is None else list(roi),
                "aggregate": aggregate,
            }
            (output_dir / name / "metrics.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            row["metrics"] = aggregate
            by_variant[name] = receipt
    (output_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 1, "variants": by_variant}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return by_variant


def write_prediction(directory: Path, name: str, rgb: torch.Tensor, ground_truth: Path | None) -> None:
    variant = directory / name
    variant.mkdir(parents=True, exist_ok=False)
    array = rgb.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu()
    write_exr_image(variant / "eval_pred_0000.exr", array)
    Image.fromarray(array.mul(255.0).add(0.5).byte().numpy(), "RGB").save(
        variant / "eval_pred_0000.png", compress_level=3
    )
    if ground_truth is not None:
        (variant / "eval_gt_0000.exr").symlink_to(ground_truth)


def write_source_selection(directory: Path, name: str, selection: torch.Tensor, count: int) -> None:
    """Write a categorical map of the hard or maximum-contribution source rank."""

    palette = torch.tensor(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=torch.uint8,
        device=selection.device,
    )
    if count > len(palette):
        import colorsys

        extra = [
            [round(channel * 255) for channel in colorsys.hsv_to_rgb((rank * 0.61803398875) % 1, 0.65, 0.95)]
            for rank in range(len(palette), count)
        ]
        palette = torch.cat((palette, torch.tensor(extra, dtype=torch.uint8, device=selection.device)))
    valid = selection >= 0
    safe = selection.clamp(0, count - 1)
    rgb = palette[safe]
    rgb = torch.where(valid.unsqueeze(-1), rgb, torch.zeros_like(rgb))
    Image.fromarray(rgb.cpu().numpy(), "RGB").save(
        directory / name / "source_selection.png", compress_level=3
    )


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    manifest = json.loads(args.mesh_depth_manifest.read_text(encoding="utf-8"))
    depth_by_image = {
        resolve_manifest_path(row["image"], args.data, args.mesh_depth_manifest): resolve_manifest_path(
            row["depth"], args.data, args.mesh_depth_manifest
        )
        for row in manifest["images"]
    }
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
    train = parser_config.setup().get_dataparser_outputs(split="train")
    target = parser_config.setup().get_dataparser_outputs(split="val")
    observed_depth_by_stem = {}
    if args.source_observed_depth_data is not None:
        observed = json.loads((args.source_observed_depth_data / "transforms.json").read_text())
        observed_train = set(observed["train_filenames"])
        for frame in observed["frames"]:
            if frame["file_path"] in observed_train:
                if frame.get("mask_path"):
                    raise ValueError("Observed source depths must not use image/person masks")
                observed_depth_by_stem[Path(frame["file_path"]).stem] = frame
    if train.mask_filenames is not None or target.mask_filenames is not None:
        raise ValueError("Mesh reprojection diagnostic forbids image/person masks")
    if len(target.image_filenames) != 1:
        raise ValueError(f"Expected exactly one eval image, got {len(target.image_filenames)}")
    if max(args.neighbors) > len(train.image_filenames):
        raise ValueError("Requested more neighbors than train cameras")

    target_image = Path(target.image_filenames[0]).resolve()
    target_depth_array, target_depth_component_filter = filter_small_target_depth_components(
        load_depth(depth_by_image[target_image]),
        min_area=args.target_depth_component_min_area,
        max_log_jump=args.target_depth_component_max_log_jump,
    )
    target_depth_array, target_depth_hole_fill = fill_small_consistent_depth_holes(
        target_depth_array,
        max_area=args.depth_hole_fill_max_area,
        boundary_radius=args.depth_hole_fill_boundary_radius,
        max_relative_plane_rmse=args.depth_hole_fill_max_relative_plane_rmse,
    )
    target_depth = torch.from_numpy(target_depth_array).to(device)
    target_depth = target_depth * float(target.dataparser_scale)
    target_surface_mask = target_depth > 0.0
    metric_surface_mask = target_surface_mask
    if args.metric_surface_depth_manifest is not None:
        metric_manifest = json.loads(args.metric_surface_depth_manifest.read_text(encoding="utf-8"))
        metric_depth_by_image = {
            resolve_manifest_path(row["image"], args.data, args.metric_surface_depth_manifest): resolve_manifest_path(
                row["depth"], args.data, args.metric_surface_depth_manifest
            )
            for row in metric_manifest["images"]
        }
        if target_image not in metric_depth_by_image:
            raise ValueError(
                f"Independent metric surface manifest has no eval depth for {target_image}"
            )
        metric_surface_depth = torch.from_numpy(load_depth(metric_depth_by_image[target_image])).to(device)
        if metric_surface_depth.shape != target_depth.shape:
            raise ValueError(
                "Independent metric surface depth shape does not match rendered target: "
                f"{tuple(metric_surface_depth.shape)} vs {tuple(target_depth.shape)}"
            )
        metric_surface_mask = metric_surface_depth > 0.0
    target_c2w, target_intrinsics = camera_parameters(target.cameras, 0, device)
    train_cameras = [camera_parameters(train.cameras, index, device) for index in range(len(train.image_filenames))]
    color_calibration = None
    color_source_frames = {}
    if args.camera_color_calibration is not None:
        color_calibration = json.loads(args.camera_color_calibration.read_text())
        if (color_calibration.get("uses_eval_rgb") is not False
                or color_calibration.get("domain") != "inverse_srgb_inverse_reinhard_exposed_linear"):
            raise ValueError("Color calibration must be train-only and use the exact ingest display curve")
        source_payload = json.loads((args.data / "transforms.json").read_text())
        color_source_frames = {(args.data / f["file_path"]).resolve(): f for f in source_payload["frames"]}
    centers = torch.stack([camera[0][:, 3] for camera in train_cameras])
    distances = torch.linalg.vector_norm(centers - target_c2w[:, 3], dim=-1)
    order = torch.argsort(distances).tolist()
    if args.primary_angular_camera_count:
        from build_angular_camera_subset import select_angular_frames

        source_payload = json.loads((args.data / "transforms.json").read_text())
        source_by_path = {
            (args.data / frame["file_path"]).resolve(): frame for frame in source_payload["frames"]
        }
        source_frames = [source_by_path[Path(path).resolve()] for path in train.image_filenames]
        angular_indices, _ = select_angular_frames(source_frames, args.primary_angular_camera_count, None)
        primary = next(index for index in order if index in angular_indices)
        order = [primary] + [index for index in order if index != primary]
    max_neighbors = max(args.neighbors)
    selected = order[:max_neighbors]

    warped: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    projectable_masks: list[torch.Tensor] = []
    weighted_scores: list[torch.Tensor] = []
    best_scores: list[torch.Tensor] = []
    nearest_source_rgb: torch.Tensor | None = None
    source_rows: list[dict[str, object]] = []
    target_world = target_depth_to_world(target_depth, target_c2w, target_intrinsics)
    with torch.inference_mode():
        for rank, source_index in enumerate(selected):
            source_image = Path(train.image_filenames[source_index]).resolve()
            source_rgb = load_rgb(source_image, device)
            source_color_gain = None
            if color_calibration is not None:
                from patchmatch_color_calibration import apply_camera_gain

                physical = color_source_frames[source_image]["physical_camera"]
                calibration_row = color_calibration["cameras"][physical]
                if sha256(source_image) != calibration_row["image_sha256"]:
                    raise ValueError(f"Source RGB changed since color calibration: {physical}")
                key = {"ingest": "ingest_gain_correction", "spatial": "exposure_gain"}.get(
                    args.camera_color_model, args.camera_color_model + "_gain")
                source_color_gain = calibration_row[key]
                source_rgb = apply_camera_gain(source_rgb, source_color_gain,
                    calibration_row["spatial_log_gain_grid"] if args.camera_color_model == "spatial" else None)
            if rank == 0:
                nearest_source_rgb = source_rgb
            source_depth_array, source_depth_hole_fill = fill_small_consistent_depth_holes(
                load_depth(depth_by_image[source_image]),
                max_area=args.depth_hole_fill_max_area,
                boundary_radius=args.depth_hole_fill_boundary_radius,
                max_relative_plane_rmse=args.depth_hole_fill_max_relative_plane_rmse,
            )
            source_depth = torch.from_numpy(source_depth_array).to(device)
            source_depth = source_depth * float(train.dataparser_scale)
            source_c2w, source_intrinsics = train_cameras[source_index]
            u, v, projected_z = project_target_to_source(
                target_depth, target_c2w, target_intrinsics, source_c2w, source_intrinsics
            )
            sampled_rgb = grid_sample(source_rgb, u, v)
            sampled_depth = grid_sample(source_depth.unsqueeze(0), u, v)[0]
            in_bounds = (
                (target_depth > 0.0)
                & (projected_z > 0.0)
                & (u >= 0.0)
                & (u <= source_intrinsics["width"] - 1.0)
                & (v >= 0.0)
                & (v <= source_intrinsics["height"] - 1.0)
            )
            projectable = in_bounds
            in_bounds = in_bounds & (sampled_depth > 0.0)
            log_error = torch.abs(
                torch.log(projected_z.clamp_min(1e-6)) - torch.log(sampled_depth.clamp_min(1e-6))
            )
            valid = in_bounds & (log_error <= args.depth_log_tolerance)
            observed_veto_pixels = 0
            if args.source_observed_depth_data is not None:
                observed_frame = observed_depth_by_stem[source_image.stem]
                for saved_key, camera_key in (("fl_x", "fx"), ("fl_y", "fy"), ("cx", "cx"), ("cy", "cy")):
                    if not math.isclose(float(observed_frame[saved_key]), source_intrinsics[camera_key], rel_tol=1e-6):
                        raise ValueError("Observed depth intrinsics do not match source RGB raster")
                observed_array = load_depth(args.source_observed_depth_data / observed_frame["depth_file_path"])
                if observed_array.shape != source_depth_array.shape:
                    raise ValueError("Observed depth shape does not match source RGB raster")
                observed_depth = torch.from_numpy(observed_array).to(device) * float(train.dataparser_scale)
                sampled_observed = grid_sample(observed_depth.unsqueeze(0), u, v)[0]
                occluded = (sampled_observed > 0) & (
                    projected_z > sampled_observed * math.exp(args.source_observed_depth_log_tolerance)
                )
                observed_veto_pixels = int((valid & occluded).sum().item())
                valid = valid & ~occluded
            depth_weight = torch.exp(-0.5 * (log_error / args.depth_log_tolerance).square())
            pose_weight = 1.0 / max(
                float(distances[source_index].item()) ** args.camera_distance_power,
                1e-6,
            )
            weighted_score = valid.float() * depth_weight * pose_weight
            per_pixel_best_score = best_view_score(
                valid=valid,
                depth_confidence=depth_weight,
                target_world=target_world,
                target_camera_center=target_c2w[:, 3],
                source_camera_center=source_c2w[:, 3],
                target_depth=target_depth,
                projected_depth=projected_z,
                source_u=u,
                source_v=v,
                target_intrinsics=target_intrinsics,
                source_intrinsics=source_intrinsics,
                angle_power=args.best_view_angle_power,
                border_margin=args.best_view_border_margin,
            )
            warped.append(sampled_rgb)
            valid_masks.append(valid)
            projectable_masks.append(projectable)
            weighted_scores.append(weighted_score)
            best_scores.append(per_pixel_best_score)
            source_rows.append(
                {
                    "rank": rank,
                    "train_index": source_index,
                    "source_image": str(source_image),
                    "source_color_gain": source_color_gain,
                    "observed_depth_veto_pixels": observed_veto_pixels,
                    "camera_distance": float(distances[source_index].item()),
                    "valid_target_fraction": float(valid.float().mean().item()),
                    "valid_mesh_fraction": float(valid[target_depth > 0].float().mean().item()),
                    "median_log_depth_error_in_bounds": (
                        float(torch.median(log_error[in_bounds]).item()) if bool(in_bounds.any()) else None
                    ),
                    "median_best_view_score_valid": (
                        float(torch.median(per_pixel_best_score[valid]).item()) if bool(valid.any()) else None
                    ),
                    "depth_hole_fill": source_depth_hole_fill,
                }
            )

        overlap_exposure = None
        if args.overlap_exposure_grid is not None:
            from patchmatch_color_calibration import correct_projected_exposure

            warped, overlap_exposure = correct_projected_exposure(warped, valid_masks, *args.overlap_exposure_grid)
        if args.base_prediction_exr is None:
            base = torch.zeros_like(warped[0])
        else:
            base = load_rgb(args.base_prediction_exr, device)
            if base.shape != warped[0].shape:
                # Nerfstudio's ``eval_img`` PNG is a horizontal GT|prediction
                # review image.  Taking only the right half is deterministic
                # and avoids ever using its embedded held-out GT as input.
                if base.shape[:2] == warped[0].shape[:2] and base.shape[2] == 2 * warped[0].shape[2]:
                    base = base[:, :, warped[0].shape[2] :]
                else:
                    raise ValueError(
                        f"Base prediction shape {tuple(base.shape)} does not match target {tuple(warped[0].shape)}"
                    )
        args.output_dir.mkdir(parents=True)
        if args.write_source_warp_audit:
            audit_directory=args.output_dir/"source_warps"
            audit_directory.mkdir()
            for rank,(rgb,valid) in enumerate(zip(warped,valid_masks)):
                pixels=rgb.detach().permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
                Image.fromarray(pixels).save(audit_directory/f"source_{rank:02d}.png")
                Image.fromarray(valid.byte().mul(255).cpu().numpy()).save(audit_directory/f"valid_{rank:02d}.png")
        if args.score_metrics and args.metric_regions is not None and any(
            region.startswith("surface") for region in args.metric_regions
        ):
            Image.fromarray(metric_surface_mask.byte().mul(255).cpu().numpy(), "L").save(
                args.output_dir / "target_surface_metric_mask.png", compress_level=3
            )
        ground_truth = args.ground_truth_exr
        if ground_truth is None and not args.skip_ground_truth_copy:
            # Held-out RGB is loaded only after every prediction input has
            # already been constructed.  It is written solely for metrics.
            ground_truth = args.output_dir / "eval_gt_0000.exr"
            target_rgb = load_rgb(target_image, device)
            write_exr_image(ground_truth, target_rgb.permute(1, 2, 0).detach().float().cpu())
        variants: list[dict[str, object]] = []
        variant_metric_masks: dict[str, torch.Tensor] = {}
        assert nearest_source_rgb is not None
        write_prediction(args.output_dir, "unwarped_nearest_control", nearest_source_rgb, ground_truth)
        variants.append(
            {
                "name": "unwarped_nearest_control",
                "neighbors": 1,
                "control_without_geometry_reprojection": True,
                "valid_fraction": 1.0,
            }
        )
        variant_metric_masks["unwarped_nearest_control"] = target_surface_mask
        for count in sorted(set(args.neighbors)):
            for aggregation_mode in dict.fromkeys(args.aggregation_modes):
                blend, valid, selection = aggregate_warped_sources(
                    warped[:count],
                    valid_masks[:count],
                    weighted_scores[:count],
                    best_scores[:count],
                    mode=aggregation_mode,
                    nearest_fill_color_continuity=args.nearest_fill_color_continuity,
                    nearest_fill_color_continuity_mode=args.nearest_fill_color_continuity_mode,
                    nearest_fill_rank_penalty=args.nearest_fill_rank_penalty,
                    seam_cut_rank_penalty=args.seam_cut_rank_penalty,
                )
                continuation: dict[str, object] = {"enabled": False}
                if (
                    aggregation_mode == "nearest-fill"
                    and args.nearest_fill_primary_color_continuation
                ):
                    blend, selection, continuation = continue_primary_color_across_small_fallback_components(
                        blend,
                        selection,
                        warped[0],
                        valid_masks[0],
                        projectable_masks[0],
                        min_area=args.nearest_fill_primary_color_continuation_min_area,
                        max_area=args.nearest_fill_primary_color_continuation_max_area,
                        min_median_l1=args.nearest_fill_primary_color_continuation_min_median_l1,
                    )
                variant_stem = (
                    f"blend{count}"
                    if aggregation_mode == "weighted"
                    else f"{aggregation_mode.replace('-', '_')}{count}"
                )
                selection_fractions = source_selection_fractions(selection, count)
                for alpha in sorted(set(args.blend_alphas)):
                    prediction = torch.where(valid, base * (1.0 - alpha) + blend * alpha, base)
                    alpha_token = f"{alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                    name = variant_stem if alpha == 1.0 else f"{variant_stem}_a{alpha_token}"
                    write_prediction(args.output_dir, name, prediction, ground_truth)
                    write_source_selection(args.output_dir, name, selection, count)
                    variants.append(
                        {
                            "name": name,
                            "neighbors": count,
                            "aggregation_mode": aggregation_mode,
                            "blend_alpha": alpha,
                            "valid_fraction": float(valid[0].float().mean().item()),
                            "valid_mesh_fraction": float(valid[0][target_depth > 0.0].float().mean().item()),
                            "source_rank_fraction_on_valid": selection_fractions,
                            "primary_color_continuation": continuation,
                        }
                    )
                    variant_metric_masks[name] = valid[0]
                for sigma in sorted(set(args.detail_transfer_sigmas)):
                    for strength in sorted(set(args.detail_transfer_strengths)):
                        prediction = surface_detail_transfer(
                            base=base,
                            source=blend,
                            valid=valid,
                            sigma=sigma,
                            strength=strength,
                        )
                        sigma_token = f"{sigma:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                        strength_token = f"{strength:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                        name = f"{variant_stem}_detail_s{sigma_token}_w{strength_token}"
                        write_prediction(args.output_dir, name, prediction, ground_truth)
                        write_source_selection(args.output_dir, name, selection, count)
                        variants.append(
                            {
                                "name": name,
                                "neighbors": count,
                                "aggregation_mode": aggregation_mode,
                                "detail_transfer_sigma": sigma,
                                "detail_transfer_strength": strength,
                                "valid_fraction": float(valid[0].float().mean().item()),
                                "valid_mesh_fraction": float(valid[0][target_depth > 0.0].float().mean().item()),
                                "source_rank_fraction_on_valid": selection_fractions,
                            }
                        )
                        variant_metric_masks[name] = valid[0]

    metrics = (
        score_written_variants(
            args.output_dir,
            variants,
            ground_truth,
            device=device,
            roi_boxes_json=args.roi_boxes_json,
            surface_mask=metric_surface_mask,
            metric_regions=args.metric_regions,
            variant_masks=variant_metric_masks,
        )
        if args.score_metrics
        else None
    )

    audit = {
        "schema_version": 1,
        "method": "continuous_mesh_calibrated_train_image_reprojection",
        "uses_eval_rgb_for_prediction": False,
        "eval_rgb_use": "not_read" if args.skip_ground_truth_copy else "metrics_only",
        "uses_masks": False,
        "base_prediction": (
            None
            if args.base_prediction_exr is None
            else {
                "path": str(args.base_prediction_exr),
                "sha256": sha256(args.base_prediction_exr),
                "input_policy": "prediction_only_or_right_half_of_nerfstudio_gt_prediction_review",
            }
        ),
        "ground_truth": None if args.skip_ground_truth_copy else {
            "path": str(target_image if args.ground_truth_exr is None else args.ground_truth_exr),
            "use": "metrics_only",
        },
        "data": str(args.data),
        "mesh_depth_manifest": str(args.mesh_depth_manifest),
        "mesh_depth_manifest_sha256": sha256(args.mesh_depth_manifest),
        "metric_surface_depth_manifest": (
            None if args.metric_surface_depth_manifest is None else str(args.metric_surface_depth_manifest)
        ),
        "metric_surface_depth_manifest_sha256": (
            None
            if args.metric_surface_depth_manifest is None
            else sha256(args.metric_surface_depth_manifest)
        ),
        "target_image": str(target_image),
        "depth_log_tolerance": args.depth_log_tolerance,
        "depth_hole_fill": {
            "max_area": args.depth_hole_fill_max_area,
            "boundary_radius": args.depth_hole_fill_boundary_radius,
            "max_relative_plane_rmse": args.depth_hole_fill_max_relative_plane_rmse,
            "target": target_depth_hole_fill,
        },
        "target_depth_component_filter": target_depth_component_filter,
        "nearest_fill_color_continuity": args.nearest_fill_color_continuity,
        "nearest_fill_color_continuity_mode": args.nearest_fill_color_continuity_mode,
        "nearest_fill_rank_penalty": args.nearest_fill_rank_penalty,
        "nearest_fill_primary_color_continuation": {
            "enabled": args.nearest_fill_primary_color_continuation,
            "min_area": args.nearest_fill_primary_color_continuation_min_area,
            "max_area": args.nearest_fill_primary_color_continuation_max_area,
            "min_median_l1": args.nearest_fill_primary_color_continuation_min_median_l1,
        },
        "camera_distance_power": args.camera_distance_power,
        "overlap_exposure_calibration": overlap_exposure,
        "camera_color_calibration": None if color_calibration is None else {
            "path": str(args.camera_color_calibration.resolve()),
            "sha256": sha256(args.camera_color_calibration),
            "helper_sha256": sha256(Path(__file__).with_name("patchmatch_color_calibration.py")),
            "model": args.camera_color_model,
            "domain": color_calibration["domain"],
            "gauge": color_calibration["gauge"],
            "uses_eval_rgb": False,
            "source_averaging": False,
        },
        "primary_angular_camera_count": args.primary_angular_camera_count,
        "seam_cut": {
            "enabled": "seam-cut" in args.aggregation_modes,
            "rank_penalty": args.seam_cut_rank_penalty,
            "helper_sha256": sha256(Path(__file__).with_name("hard_texture_seam_cut.py"))
            if "seam-cut" in args.aggregation_modes else None,
            "color_averaging": False,
        },
        "aggregation_modes": list(args.aggregation_modes),
        "best_view_angle_power": args.best_view_angle_power,
        "best_view_border_margin": args.best_view_border_margin,
        "detail_transfer_sigmas": list(args.detail_transfer_sigmas),
        "detail_transfer_strengths": list(args.detail_transfer_strengths),
        "metric_regions": args.metric_regions,
        "target_surface_fraction": float(target_surface_mask.float().mean().item()),
        "metric_surface_fraction": float(metric_surface_mask.float().mean().item()),
        "dataparser_scale": float(target.dataparser_scale),
        "sources": source_rows,
        "variants": variants,
        "metrics": metrics,
    }
    (args.output_dir / "reprojection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
