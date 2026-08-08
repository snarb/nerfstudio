#!/usr/bin/env python3
"""Build scene-calibrated LookCloser frequency maps directly from linear EXR.

The progressive 2D regressor is trained once per image.  Its complete PQ-SSIM
recovery cube is then reused by three automatic assignment families:
scene-calibrated absolute crossing, relative multi-crossing, and threshold-free
knee selection.  The historical fixed 0.95/0.97 thresholds are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nerfstudio.scripts.lookcloser_preprocess import InstantNGP2D, compute_patch_starts, compute_ssim
from nerfstudio.utils.hdr import (
    BT709_LUMA,
    HDRCalibration,
    activate_hdr_rgb,
    calibrate_exr_paths,
    hdr_display_preview,
    scene_linear_to_pq,
)
from nerfstudio.utils.lookcloser_frequency import (
    FrequencyMapQuality,
    bootstrap_select,
    first_crossing_levels,
    guided_median_levels,
    knee_levels,
    levels_to_resolutions,
    map_quality,
    relative_ensemble_levels,
)

MIN_RES = 16
N_LEVELS = 16
N_FEATURES = 2
LOG2_HASHMAP_SIZE = 23


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--glob", default="frame_train_*.exr")
    parser.add_argument("--steps-per-level", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--eval-patch-batch", type=int, default=8192)
    parser.add_argument("--max-res", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--ssim-window", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--loss",
        choices=("linear_l1", "rawnerf_weighted_l2", "linear_pq", "pq_l1", "eag_pq_dssim"),
        default="linear_pq",
    )
    parser.add_argument("--softplus-beta", type=float, default=1.0)
    parser.add_argument("--pq-code-temperature", type=float, default=1.0)
    parser.add_argument("--rawnerf-epsilon", type=float, default=1e-3)
    parser.add_argument("--linear-anchor-weight", type=float, default=0.0)
    parser.add_argument("--eag-dssim-weight", type=float, default=0.2)
    parser.add_argument("--eag-patch-size", type=int, default=11)
    parser.add_argument("--proxy-sobel-weight", type=float, default=0.5)
    parser.add_argument(
        "--proxy-domain",
        choices=("pq", "linear"),
        default="pq",
        help="Domain for the independent edge/high-pass selector proxy.",
    )
    parser.add_argument("--relative-half-width", type=float, default=0.15)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visual-count", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.steps_per_level <= 0 or args.train_batch_size <= 0 or args.eval_patch_batch <= 0:
        parser.error("training/evaluation counts must be positive")
    if args.patch_size <= 0 or args.max_res <= MIN_RES:
        parser.error("expected patch_size > 0 and max_res > min_res")
    if args.ssim_window <= 0 or args.ssim_window % 2 == 0:
        parser.error("ssim-window must be a positive odd integer")
    if not 0 <= args.proxy_sobel_weight <= 1:
        parser.error("proxy-sobel-weight must be in [0, 1]")
    if not 0 < args.relative_half_width < 0.5:
        parser.error("relative-half-width must be in (0, 0.5)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_linear_exr(path: Path, device: torch.device) -> torch.Tensor:
    from nerfstudio.data.utils.data_utils import load_exr_image

    image = load_exr_image(path)
    if image.shape[-1] == 4:
        alpha = image[..., 3:4]
        image = image[..., :3] * alpha
    return torch.from_numpy(image[..., :3]).to(device=device, dtype=torch.float32)


def precompute_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    height, width, _ = image.shape
    h_steps = len(compute_patch_starts(height, patch_size, patch_size))
    w_steps = len(compute_patch_starts(width, patch_size, patch_size))
    patches = F.unfold(image.permute(2, 0, 1).unsqueeze(0), kernel_size=patch_size, stride=patch_size)
    if patches.shape[-1] != h_steps * w_steps:
        raise AssertionError("Patch extraction shape differs from LookCloser patch-start contract")
    return patches.view(3, patch_size, patch_size, h_steps, w_steps).permute(3, 4, 0, 1, 2).contiguous()


def make_patch_uv(xs: torch.Tensor, ys: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    local = torch.arange(patch_size, device=xs.device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(local, local, indexing="ij")
    x = xx[None] + xs.float()[:, None, None]
    y = yy[None] + ys.float()[:, None, None]
    return torch.stack((x / float(width), y / float(height)), dim=-1).reshape(-1, 2)


def activate_prediction(raw: torch.Tensor, args: argparse.Namespace, calibration: HDRCalibration) -> torch.Tensor:
    parameterization = "pq_code" if args.loss == "pq_l1" else "linear_softplus"
    return activate_hdr_rgb(
        raw,
        parameterization=parameterization,
        linear_scale=calibration.linear_scale,
        initial_radiance=calibration.initial_radiance,
        nits_per_scene_unit=calibration.nits_per_scene_unit,
        black_nits=calibration.black_nits,
        peak_nits=calibration.peak_nits,
        softplus_beta=args.softplus_beta,
        pq_code_temperature=args.pq_code_temperature,
    )


def pointwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    calibration: HDRCalibration,
) -> torch.Tensor:
    valid = torch.isfinite(target)
    target = torch.where(valid, target, torch.zeros_like(target))
    normalized_prediction = prediction / calibration.linear_scale
    normalized_target = target / calibration.linear_scale
    linear_l1 = (normalized_prediction - normalized_target).abs()[valid].mean()
    if args.loss == "linear_l1":
        return linear_l1
    if args.loss == "rawnerf_weighted_l2":
        denominator = normalized_prediction.detach().clamp_min(0.0) + args.rawnerf_epsilon
        return (((normalized_prediction - normalized_target) / denominator).square())[valid].mean()
    pq_prediction = scene_linear_to_pq(
        prediction,
        nits_per_scene_unit=calibration.nits_per_scene_unit,
        black_nits=calibration.black_nits,
    )
    pq_target = scene_linear_to_pq(
        target,
        nits_per_scene_unit=calibration.nits_per_scene_unit,
        black_nits=calibration.black_nits,
    )
    return (pq_prediction - pq_target).abs()[valid].mean() + args.linear_anchor_weight * linear_l1


def train_and_measure_recovery(
    image: torch.Tensor,
    args: argparse.Namespace,
    calibration: HDRCalibration,
) -> torch.Tensor:
    """Train one progressive 2D HashGrid and return (levels,Hpatch,Wpatch) PQ-SSIM."""

    if image.device.type != "cuda":
        raise RuntimeError("Frequency-map regression requires CUDA/tiny-cuda-nn")
    height, width, _ = image.shape
    y_starts = compute_patch_starts(height, args.patch_size, args.patch_size)
    x_starts = compute_patch_starts(width, args.patch_size, args.patch_size)
    h_steps, w_steps = len(y_starts), len(x_starts)
    gt_patches = precompute_patches(image, args.patch_size)
    gt_pq = scene_linear_to_pq(
        gt_patches,
        nits_per_scene_unit=calibration.nits_per_scene_unit,
        black_nits=calibration.black_nits,
    )
    ys_grid = torch.tensor(y_starts, device=image.device, dtype=torch.long)
    xs_grid = torch.tensor(x_starts, device=image.device, dtype=torch.long)

    model = InstantNGP2D(
        N_LEVELS,
        N_FEATURES,
        MIN_RES,
        args.max_res,
        LOG2_HASHMAP_SIZE,
        output_activation="None",
    ).to(image.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-15)
    recovery = torch.empty((N_LEVELS, h_steps, w_steps), dtype=torch.float32, device="cpu")

    for level in range(N_LEVELS):
        model.train()
        for _ in range(args.steps_per_level):
            optimizer.zero_grad(set_to_none=True)
            if args.loss == "eag_pq_dssim":
                patch_size = args.eag_patch_size
                patch_count = max(1, args.train_batch_size // (patch_size * patch_size))
                ys = torch.randint(0, max(1, height - patch_size + 1), (patch_count,), device=image.device)
                xs = torch.randint(0, max(1, width - patch_size + 1), (patch_count,), device=image.device)
                uv = make_patch_uv(xs, ys, height, width, patch_size)
                raw = model.render_masked(uv, level)
                prediction = activate_prediction(raw, args, calibration)
                local = torch.arange(patch_size, device=image.device)
                yy, xx = torch.meshgrid(local, local, indexing="ij")
                target = image[(ys[:, None, None] + yy).reshape(-1), (xs[:, None, None] + xx).reshape(-1)]
                pq_prediction = scene_linear_to_pq(
                    prediction,
                    nits_per_scene_unit=calibration.nits_per_scene_unit,
                    black_nits=calibration.black_nits,
                ).reshape(patch_count, patch_size, patch_size, 3).permute(0, 3, 1, 2)
                pq_target = scene_linear_to_pq(
                    target,
                    nits_per_scene_unit=calibration.nits_per_scene_unit,
                    black_nits=calibration.black_nits,
                ).reshape(patch_count, patch_size, patch_size, 3).permute(0, 3, 1, 2)
                pq_l1 = (pq_prediction - pq_target).abs().mean()
                dssim = 1.0 - compute_ssim(
                    pq_prediction.float(), pq_target.float(), window_size=args.ssim_window, size_average=True
                )
                loss = (1.0 - args.eag_dssim_weight) * pq_l1 + args.eag_dssim_weight * dssim
            else:
                ys = torch.randint(0, height, (args.train_batch_size,), device=image.device)
                xs = torch.randint(0, width, (args.train_batch_size,), device=image.device)
                uv = torch.stack(((xs.float() + 0.5) / width, (ys.float() + 0.5) / height), dim=-1)
                prediction = activate_prediction(model.render_masked(uv, level), args, calibration)
                loss = pointwise_loss(prediction, image[ys, xs], args, calibration)
            loss.backward()
            if args.loss == "rawnerf_weighted_l2":
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            flat_scores = torch.empty(h_steps * w_steps, device=image.device, dtype=torch.float32)
            flat_indices = torch.arange(h_steps * w_steps, device=image.device)
            for start in range(0, flat_indices.numel(), args.eval_patch_batch):
                indices = flat_indices[start : start + args.eval_patch_batch]
                iy = torch.div(indices, w_steps, rounding_mode="floor")
                ix = indices % w_steps
                uv = make_patch_uv(xs_grid[ix], ys_grid[iy], height, width, args.patch_size)
                raw = model.render_masked(uv, level)
                prediction = activate_prediction(raw, args, calibration)
                prediction = prediction.reshape(-1, args.patch_size, args.patch_size, 3).permute(0, 3, 1, 2)
                prediction_pq = scene_linear_to_pq(
                    prediction,
                    nits_per_scene_unit=calibration.nits_per_scene_unit,
                    black_nits=calibration.black_nits,
                )
                scores = compute_ssim(
                    gt_pq[iy, ix].float(), prediction_pq.float(), window_size=args.ssim_window, size_average=False
                )
                flat_scores[indices] = scores
            recovery[level] = flat_scores.reshape(h_steps, w_steps).cpu()
        print(f"    level={level:02d}/{N_LEVELS - 1} pq_ssim={float(recovery[level].mean()):.5f}", flush=True)

    del model, optimizer
    return recovery


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, flat.numel(), device=flat.device)
    return ranks.reshape_as(values)


def structural_proxy(image: torch.Tensor, patch_size: int, sobel_weight: float) -> torch.Tensor:
    """Rank-normalized Sobel plus local high-pass RMS at frequency-map grain."""

    luminance = torch.tensordot(
        image[..., :3], torch.tensor(BT709_LUMA, device=image.device, dtype=image.dtype), dims=([-1], [0])
    )[None, None]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=image.dtype)[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(luminance, sobel_x, padding=1)
    gy = F.conv2d(luminance, sobel_y, padding=1)
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-12)
    local_mean = F.avg_pool2d(luminance, kernel_size=5, stride=1, padding=2)
    highpass = (luminance - local_mean).square().sqrt()
    gradient_patches = F.avg_pool2d(gradient, kernel_size=patch_size, stride=patch_size)[0, 0]
    highpass_patches = F.avg_pool2d(highpass, kernel_size=patch_size, stride=patch_size)[0, 0]
    return sobel_weight * _rank_normalize(gradient_patches) + (1.0 - sobel_weight) * _rank_normalize(highpass_patches)


def candidate_maps(
    recovery_by_image: Mapping[str, torch.Tensor],
    relative_half_width: float,
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Generate every automatic candidate map without any legacy threshold."""

    scene_budget = 200_000
    per_image_budget = max(1, scene_budget // len(recovery_by_image))
    sampled_parts = []
    for cube in recovery_by_image.values():
        flat = cube.flatten()
        stride = max(1, math.ceil(flat.numel() / per_image_budget))
        sampled_parts.append(flat[::stride][:per_image_budget])
    sampled = torch.cat(sampled_parts)
    thresholds = torch.quantile(sampled, torch.linspace(0.1, 0.9, 9)).tolist()
    output: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, cube in recovery_by_image.items():
        per_image: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for index, threshold in enumerate(thresholds, 1):
            per_image[f"calibrated_q{index}0_t{float(threshold):.8f}"] = first_crossing_levels(
                cube, float(threshold)
            )
        for center in np.linspace(0.2, 0.8, 7):
            per_image[f"relative_{center:.2f}"] = relative_ensemble_levels(
                cube, center=float(center), half_width=relative_half_width
            )
        per_image["knee_max_distance"] = knee_levels(cube)
        output[name] = per_image
    return output


def select_candidates(
    candidates: Mapping[str, Mapping[str, Tuple[torch.Tensor, torch.Tensor]]],
    proxies: Mapping[str, torch.Tensor],
    resamples: int,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, FrequencyMapQuality]]]:
    quality_by_image: Dict[str, Dict[str, FrequencyMapQuality]] = {}
    names = sorted(candidates)
    candidate_names = sorted(next(iter(candidates.values())))
    for image_name in names:
        quality_by_image[image_name] = {}
        for candidate_name, (levels, unresolved) in candidates[image_name].items():
            quality_by_image[image_name][candidate_name] = map_quality(
                levels, proxies[image_name], n_levels=N_LEVELS, unresolved=unresolved
            )

    groups = {
        "calibrated": [name for name in candidate_names if name.startswith("calibrated_")],
        "relative": [name for name in candidate_names if name.startswith("relative_")],
        "knee": [name for name in candidate_names if name.startswith("knee_")],
    }
    winners: Dict[str, str] = {}
    reports: Dict[str, Any] = {}
    for method, group_names in groups.items():
        qualities = {
            candidate_name: [quality_by_image[image_name][candidate_name] for image_name in names]
            for candidate_name in group_names
        }
        admissible = {
            candidate_name: rows
            for candidate_name, rows in qualities.items()
            if np.mean([row.nonempty_bins for row in rows]) >= 4
            and np.mean([row.top2_bin_fraction for row in rows]) <= 0.85
        }
        if not admissible:
            admissible = qualities
        report = bootstrap_select(admissible, resamples=resamples, seed=seed)
        winners[method] = str(report["winner"])
        reports[method] = report

    method_qualities = {
        method: [quality_by_image[image_name][candidate_name] for image_name in names]
        for method, candidate_name in winners.items()
    }
    globally_admissible = {
        method: rows
        for method, rows in method_qualities.items()
        if np.mean([row.spearman for row in rows]) >= 0.1
        and np.mean([row.detail_overlap for row in rows]) >= 0.25
    }
    if not globally_admissible:
        globally_admissible = method_qualities
    global_report = bootstrap_select(globally_admissible, resamples=resamples, seed=seed + 1)
    global_report["admissible"] = sorted(globally_admissible)
    global_report["excluded"] = sorted(set(method_qualities) - set(globally_admissible))
    global_report["structural_gates"] = {"mean_spearman_min": 0.1, "mean_detail_overlap_min": 0.25}
    reports["global"] = global_report
    winners["global"] = str(global_report["winner"])
    return winners, reports, quality_by_image


def save_map(
    output_dir: Path,
    image_path: Path,
    levels: torch.Tensor,
    quality: FrequencyMapQuality,
    candidate_name: str,
    recovery_path: Path,
    calibration: HDRCalibration,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / f"{image_path.stem}.pt"
    frequency_map = levels_to_resolutions(levels, MIN_RES, args.max_res, N_LEVELS).cpu()
    torch.save(frequency_map, map_path)
    per_level_scale = float(np.exp((np.log(args.max_res) - np.log(MIN_RES)) / (N_LEVELS - 1)))
    sidecar = {
        "schema_version": 2,
        "image": image_path.name,
        "image_sha256": sha256_file(image_path),
        "recovery_cube_sha256": sha256_file(recovery_path),
        "map_sha256": sha256_file(map_path),
        "image_shape": [int(levels.shape[0] * args.patch_size), int(levels.shape[1] * args.patch_size)],
        "value_type": "scalar_resolution",
        "patch_size": args.patch_size,
        "stride": args.patch_size,
        "min_res": MIN_RES,
        "max_res": args.max_res,
        "n_levels": N_LEVELS,
        "n_features": N_FEATURES,
        "log2_hashmap_size": LOG2_HASHMAP_SIZE,
        "per_level_scale": per_level_scale,
        "level_resolution_schedule": [float(MIN_RES * per_level_scale**level) for level in range(N_LEVELS)],
        "method_candidate": candidate_name,
        "quality": quality.as_dict(),
        "hdr_calibration": calibration.as_metadata(),
        "loss": args.loss,
        "rgb_output_parameterization": "pq_code" if args.loss == "pq_l1" else "linear_softplus",
    }
    atomic_json(output_dir / f"{image_path.stem}.json", sidecar)


def save_visual(path: Path, image: torch.Tensor, levels: torch.Tensor, calibration: HDRCalibration) -> None:
    import matplotlib

    exposure = math.log2(0.18 / max(calibration.initial_radiance, 1e-8))
    preview = hdr_display_preview(image.cpu(), exposure_ev=exposure).numpy()
    heat = matplotlib.colormaps["turbo"]((levels.float().numpy() / max(N_LEVELS - 1, 1)))[..., :3]
    heat_image = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).resize(
        (preview.shape[1], preview.shape[0]), resample=Image.Resampling.NEAREST
    )
    preview_image = Image.fromarray(np.uint8(np.clip(preview, 0, 1) * 255))
    canvas = Image.new("RGB", (preview.shape[1] * 2, preview.shape[0]))
    canvas.paste(preview_image, (0, 0))
    canvas.paste(heat_image, (preview.shape[1], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((1600, 900), resample=Image.Resampling.LANCZOS)
    canvas.save(path)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    paths = sorted(args.images_dir.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No EXR images matched {args.images_dir / args.glob}")
    if any(path.suffix.lower() != ".exr" for path in paths):
        raise ValueError("This builder accepts only EXR inputs")
    args.out.mkdir(parents=True, exist_ok=True)
    recovery_dir = args.out / "recovery"
    recovery_dir.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    calibration = calibrate_exr_paths(paths)
    recovery_by_image: Dict[str, torch.Tensor] = {}
    proxy_by_image: Dict[str, torch.Tensor] = {}
    started = time.monotonic()

    for index, path in enumerate(paths, 1):
        recovery_path = recovery_dir / f"{path.stem}.pt"
        proxy_path = recovery_dir / f"{path.stem}.proxy_{args.proxy_domain}.pt"
        image = load_linear_exr(path, device)
        if recovery_path.exists() and not args.force:
            recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
            recovery_status = "existing"
        else:
            print(f"  [{index}/{len(paths)}] {path.name}", flush=True)
            recovery = train_and_measure_recovery(image, args, calibration)
            torch.save(recovery, recovery_path)
            recovery_status = "generated"
        if proxy_path.exists() and not args.force:
            proxy = torch.load(proxy_path, map_location="cpu", weights_only=True)
            proxy_status = "existing"
        else:
            proxy_image = image
            if args.proxy_domain == "pq":
                proxy_image = scene_linear_to_pq(
                    image,
                    nits_per_scene_unit=calibration.nits_per_scene_unit,
                    black_nits=calibration.black_nits,
                )
            proxy = structural_proxy(proxy_image, args.patch_size, args.proxy_sobel_weight).cpu()
            torch.save(proxy, proxy_path)
            proxy_status = "generated"
        recovery_by_image[path.name] = recovery
        proxy_by_image[path.name] = proxy
        del image
        torch.cuda.empty_cache()
        print(
            f"  [{index}/{len(paths)}] {path.stem} recovery={recovery_status} proxy={proxy_status}",
            flush=True,
        )

    candidates = candidate_maps(recovery_by_image, args.relative_half_width)
    for image_name, image_candidates in candidates.items():
        proxy = proxy_by_image[image_name]
        regularized = {
            f"{candidate_name}_guided3": (guided_median_levels(levels, proxy), unresolved)
            for candidate_name, (levels, unresolved) in image_candidates.items()
        }
        image_candidates.update(regularized)
    winners, selection_reports, quality_by_image = select_candidates(
        candidates, proxy_by_image, args.bootstrap_resamples, args.seed
    )
    path_lookup = {path.name: path for path in paths}
    for method in ("calibrated", "relative", "knee"):
        candidate_name = winners[method]
        for image_name in sorted(recovery_by_image):
            levels, _ = candidates[image_name][candidate_name]
            save_map(
                args.out / method,
                path_lookup[image_name],
                levels,
                quality_by_image[image_name][candidate_name],
                candidate_name,
                recovery_dir / f"{Path(image_name).stem}.pt",
                calibration,
                args,
            )
    selected_method = winners["global"]
    selected_candidate = winners[selected_method]
    for image_name in sorted(recovery_by_image):
        levels, _ = candidates[image_name][selected_candidate]
        save_map(
            args.out / "selected",
            path_lookup[image_name],
            levels,
            quality_by_image[image_name][selected_candidate],
            selected_candidate,
            recovery_dir / f"{Path(image_name).stem}.pt",
            calibration,
            args,
        )

    if args.visual_count > 0:
        detail_rows = sorted(
            (
                float(candidates[name][selected_candidate][0].float().mean()),
                name,
            )
            for name in proxy_by_image
        )
        picks = [detail_rows[0][1], detail_rows[len(detail_rows) // 2][1], detail_rows[-1][1]]
        frame19 = next((name for name in recovery_by_image if "frame_train_00019" in name), None)
        if frame19 is not None:
            picks.append(frame19)
        for image_name in list(dict.fromkeys(picks))[: args.visual_count]:
            image = load_linear_exr(path_lookup[image_name], torch.device("cpu"))
            for method in ("calibrated", "relative", "knee"):
                levels, _ = candidates[image_name][winners[method]]
                save_visual(
                    args.out / "visuals" / f"{Path(image_name).stem}_{method}.png",
                    image,
                    levels,
                    calibration,
                )

    provenance = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha256_file(Path(__file__)),
        "input_dir": str(args.images_dir.resolve()),
        "output_dir": str(args.out.resolve()),
        "image_count": len(paths),
        "hdr_calibration": calibration.as_metadata(),
        "parameters": vars(args) | {"images_dir": str(args.images_dir), "out": str(args.out)},
        "method_winners": winners,
        "selection": selection_reports,
        "selected_method": selected_method,
        "selected_candidate": selected_candidate,
        "total_seconds": time.monotonic() - started,
    }
    atomic_json(args.out / "provenance.json", provenance)
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    provenance = build(parse_args(argv))
    print(
        f"DONE method={provenance['selected_method']} candidate={provenance['selected_candidate']} "
        f"images={provenance['image_count']} seconds={provenance['total_seconds']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
