#!/usr/bin/env python3
"""Render a thin surface-depth teacher from a trained Splatfacto checkpoint.

Nerfstudio's Splatfacto ``depth`` output is expected depth.  Expected depth can
lie between several translucent Gaussian layers and is therefore a poor target
for forcing a radiance field to form a surface.  This tool instead follows the
same front-to-back alpha compositing order as gsplat and stores the depth of the
first Gaussian for which accumulated opacity reaches a requested quantile
(0.5 by default).

The selection dataset controls which named cameras are rendered.  It is never
used to alter RGB, and datasets declaring masks are deliberately rejected.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from gsplat import rasterization, rasterize_to_indices_in_range

from nerfstudio.models.splatfacto import SplatfactoModel, get_viewmat
from nerfstudio.utils.eval_utils import eval_setup


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Splatfacto config.yml")
    parser.add_argument(
        "--selection-dataset",
        type=Path,
        required=True,
        help="Nerfstudio dataset whose explicit train_filenames select camera names",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-split",
        choices=("train", "val", "test"),
        default="train",
        help="Explicit filename list in the selection dataset to render.",
    )
    parser.add_argument("--alpha-quantile", type=float, default=0.5)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--compression-level", type=int, default=3)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_alpha_quantile_indices(
    pixel_ids: torch.Tensor,
    alphas: torch.Tensor,
    *,
    num_pixels: int,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first front-to-back intersection crossing an alpha quantile.

    ``rasterize_to_indices_in_range`` emits intersections grouped by pixel and
    ordered from near to far.  The returned index addresses that intersection
    list; ``valid`` is false when a ray never reaches the requested opacity.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if pixel_ids.ndim != 1 or alphas.ndim != 1 or pixel_ids.shape != alphas.shape:
        raise ValueError("pixel_ids and alphas must be same-shaped vectors")
    if num_pixels <= 0:
        raise ValueError("num_pixels must be positive")
    if pixel_ids.numel() == 0:
        selected = torch.zeros((num_pixels,), device=pixel_ids.device, dtype=torch.long)
        return selected, torch.zeros((num_pixels,), device=pixel_ids.device, dtype=torch.bool)
    if bool(((pixel_ids < 0) | (pixel_ids >= num_pixels)).any()):
        raise ValueError("pixel id outside image")
    if pixel_ids.numel() > 1 and bool((pixel_ids[1:] < pixel_ids[:-1]).any()):
        raise ValueError("intersections must be grouped in nondecreasing pixel order")

    safe_alpha = alphas.float().clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
    cumulative_log_transmittance = torch.cumsum(torch.log1p(-safe_alpha), dim=0)
    counts = torch.bincount(pixel_ids, minlength=num_pixels)
    starts = torch.cumsum(counts, dim=0) - counts
    prefix = torch.zeros((num_pixels,), device=alphas.device, dtype=torch.float32)
    has_prefix = starts > 0
    prefix[has_prefix] = cumulative_log_transmittance[starts[has_prefix] - 1]
    ray_log_transmittance = cumulative_log_transmittance - torch.repeat_interleave(prefix, counts)
    crossed = ray_log_transmittance <= float(np.log1p(-quantile))

    positions = torch.arange(pixel_ids.numel(), device=pixel_ids.device, dtype=torch.long)
    missing = pixel_ids.numel()
    selected = torch.full((num_pixels,), missing, device=pixel_ids.device, dtype=torch.long)
    selected.scatter_reduce_(
        0,
        pixel_ids[crossed],
        positions[crossed],
        reduce="amin",
        include_self=True,
    )
    valid = selected < missing
    return selected, valid


def alpha_median_depth(
    info: dict[str, torch.Tensor | int],
    *,
    width: int,
    height: int,
    quantile: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Extract quantile surface depth from one unpacked gsplat rasterization."""

    device = info["means2d"].device  # type: ignore[union-attr]
    gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
        0,
        2_000_000_000,
        torch.ones((1, height, width), device=device, dtype=torch.float32),
        info["means2d"],  # type: ignore[arg-type]
        info["conics"],  # type: ignore[arg-type]
        info["opacities"],  # type: ignore[arg-type]
        width,
        height,
        int(info["tile_size"]),
        info["isect_offsets"],  # type: ignore[arg-type]
        info["flatten_ids"],  # type: ignore[arg-type]
    )
    if pixel_ids.numel() and bool((pixel_ids[1:] < pixel_ids[:-1]).any()):
        raise RuntimeError("gsplat returned intersections outside documented pixel grouping")

    means2d = info["means2d"][0, gaussian_ids]  # type: ignore[index]
    conics = info["conics"][0, gaussian_ids]  # type: ignore[index]
    opacities = info["opacities"][0, gaussian_ids]  # type: ignore[index]
    pixel_x = torch.remainder(pixel_ids, width).float() + 0.5
    pixel_y = torch.div(pixel_ids, width, rounding_mode="floor").float() + 0.5
    delta_x = means2d[:, 0] - pixel_x
    delta_y = means2d[:, 1] - pixel_y
    sigma = (
        0.5 * (conics[:, 0] * delta_x.square() + conics[:, 2] * delta_y.square())
        + conics[:, 1] * delta_x * delta_y
    )
    alphas = torch.minimum(
        torch.tensor(0.999, device=device),
        opacities * torch.exp(-sigma),
    )
    selected, valid = first_alpha_quantile_indices(
        pixel_ids,
        alphas,
        num_pixels=height * width,
        quantile=quantile,
    )
    depth = torch.zeros((height * width,), device=device, dtype=torch.float32)
    depths = info["depths"]  # type: ignore[assignment]
    depth[valid] = depths[0, gaussian_ids[selected[valid]]]  # type: ignore[index]
    result = depth.reshape(height, width)
    stats: dict[str, float | int] = {
        "intersection_count": int(pixel_ids.numel()),
        "valid_pixel_count": int(valid.sum()),
        "valid_pixel_fraction": float(valid.float().mean()),
    }
    if bool(valid.any()):
        values = result[result > 0]
        stats.update(
            depth_min_normalized=float(values.min()),
            depth_median_normalized=float(values.median()),
            depth_max_normalized=float(values.max()),
        )
    return result, stats


def inherited(frame: dict[str, Any], payload: dict[str, Any], key: str) -> int:
    value = frame.get(key, payload.get(key))
    if value is None:
        raise ValueError(f"Missing {key!r} for {frame.get('file_path')!r}")
    return int(value)


def save_depth(path: Path, depth: np.ndarray, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=compression_level) as stream:
        np.save(stream, np.asarray(depth, dtype=np.float32), allow_pickle=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.alpha_quantile < 1.0:
        raise ValueError("--alpha-quantile must be strictly between zero and one")
    if not 0.0 < args.resolution_scale <= 1.0:
        raise ValueError("--resolution-scale must be in (0, 1]")
    if not 0 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be in [0, 9]")
    config_path = args.config.expanduser().resolve()
    selection_root = args.selection_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    selection = json.loads((selection_root / "transforms.json").read_text(encoding="utf-8"))
    frames = selection.get("frames")
    selection_key = f"{args.selection_split}_filenames"
    declared_train = selection.get(selection_key)
    if not isinstance(frames, list) or not frames:
        raise ValueError("Selection dataset has no frames")
    if not isinstance(declared_train, list) or not declared_train:
        raise ValueError(f"Selection dataset must declare non-empty {selection_key}")
    if any(frame.get("mask_path") for frame in frames):
        raise ValueError("Alpha-median depth generation forbids person/foreground masks")
    frame_by_name = {Path(str(frame["file_path"])).name: frame for frame in frames}
    selected_names = [Path(str(path)).name for path in declared_train]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Selected train image names are not unique")
    unknown = sorted(set(selected_names) - set(frame_by_name))
    if unknown:
        raise ValueError(f"Unknown selected frames: {unknown[:8]}")

    _, pipeline, checkpoint_path, step = eval_setup(config_path, test_mode="inference")
    model = pipeline.model
    if not isinstance(model, SplatfactoModel):
        raise TypeError(f"Expected SplatfactoModel, got {type(model).__name__}")
    datasets = [pipeline.datamanager.train_dataset, pipeline.datamanager.eval_dataset]
    camera_by_name: dict[str, Any] = {}
    for dataset in datasets:
        if dataset is None:
            continue
        for index, path in enumerate(dataset._dataparser_outputs.image_filenames):
            camera_by_name.setdefault(path.name, dataset.cameras[index : index + 1])
    missing = sorted(set(selected_names) - set(camera_by_name))
    if missing:
        raise ValueError(f"Checkpoint dataset lacks selected cameras: {missing[:8]}")

    parser_scale = float(pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale)
    colors = torch.zeros((len(model.means), 3), device=model.device, dtype=torch.float32)
    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rows: list[dict[str, Any]] = []
    try:
        raw_dir = stage / args.selection_split / "raw-alpha-median-depth"
        raw_dir.mkdir(parents=True)
        for ordinal, name in enumerate(selected_names, 1):
            camera = camera_by_name[name].to(model.device)
            source_frame = frame_by_name[name]
            source_width = inherited(source_frame, selection, "w")
            source_height = inherited(source_frame, selection, "h")
            camera_width = int(camera.width.item())
            camera_height = int(camera.height.item())
            if (camera_width, camera_height) != (source_width, source_height):
                raise ValueError(
                    f"Camera/image size mismatch for {name}: checkpoint={(camera_width, camera_height)} "
                    f"selection={(source_width, source_height)}"
                )
            camera.rescale_output_resolution(args.resolution_scale)
            width, height = int(camera.width.item()), int(camera.height.item())
            with torch.inference_mode():
                _, _, info = rasterization(
                    means=model.means,
                    quats=model.quats,
                    scales=torch.exp(model.scales),
                    opacities=torch.sigmoid(model.opacities).squeeze(-1),
                    colors=colors,
                    viewmats=get_viewmat(camera.camera_to_worlds),
                    Ks=camera.get_intrinsics_matrices().to(model.device),
                    width=width,
                    height=height,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="D",
                    sh_degree=None,
                    rasterize_mode=model.config.rasterize_mode,
                )
                normalized_depth, stats = alpha_median_depth(
                    info,
                    width=width,
                    height=height,
                    quantile=args.alpha_quantile,
                )
                saved_depth = (normalized_depth / parser_scale).cpu().numpy()
            relative = Path(args.selection_split) / "raw-alpha-median-depth" / f"{Path(name).stem}.npy.gz"
            save_depth(stage / relative, saved_depth, args.compression_level)
            positive = saved_depth > 0
            row = {
                "ordinal": ordinal,
                "image": name,
                "depth": relative.as_posix(),
                "width": width,
                "height": height,
                **stats,
                "depth_min_saved_units": float(saved_depth[positive].min()) if positive.any() else 0.0,
                "depth_median_saved_units": float(np.median(saved_depth[positive])) if positive.any() else 0.0,
                "depth_max_saved_units": float(saved_depth[positive].max()) if positive.any() else 0.0,
            }
            rows.append(row)
            print(
                f"camera={ordinal}/{len(selected_names)} image={name} "
                f"valid={row['valid_pixel_fraction']:.6f} intersections={row['intersection_count']}",
                flush=True,
            )
            del info, normalized_depth
            torch.cuda.empty_cache()

        manifest = {
            "schema_version": 1,
            "method": "splatfacto_front_to_back_alpha_quantile",
            "depth_definition": "opencv_camera_z_in_saved_dataset_units",
            "alpha_quantile": float(args.alpha_quantile),
            "resolution_scale": float(args.resolution_scale),
            "person_masks": False,
            "selection_dataset": str(selection_root),
            "selection_split": args.selection_split,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(Path(checkpoint_path)),
            "checkpoint_step": int(step),
            "dataparser_scale_divisor": parser_scale,
            "cameras": rows,
        }
        (stage / "alpha_median_depth_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(f"complete cameras={len(rows)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
