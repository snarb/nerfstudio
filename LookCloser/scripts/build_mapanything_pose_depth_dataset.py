#!/usr/bin/env python3
"""Build a Nerfstudio depth dataset with pose-conditioned MapAnything.

The source cameras are provided as geometric inputs.  MapAnything still emits
its own camera rig, so its predicted depths are rescaled from the ratio between
predicted and source camera baselines.  Depth pixels are then reprojected from
the model's inferred pinhole calibration back onto the original image grid.

No person/foreground segmentation is created or consumed.  The optional
MapAnything ambiguity/edge mask is only a geometric confidence filter.
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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=20.0,
        help="Per-view low-confidence percentile rejected after inference; zero disables it.",
    )
    parser.add_argument(
        "--no-geometric-mask",
        action="store_true",
        help="Disable MapAnything's ambiguity and depth-edge validity filter.",
    )
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--preview-count", type=int, default=4)
    args = parser.parse_args(argv)
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not (args.input / "transforms.json").is_file():
        parser.error(f"Missing transforms.json below {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.minibatch_size <= 0 or args.preview_count <= 0:
        parser.error("Minibatch size and preview count must be positive")
    if not 0 <= args.confidence_percentile < 100:
        parser.error("Confidence percentile must be in [0, 100)")
    if not 0 <= args.compression_level <= 9:
        parser.error("Compression level must be in [0, 9]")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inherited(frame: dict[str, Any], payload: dict[str, Any], key: str) -> float:
    value = frame.get(key, payload.get(key))
    if value is None:
        raise ValueError(f"Missing {key!r} for {frame.get('file_path')!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} for {frame.get('file_path')!r}")
    return value


def selected_train_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Dataset has no frames")
    if any(frame.get("mask_path") for frame in frames):
        raise ValueError("MapAnything depth generation forbids person/image masks")
    declared = payload.get("train_filenames")
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            raise ValueError("train_filenames must be a non-empty list when declared")
        names = set(map(str, declared))
        selected = [frame for frame in frames if str(frame.get("file_path")) in names]
        if len(selected) != len(names):
            found = {str(frame.get("file_path")) for frame in selected}
            raise ValueError(f"Unknown train filenames: {sorted(names - found)[:8]}")
        order = {str(name): index for index, name in enumerate(declared)}
        return sorted(selected, key=lambda frame: order[str(frame["file_path"])])
    selected = [frame for frame in frames if "eval" not in Path(str(frame.get("file_path", ""))).stem.lower()]
    if not selected:
        raise ValueError("Filename-inferred train split is empty")
    return selected


def source_camera(
    frame: dict[str, Any], payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV c2w and pinhole K from a Nerfstudio frame."""

    c2w_gl = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
    if c2w_gl.shape != (4, 4) or not np.isfinite(c2w_gl).all():
        raise ValueError(f"Invalid transform_matrix for {frame.get('file_path')!r}")
    c2w_cv = c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
    intrinsic = np.asarray(
        [
            [inherited(frame, payload, "fl_x"), 0.0, inherited(frame, payload, "cx")],
            [0.0, inherited(frame, payload, "fl_y"), inherited(frame, payload, "cy")],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return c2w_cv.astype(np.float32), intrinsic.astype(np.float32)


def robust_baseline_scale(predicted_centers: np.ndarray, source_centers: np.ndarray) -> float:
    if predicted_centers.shape != source_centers.shape or predicted_centers.ndim != 2:
        raise ValueError("Camera center arrays must have the same (N, D) shape")
    ratios: list[float] = []
    for first in range(len(predicted_centers)):
        for second in range(first + 1, len(predicted_centers)):
            predicted = float(np.linalg.norm(predicted_centers[first] - predicted_centers[second]))
            source = float(np.linalg.norm(source_centers[first] - source_centers[second]))
            if predicted > 1e-8 and source > 1e-8:
                ratios.append(source / predicted)
    if not ratios:
        raise ValueError("Cannot estimate depth scale from degenerate camera centers")
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid camera-baseline scale: {scale}")
    return scale


def similarity_residual(
    predicted_centers: np.ndarray, source_centers: np.ndarray, scale: float
) -> float:
    """Fit rotation/translation after fixed scale and return normalized RMSE."""

    predicted_mean = predicted_centers.mean(axis=0)
    source_mean = source_centers.mean(axis=0)
    left = scale * (predicted_centers - predicted_mean)
    right = source_centers - source_mean
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = left @ rotation + source_mean
    rmse = float(np.sqrt(np.mean(np.sum((aligned - source_centers) ** 2, axis=1))))
    extent = float(np.sqrt(np.mean(np.sum(right**2, axis=1))))
    return rmse / max(extent, 1e-8)


def remap_pinhole_scalar(
    image: np.ndarray,
    *,
    predicted_intrinsic: np.ndarray,
    source_intrinsic: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
) -> np.ndarray:
    """Sample a scalar prediction onto the source camera's pixel grid."""

    rows, columns = np.indices((height, width), dtype=np.float32)
    map_x = (
        float(predicted_intrinsic[0, 0]) / float(source_intrinsic[0, 0])
    ) * (columns - float(source_intrinsic[0, 2])) + float(predicted_intrinsic[0, 2])
    map_y = (
        float(predicted_intrinsic[1, 1]) / float(source_intrinsic[1, 1])
    ) * (rows - float(source_intrinsic[1, 2])) + float(predicted_intrinsic[1, 2])
    return cv2.remap(
        np.asarray(image, dtype=np.float32),
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def clone_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for root, directories, files in os.walk(source):
        relative = Path(root).relative_to(source)
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            if relative == Path(".") and name in {"transforms.json", "mapanything_pose_depth_manifest.json"}:
                continue
            source_file = Path(root) / name
            materialized = source_file.resolve(strict=True) if source_file.is_symlink() else source_file
            destination_file = target / name
            try:
                os.link(materialized, destination_file)
            except OSError:
                shutil.copyfile(materialized, destination_file)


def save_depth(path: Path, depth: np.ndarray, compression_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=compression_level) as stream:
        np.save(stream, np.asarray(depth, dtype=np.float32), allow_pickle=False)


def preview_panel(rgb_path: Path, depth: np.ndarray, confidence: np.ndarray, label: str) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    rgb.thumbnail((360, 220), Image.Resampling.LANCZOS)
    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        low, high = np.quantile(depth[valid], [0.02, 0.98])
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    else:
        normalized = np.zeros_like(depth)
    colored = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).resize(
        rgb.size, Image.Resampling.NEAREST
    )
    conf = np.asarray(confidence, dtype=np.float32)
    finite_conf = np.isfinite(conf)
    if finite_conf.any():
        low, high = np.quantile(conf[finite_conf], [0.02, 0.98])
        conf = np.clip((conf - low) / max(high - low, 1e-6), 0, 1)
    else:
        conf = np.zeros_like(conf)
    conf_image = Image.fromarray(np.round(conf * 255).astype(np.uint8), mode="L").convert("RGB")
    conf_image = conf_image.resize(rgb.size, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (rgb.width * 3, rgb.height + 24), "black")
    panel.paste(rgb, (0, 24))
    panel.paste(depth_image, (rgb.width, 24))
    panel.paste(conf_image, (2 * rgb.width, 24))
    ImageDraw.Draw(panel).text((4, 4), label, fill="white")
    return panel


def save_preview(path: Path, panels: list[Image.Image]) -> None:
    width = max(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, sum(panel.height for panel in panels)), "black")
    top = 0
    for panel in panels:
        canvas.paste(ImageOps.pad(panel, (width, panel.height), color="black"), (0, top))
        top += panel.height
    canvas.save(path, quality=92, subsampling=0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads((args.input / "transforms.json").read_text(encoding="utf-8"))
    frames = selected_train_frames(payload)
    image_paths = [(args.input / str(frame["file_path"])).resolve(strict=True) for frame in frames]
    cameras = [source_camera(frame, payload) for frame in frames]

    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    views = []
    for image_path, (c2w_cv, intrinsic) in zip(image_paths, cameras):
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        views.append(
            {
                "img": torch.from_numpy(image.copy()),
                "intrinsics": torch.from_numpy(intrinsic),
                "camera_poses": torch.from_numpy(c2w_cv),
                "is_metric_scale": torch.tensor([True]),
            }
        )
    processed_views = preprocess_inputs(views, verbose=False)
    model = MapAnything.from_pretrained(args.model).to(args.device).eval()
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=True,
        minibatch_size=args.minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=not args.no_geometric_mask,
        mask_edges=not args.no_geometric_mask,
        apply_confidence_mask=False,
        use_multiview_confidence=False,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=True,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=True,
        ignore_pose_scale_inputs=False,
    )
    if len(predictions) != len(frames):
        raise RuntimeError("MapAnything returned an unexpected view count")

    predicted_centers = np.stack(
        [prediction["camera_poses"][0, :3, 3].detach().float().cpu().numpy() for prediction in predictions]
    )
    source_centers = np.stack([camera[0][:3, 3] for camera in cameras])
    depth_scale = robust_baseline_scale(predicted_centers, source_centers)
    camera_center_nrmse = similarity_residual(predicted_centers, source_centers, depth_scale)

    stage = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    rows: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    try:
        clone_tree(args.input, stage)
        output_payload = json.loads(json.dumps(payload))
        selected_by_path = {str(frame["file_path"]): index for index, frame in enumerate(frames)}
        preview_ordinals = set(
            np.linspace(0, len(frames) - 1, min(args.preview_count, len(frames))).round().astype(int)
        )
        for frame in output_payload["frames"]:
            relative_image = str(frame["file_path"])
            width = int(inherited(frame, output_payload, "w"))
            height = int(inherited(frame, output_payload, "h"))
            relative_depth = Path("mapanything_pose_depth") / f"{Path(relative_image).stem}.npy.gz"
            if relative_image in selected_by_path:
                index = selected_by_path[relative_image]
                prediction = predictions[index]
                predicted_intrinsic = prediction["intrinsics"][0].detach().float().cpu().numpy()
                source_intrinsic = cameras[index][1]
                depth_low = prediction["depth_z"][0].squeeze(-1).detach().float().cpu().numpy()
                confidence_low = prediction["conf"][0].detach().float().cpu().numpy()
                depth = remap_pinhole_scalar(
                    depth_low * depth_scale,
                    predicted_intrinsic=predicted_intrinsic,
                    source_intrinsic=source_intrinsic,
                    width=width,
                    height=height,
                    interpolation=cv2.INTER_LINEAR,
                )
                confidence = remap_pinhole_scalar(
                    confidence_low,
                    predicted_intrinsic=predicted_intrinsic,
                    source_intrinsic=source_intrinsic,
                    width=width,
                    height=height,
                    interpolation=cv2.INTER_LINEAR,
                )
                valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
                if not args.no_geometric_mask and "mask" in prediction:
                    mask_low = prediction["mask"][0].squeeze(-1).detach().float().cpu().numpy()
                    geometric_mask = remap_pinhole_scalar(
                        mask_low,
                        predicted_intrinsic=predicted_intrinsic,
                        source_intrinsic=source_intrinsic,
                        width=width,
                        height=height,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    valid &= geometric_mask > 0.5
                threshold = -np.inf
                if args.confidence_percentile > 0 and valid.any():
                    threshold = float(np.percentile(confidence[valid], args.confidence_percentile))
                    valid &= confidence >= threshold
                depth = np.where(valid, depth, 0).astype(np.float32)
                save_depth(stage / relative_depth, depth, args.compression_level)
                values = depth[depth > 0]
                if not values.size:
                    raise RuntimeError(f"MapAnything produced no usable depth for {relative_image}")
                row = {
                    "image": relative_image,
                    "depth": relative_depth.as_posix(),
                    "confidence_threshold": threshold,
                    "valid_pixel_fraction": float((depth > 0).mean()),
                    "depth_min": float(values.min()),
                    "depth_median": float(np.median(values)),
                    "depth_max": float(values.max()),
                }
                rows.append(row)
                if index in preview_ordinals:
                    previews.append(
                        preview_panel(args.input / relative_image, depth, confidence, f"{index}: {Path(relative_image).name}")
                    )
            else:
                save_depth(stage / relative_depth, np.zeros((height, width), dtype=np.float32), args.compression_level)
            frame["depth_file_path"] = relative_depth.as_posix()

        fractions = np.asarray([row["valid_pixel_fraction"] for row in rows])
        medians = np.asarray([row["depth_median"] for row in rows])
        teacher = {
            "method": "mapanything_pose_conditioned",
            "model": args.model,
            "input_dataset": str(args.input),
            "input_transforms_sha256": sha256(args.input / "transforms.json"),
            "image_count": len(rows),
            "model_resolution": list(predictions[0]["depth_z"].shape[1:3]),
            "confidence_percentile": args.confidence_percentile,
            "geometric_confidence_mask": not args.no_geometric_mask,
            "depth_scale_from_camera_baselines": depth_scale,
            "camera_center_similarity_nrmse": camera_center_nrmse,
            "depth_definition": "opencv_camera_z_in_saved_dataset_units",
            "required_depth_unit_scale_factor": 1.0,
            "person_masks": False,
            "valid_pixel_fraction": {
                "min": float(fractions.min()),
                "median": float(np.median(fractions)),
                "max": float(fractions.max()),
            },
            "depth_median_across_views": {
                "min": float(medians.min()),
                "median": float(np.median(medians)),
                "max": float(medians.max()),
            },
        }
        output_payload["rendered_depth_teacher"] = teacher
        (stage / "transforms.json").write_text(
            json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "mapanything_pose_depth_manifest.json").write_text(
            json.dumps({"schema_version": 1, **teacher, "cameras": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_preview(stage / "mapanything_pose_depth_preview.jpg", previews)
        os.replace(stage, args.output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        del predictions, model
        torch.cuda.empty_cache()

    print(
        f"complete images={len(rows)} coverage_median={np.median(fractions):.6f} "
        f"depth_median={np.median(medians):.6f} scale={depth_scale:.6f} "
        f"camera_nrmse={camera_center_nrmse:.6f} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
