#!/usr/bin/env python3
"""Run an external pretrained MVS model on calibrated Nerfstudio images.

Supported backends are the official MonoMVSNet and MVSMamba repositories.  The
wrapper supplies the source OpenCV calibration directly, selects nearby source
views, and writes full-resolution Z-depth maps in the source dataset units.
External repositories and checkpoints are explicit arguments; they are never
installed into or imported by the normal LookCloser/nerfstudio code path.

No person or foreground mask is created or consumed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("monomvsnet", "mvsmamba"), required=True)
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-views", type=int, default=5)
    parser.add_argument("--max-width", type=int, default=1152)
    parser.add_argument("--max-height", type=int, default=864)
    parser.add_argument("--depth-min", type=float, required=True)
    parser.add_argument("--depth-max", type=float, required=True)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument(
        "--limit-references",
        type=int,
        default=None,
        help="Debug-only limit; unprocessed train frames receive zero depth.",
    )
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--preview-count", type=int, default=4)
    args = parser.parse_args(argv)
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.backend_root = args.backend_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not (args.input / "transforms.json").is_file():
        parser.error(f"Missing transforms.json below {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not args.backend_root.is_dir() or not args.checkpoint.is_file():
        parser.error("Backend root and checkpoint must exist")
    if args.num_views < 2 or args.max_width < 64 or args.max_height < 64:
        parser.error("Need at least two views and a >=64 pixel inference extent")
    if not 0 < args.depth_min < args.depth_max:
        parser.error("Require 0 < depth-min < depth-max")
    if not 0 <= args.confidence_percentile < 100:
        parser.error("Confidence percentile must be in [0, 100)")
    if args.limit_references is not None and args.limit_references <= 0:
        parser.error("limit-references must be positive")
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
        raise ValueError("Pretrained MVS inference forbids person/image masks")
    declared = payload.get("train_filenames")
    if declared is not None:
        names = set(map(str, declared))
        order = {str(name): index for index, name in enumerate(declared)}
        selected = [frame for frame in frames if str(frame.get("file_path")) in names]
        if len(selected) != len(names):
            found = {str(frame.get("file_path")) for frame in selected}
            raise ValueError(f"Unknown train filenames: {sorted(names - found)[:8]}")
        return sorted(selected, key=lambda frame: order[str(frame["file_path"])])
    selected = [frame for frame in frames if "eval" not in Path(str(frame.get("file_path", ""))).stem.lower()]
    if not selected:
        raise ValueError("Filename-inferred train split is empty")
    return selected


def source_camera(
    frame: dict[str, Any], payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
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


def resize_dimensions(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    resized_width = max(64, int(np.floor(width * scale / 64)) * 64)
    resized_height = max(64, int(np.floor(height * scale / 64)) * 64)
    return resized_width, resized_height


def source_view_order(cameras: list[tuple[np.ndarray, np.ndarray]], reference: int) -> list[int]:
    """Rank views by camera-center distance with an optical-axis tie breaker."""

    reference_pose = cameras[reference][0]
    reference_center = reference_pose[:3, 3]
    reference_axis = reference_pose[:3, 2]
    scores: list[tuple[float, int]] = []
    distances = [
        float(np.linalg.norm(camera[0][:3, 3] - reference_center))
        for index, camera in enumerate(cameras)
        if index != reference
    ]
    distance_scale = max(float(np.median(distances)), 1e-8)
    for index, (pose, _) in enumerate(cameras):
        if index == reference:
            continue
        distance = float(np.linalg.norm(pose[:3, 3] - reference_center)) / distance_scale
        axis_similarity = float(np.clip(np.dot(reference_axis, pose[:3, 2]), -1, 1))
        scores.append((distance + 0.25 * (1 - axis_similarity), index))
    return [reference] + [index for _, index in sorted(scores)]


def build_projection_pyramid(
    cameras: list[tuple[np.ndarray, np.ndarray]],
    view_ids: list[int],
    *,
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    device: str,
):
    import torch

    projections = []
    for view_id in view_ids:
        c2w, source_intrinsic = cameras[view_id]
        full_intrinsic = source_intrinsic.copy()
        full_intrinsic[0, :] *= resized_width / width
        full_intrinsic[1, :] *= resized_height / height
        projection = np.zeros((2, 4, 4), dtype=np.float32)
        projection[0] = np.linalg.inv(c2w).astype(np.float32)
        projection[1, :3, :3] = full_intrinsic / np.asarray(
            [[4, 4, 4], [4, 4, 4], [1, 1, 1]], dtype=np.float32
        )
        projection[1, 2, 2] = 1.0
        projections.append(projection)
    stage2 = np.stack(projections)
    stage1 = stage2.copy()
    stage1[:, 1, :2, :] /= 2
    stage3 = stage2.copy()
    stage3[:, 1, :2, :] *= 2
    stage4 = stage2.copy()
    stage4[:, 1, :2, :] *= 4
    return {
        "stage1": torch.from_numpy(stage1)[None].to(device),
        "stage2": torch.from_numpy(stage2)[None].to(device),
        "stage3": torch.from_numpy(stage3)[None].to(device),
        "stage4": torch.from_numpy(stage4)[None].to(device),
    }


def load_images(
    image_paths: list[Path],
    view_ids: list[int],
    *,
    width: int,
    height: int,
    backend: str,
    device: str,
):
    import torch

    images = []
    raw_images = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    for view_id in view_ids:
        source = np.asarray(Image.open(image_paths[view_id]).convert("RGB"), dtype=np.uint8)
        resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1)[None].to(device)
        if backend == "monomvsnet":
            raw_images.append(tensor)
            images.append((tensor.float() / 255.0 - mean) / std)
        else:
            images.append(tensor.float() / 255.0)
    return images, raw_images


def load_backend(args: argparse.Namespace):
    import torch

    sys.path.insert(0, str(args.backend_root))
    previous_cwd = Path.cwd()
    os.chdir(args.backend_root)
    try:
        if args.backend == "monomvsnet":
            from models import MonoMVSNet

            model = MonoMVSNet(
                arch_mode="fpn",
                reg_net="reg2d",
                num_stage=4,
                fpn_base_channel=8,
                reg_channel=8,
                stage_splits=[8, 8, 4, 4],
                depth_interals_ratio=[0.5, 0.5, 0.5, 0.5],
                group_cor=True,
                group_cor_dim=[8, 8, 4, 4],
                inverse_depth=True,
                agg_type="ConvBnReLU3D",
                attn_temp=2,
                mono_sampling=True,
                edge_guide=True,
                attention=True,
                max_h=args.max_height,
                max_w=args.max_width,
            )
        else:
            from models import MVSMamba

            model = MVSMamba(
                arch_mode="fpn",
                num_stage=4,
                fpn_base_channel=8,
                reg_channel=8,
                stage_splits=[32, 16, 8, 4],
                depth_interals_ratio=[2.0, 1.0, 1.0, 0.5],
                group_cor=True,
                group_cor_dim=[4, 4, 4, 4],
                inverse_depth=True,
                agg_type="ConvBnReLU3D",
                attn_temp=2,
                fpn_mamba=True,
            )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        return model.to(args.device).eval()
    finally:
        os.chdir(previous_cwd)


def predict_reference(
    model,
    *,
    args: argparse.Namespace,
    image_paths: list[Path],
    cameras: list[tuple[np.ndarray, np.ndarray]],
    reference: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import torch
    import torch.nn.functional as functional

    with Image.open(image_paths[reference]) as reference_image:
        width, height = reference_image.size
    resized_width, resized_height = resize_dimensions(width, height, args.max_width, args.max_height)
    view_ids = source_view_order(cameras, reference)[: min(args.num_views, len(cameras))]
    images, raw_images = load_images(
        image_paths,
        view_ids,
        width=resized_width,
        height=resized_height,
        backend=args.backend,
        device=args.device,
    )
    projections = build_projection_pyramid(
        cameras,
        view_ids,
        width=width,
        height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        device=args.device,
    )
    depth_values = torch.linspace(args.depth_min, args.depth_max, 192, device=args.device)[None]
    with torch.inference_mode():
        if args.backend == "monomvsnet":
            outputs = model(images, raw_images, projections, depth_values)
        else:
            outputs = model(images, projections, depth_values)
    depth = outputs["depth"][0].detach().float().cpu().numpy()
    confidence_maps = []
    for stage in range(1, 5):
        confidence = outputs[f"stage{stage}"]["photometric_confidence"]
        confidence = functional.interpolate(
            confidence[:, None], size=depth.shape, mode="bilinear", align_corners=False
        )[:, 0]
        confidence_maps.append(confidence[0].detach().float().cpu().numpy())
    confidence = np.prod(np.stack(confidence_maps), axis=0)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float32), confidence.astype(np.float32), view_ids


def clone_tree(source: Path, destination: Path) -> None:
    destination.mkdir()
    for root, directories, files in os.walk(source):
        relative = Path(root).relative_to(source)
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            if relative == Path(".") and name in {"transforms.json", "pretrained_mvs_depth_manifest.json"}:
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
    normalized = np.zeros_like(depth)
    if valid.any():
        low, high = np.quantile(depth[valid], [0.02, 0.98])
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    colored = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_image = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).resize(
        rgb.size, Image.Resampling.NEAREST
    )
    confidence_normalized = np.clip(confidence, 0, 1)
    confidence_image = Image.fromarray(
        np.round(confidence_normalized * 255).astype(np.uint8), mode="L"
    ).convert("RGB").resize(rgb.size, Image.Resampling.NEAREST)
    panel = Image.new("RGB", (3 * rgb.width, rgb.height + 24), "black")
    panel.paste(rgb, (0, 24))
    panel.paste(depth_image, (rgb.width, 24))
    panel.paste(confidence_image, (2 * rgb.width, 24))
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
    reference_count = len(frames) if args.limit_references is None else min(args.limit_references, len(frames))

    import torch

    model = load_backend(args)
    predicted: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for reference in range(reference_count):
        relative_image = str(frames[reference]["file_path"])
        predicted[relative_image] = predict_reference(
            model,
            args=args,
            image_paths=image_paths,
            cameras=cameras,
            reference=reference,
        )
        print(f"predicted={reference + 1}/{reference_count} image={Path(relative_image).name}", flush=True)

    stage = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    rows: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    try:
        clone_tree(args.input, stage)
        output_payload = json.loads(json.dumps(payload))
        preview_ordinals = set(
            np.linspace(0, reference_count - 1, min(args.preview_count, reference_count)).round().astype(int)
        )
        for frame in output_payload["frames"]:
            relative_image = str(frame["file_path"])
            width = int(inherited(frame, output_payload, "w"))
            height = int(inherited(frame, output_payload, "h"))
            relative_depth = Path(f"{args.backend}_depth") / f"{Path(relative_image).stem}.npy.gz"
            if relative_image in predicted:
                depth, confidence, view_ids = predicted[relative_image]
                valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
                threshold = -np.inf
                if args.confidence_percentile > 0 and valid.any():
                    threshold = float(np.percentile(confidence[valid], args.confidence_percentile))
                    valid &= confidence >= threshold
                depth = np.where(valid, depth, 0).astype(np.float32)
                values = depth[depth > 0]
                if not values.size:
                    raise RuntimeError(f"No usable depth for {relative_image}")
                save_depth(stage / relative_depth, depth, args.compression_level)
                row = {
                    "image": relative_image,
                    "depth": relative_depth.as_posix(),
                    "source_images": [str(frames[index]["file_path"]) for index in view_ids[1:]],
                    "confidence_threshold": threshold,
                    "valid_pixel_fraction": float((depth > 0).mean()),
                    "depth_min": float(values.min()),
                    "depth_median": float(np.median(values)),
                    "depth_max": float(values.max()),
                }
                rows.append(row)
                source_index = next(index for index, source in enumerate(frames) if source["file_path"] == relative_image)
                if source_index in preview_ordinals:
                    previews.append(
                        preview_panel(args.input / relative_image, depth, confidence, f"{source_index}: {Path(relative_image).name}")
                    )
            else:
                save_depth(stage / relative_depth, np.zeros((height, width), dtype=np.float32), args.compression_level)
            frame["depth_file_path"] = relative_depth.as_posix()

        fractions = np.asarray([row["valid_pixel_fraction"] for row in rows])
        medians = np.asarray([row["depth_median"] for row in rows])
        teacher = {
            "method": f"{args.backend}_calibrated_pretrained_mvs",
            "backend_root": str(args.backend_root),
            "backend_commit": subprocess.check_output(
                ["git", "-C", str(args.backend_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "input_dataset": str(args.input),
            "input_transforms_sha256": sha256(args.input / "transforms.json"),
            "image_count": len(rows),
            "source_view_count": args.num_views,
            "max_inference_resolution": [args.max_height, args.max_width],
            "depth_range": [args.depth_min, args.depth_max],
            "confidence_percentile": args.confidence_percentile,
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
        (stage / "pretrained_mvs_depth_manifest.json").write_text(
            json.dumps({"schema_version": 1, **teacher, "cameras": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_preview(stage / f"{args.backend}_depth_preview.jpg", previews)
        os.replace(stage, args.output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        del model
        torch.cuda.empty_cache()

    print(
        f"complete backend={args.backend} images={len(rows)} coverage_median={np.median(fractions):.6f} "
        f"depth_median={np.median(medians):.6f} output={args.output}",
        flush=True,
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
