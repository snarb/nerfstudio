#!/usr/bin/env python3
"""Render camera-path videos across the temporal per-frame checkpoint chain."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup

from render_central_camera_path_videos import (
    MANIFEST,
    PATHS,
    camera_from_dataset,
    camera_name_to_file,
    stack_single_cameras,
)


LEADER_CONFIG = Path(
    "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_sanity/lookcloser/"
    "007740_leader_local_20260705_183929/eval_config_step_000106316.yml"
)
LR_SWEEP_ROOT = Path("/home/brans/lookcloser_temporal_runs/temporal_lookcloser_lr_sweep_007747/lookcloser")
CHAIN_ROOT = Path("/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser")
OUT_ROOT = Path("/home/brans/lookcloser_temporal_runs/videos/camera_paths_temporal_working")


def discover_frame_configs() -> list[tuple[str, Path]]:
    configs: dict[str, Path] = {"007740": LEADER_CONFIG}
    configs["007747"] = (
        LR_SWEEP_ROOT / "007747_const5e-4_20260705_184726" / "eval_config_step_000151880.yml"
    )
    for config in sorted(CHAIN_ROOT.glob("*/eval_config_step_*.yml")):
        match = re.match(r"(\d{6})_", config.parent.name)
        if match:
            configs[match.group(1)] = config
    missing = [frame for frame, config in configs.items() if not config.exists()]
    if missing:
        raise FileNotFoundError(f"Missing configs for frames: {missing}")
    return sorted(configs.items())


def build_path_cameras(pipeline, path_name: str, interpolation_steps: int, downscale_factor: float) -> Cameras:
    train_dataset = pipeline.datamanager.train_dataset
    eval_dataset = pipeline.datamanager.eval_dataset
    if train_dataset is None or eval_dataset is None:
        raise RuntimeError("Expected both train and eval datasets in pipeline")

    camera_to_file = camera_name_to_file()
    selected = []
    for camera_name in PATHS[path_name]:
        filename = camera_to_file[camera_name]
        camera, _ = camera_from_dataset(train_dataset, filename)
        if camera is None:
            camera, _ = camera_from_dataset(eval_dataset, filename)
        if camera is None:
            raise ValueError(f"{path_name}: {filename} not found in loaded train/eval datasets")
        selected.append(camera)

    selected_cameras = stack_single_cameras(selected)
    cameras = get_interpolated_camera_path(
        selected_cameras,
        steps=interpolation_steps,
        order_poses=False,
    )
    cameras.rescale_output_resolution(1.0 / downscale_factor)
    return cameras


def render_rgb(pipeline, camera: Cameras) -> np.ndarray:
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera.to(pipeline.device))
        rgb = colormaps.apply_colormap(outputs["rgb"]).detach().cpu().numpy()
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def write_video(images_dir: Path, video_path: Path, fps: float) -> dict[str, object]:
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No rendered images in {images_dir}")
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(files[0])
    height, width = first.shape[:2]
    even_width = (width // 2) * 2
    even_height = (height // 2) * 2
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (even_width, even_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")
    try:
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(file)
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image[:even_height, :even_width])
    finally:
        writer.release()
    return {
        "video": str(video_path),
        "frames": len(files),
        "fps": fps,
        "resolution": f"{even_width}x{even_height}",
        "bytes": video_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--paths", nargs="*", default=list(PATHS))
    parser.add_argument("--interpolation-steps", type=int, default=8)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--downscale-factor", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    frame_configs = discover_frame_configs()
    if args.max_frames is not None:
        frame_configs = frame_configs[: args.max_frames]
    total_frames = len(frame_configs)

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path_name in args.paths:
        (args.output_root / path_name).mkdir(parents=True, exist_ok=True)

    rendered: list[dict[str, object]] = []
    for temporal_idx, (frame, config) in enumerate(frame_configs):
        print(f"frame={frame} {temporal_idx + 1}/{total_frames} config={config}", flush=True)
        _, pipeline, _, _ = eval_setup(config, test_mode="inference")
        pipeline.eval()

        for path_name in args.paths:
            cameras = build_path_cameras(
                pipeline,
                path_name,
                interpolation_steps=args.interpolation_steps,
                downscale_factor=args.downscale_factor,
            )
            camera_idx = round(temporal_idx * (cameras.size - 1) / max(1, total_frames - 1))
            rgb = render_rgb(pipeline, cameras[camera_idx : camera_idx + 1])
            image_path = args.output_root / path_name / f"{temporal_idx:05d}.jpg"
            Image.fromarray(rgb).save(image_path, quality=args.jpeg_quality)
            rendered.append(
                {
                    "temporal_index": temporal_idx,
                    "frame": frame,
                    "config": str(config),
                    "path": path_name,
                    "camera_path_index": camera_idx,
                    "image": str(image_path),
                }
            )

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    videos = []
    if not args.skip_video:
        for path_name in args.paths:
            videos.append(
                write_video(
                    args.output_root / path_name,
                    args.output_root / f"{path_name}.mp4",
                    args.frame_rate,
                )
            )

    manifest = {
        "frame_count": total_frames,
        "frames": [{"frame": frame, "config": str(config)} for frame, config in frame_configs],
        "paths": {name: PATHS[name] for name in args.paths},
        "interpolation_steps": args.interpolation_steps,
        "frame_rate": args.frame_rate,
        "downscale_factor": args.downscale_factor,
        "rendered": rendered,
        "videos": videos,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
