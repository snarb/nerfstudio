#!/usr/bin/env python3
"""Render central camera-path previews from the trained full-dataset coordinate frame."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import torch

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.scripts.render import _render_trajectory_video
from nerfstudio.utils.eval_utils import eval_setup


CONFIG = Path(
    "/home/brans/lookcloser_temporal_runs/temporal_lookcloser_transfer_chain/lookcloser/"
    "008048_from_002020004_constant0p0005/eval_config_step_002050380.yml"
)
MANIFEST = Path("/home/brans/temporal_perframe_stride7_45f/perframe_manifest.json")
OUT_ROOT = Path("/home/brans/lookcloser_temporal_runs/videos/camera_paths_008048_working")


PATHS = {
    "vertical_col_c_D_to_L": [
        "D004_C014",
        "E004_C014",
        "F004_C014",
        "G004_C014",
        "H004_C016",
        "I004_C014",
        "J004_C014",
        "K004_C014",
        "L004_C014",
    ],
    "horizontal_row_h_B_to_D": ["H004_B014", "H004_C016", "H004_D014"],
    "horizontal_row_i_B_to_D": ["I004_B014", "I004_C014", "I004_D014"],
    "diagonal_center_D_B_to_L_D": [
        "D004_B014",
        "F004_C014",
        "H004_C016",
        "J004_C014",
        "L004_D014",
    ],
}


def camera_name_to_file() -> dict[str, str]:
    mapping = json.loads(MANIFEST.read_text())["camera_file_mapping"]
    return {camera_name: filename for filename, camera_name in mapping.items()}


def camera_from_dataset(dataset, filename: str):
    filename_to_idx = {
        Path(path).name: idx for idx, path in enumerate(dataset.image_filenames)
    }
    idx = filename_to_idx.get(filename)
    if idx is None:
        return None, None
    return dataset.cameras[idx : idx + 1], idx


def stack_single_cameras(cameras: list[Cameras]) -> Cameras:
    return Cameras(
        fx=torch.cat([cam.fx.reshape(-1) for cam in cameras], dim=0),
        fy=torch.cat([cam.fy.reshape(-1) for cam in cameras], dim=0),
        cx=torch.cat([cam.cx.reshape(-1) for cam in cameras], dim=0),
        cy=torch.cat([cam.cy.reshape(-1) for cam in cameras], dim=0),
        width=torch.cat([cam.width.reshape(-1) for cam in cameras], dim=0),
        height=torch.cat([cam.height.reshape(-1) for cam in cameras], dim=0),
        camera_to_worlds=torch.cat([cam.camera_to_worlds for cam in cameras], dim=0),
        camera_type=torch.cat([cam.camera_type.reshape(-1) for cam in cameras], dim=0),
        distortion_params=(
            torch.cat([cam.distortion_params for cam in cameras], dim=0)
            if cameras[0].distortion_params is not None
            else None
        ),
    )


def make_video_from_images(images_dir: Path, video_path: Path, fps: float) -> None:
    files = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.png")])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--paths", nargs="*", default=list(PATHS))
    parser.add_argument("--interpolation-steps", type=int, default=1)
    parser.add_argument("--frame-rate", type=float, default=12.0)
    parser.add_argument("--downscale-factor", type=float, default=8.0)
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    _, pipeline, _, _ = eval_setup(args.config, test_mode="inference")

    train_dataset = pipeline.datamanager.train_dataset
    eval_dataset = pipeline.datamanager.eval_dataset
    if train_dataset is None or eval_dataset is None:
        raise RuntimeError("Expected both train and eval datasets in pipeline")
    camera_to_file = camera_name_to_file()

    manifest: list[dict[str, object]] = []
    for name in args.paths:
        camera_names = PATHS[name]
        missing_cameras = [camera for camera in camera_names if camera not in camera_to_file]
        if missing_cameras:
            raise ValueError(f"{name}: cameras not found in manifest: {missing_cameras}")
        selected = []
        source_indices = []
        for camera in camera_names:
            filename = camera_to_file[camera]
            cam, idx = camera_from_dataset(train_dataset, filename)
            split = "train"
            if cam is None:
                cam, idx = camera_from_dataset(eval_dataset, filename)
                split = "eval"
            if cam is None or idx is None:
                raise ValueError(f"{name}: {filename} not found in loaded train/eval datasets")
            selected.append(cam)
            source_indices.append({"camera": camera, "filename": filename, "split": split, "index": idx})

        selected_cameras = stack_single_cameras(selected)
        camera_path = get_interpolated_camera_path(
            selected_cameras,
            steps=args.interpolation_steps,
            order_poses=False,
        )

        output_path = args.output_root / f"{name}.mp4"
        images_dir = output_path.parent / output_path.stem
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if output_path.exists():
            output_path.unlink()

        seconds = max(1e-6, len(camera_path) / args.frame_rate)
        _render_trajectory_video(
            pipeline=pipeline,
            cameras=camera_path,
            output_filename=output_path,
            rendered_output_names=["rgb"],
            rendered_resolution_scaling_factor=1.0 / args.downscale_factor,
            seconds=seconds,
            output_format="images",
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
        if not args.skip_video:
            make_video_from_images(images_dir, output_path, args.frame_rate)

        manifest.append(
            {
                "name": name,
                "cameras": camera_names,
                "source_indices": source_indices,
                "images_dir": str(images_dir),
                "video": str(output_path) if not args.skip_video else None,
                "interpolation_steps": args.interpolation_steps,
                "frame_rate": args.frame_rate,
                "downscale_factor": args.downscale_factor,
                "config": str(args.config),
                "coordinate_frame": "loaded full train/eval dataset cameras from eval_setup; no subset dataparser recentering",
            }
        )
        print(output_path, flush=True)

    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
