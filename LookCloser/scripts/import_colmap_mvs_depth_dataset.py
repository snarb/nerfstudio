#!/usr/bin/env python3
"""Attach train-only COLMAP MVS camera-z depths to a Nerfstudio dataset.

The source RGB dataset is immutable. Images are linked, masks are rejected,
and held-out frames receive no depth even if a stray MVS file exists.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np
from PIL import Image


def normalized_name(value: str) -> str:
    name = PurePosixPath(value).as_posix()
    while name.startswith("./"):
        name = name[2:]
    return name


def read_colmap_dense_array(path: Path) -> np.ndarray:
    """Read COLMAP's ``width&height&channels&`` column-major dense format."""

    with path.open("rb") as handle:
        header: list[int] = []
        for _ in range(3):
            token = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise ValueError(f"Truncated COLMAP dense header: {path}")
                if value == b"&":
                    break
                token.extend(value)
            header.append(int(token))
        width, height, channels = header
        values = np.fromfile(handle, dtype=np.float32)
    expected = width * height * channels
    if width <= 0 or height <= 0 or channels <= 0 or values.size != expected:
        raise ValueError(f"Invalid COLMAP dense payload {path}: header={header}, values={values.size}")
    return values.reshape((width, height, channels), order="F").transpose(1, 0, 2)


def save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as handle:
        np.save(handle, depth.astype(np.float32, copy=False), allow_pickle=False)


def save_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.quantile(depth[valid], (0.01, 0.99))
        normalized = np.clip((depth - low) / max(float(high - low), 1e-8), 0.0, 1.0)
        image[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--depth-maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-type", choices=("geometric", "photometric"), default="geometric")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = args.data.expanduser().resolve()
    depth_maps = args.depth_maps.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    payload = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    train_filenames = payload.get("train_filenames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json contains no frames")
    if not isinstance(train_filenames, list) or not train_filenames:
        raise ValueError("Explicit train_filenames is required to prevent eval-depth leakage")
    train_names = {normalized_name(str(value)) for value in train_filenames}
    output.mkdir(parents=True)
    depth_dir = output / "depth"
    preview_dir = output / "depth_previews"
    preview_dir.mkdir()
    coverages: list[float] = []
    medians: list[float] = []
    heldout_frames: list[dict[str, object]] = []
    depth_shape: tuple[int, int] | None = None
    imported = 0
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
            raise ValueError("Every frame must contain a string file_path")
        if "mask_path" in frame:
            raise ValueError("Person/image masks are forbidden in this pipeline")
        name = normalized_name(frame["file_path"])
        if name not in train_names:
            heldout_frames.append(frame)
            continue
        source = depth_maps / f"{name}.{args.input_type}.bin"
        if not source.is_file():
            raise FileNotFoundError(source)
        dense = read_colmap_dense_array(source)
        if dense.shape[-1] != 1:
            raise ValueError(f"Expected scalar depth map, got {dense.shape}: {source}")
        depth = dense[..., 0]
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        if depth_shape is None:
            depth_shape = depth.shape
        elif depth.shape != depth_shape:
            raise ValueError(f"MVS depth shapes are not uniform: {depth.shape} != {depth_shape}")
        target_name = f"mvs_{imported:05d}.npy.gz"
        save_depth(depth_dir / target_name, depth)
        frame["depth_file_path"] = f"depth/{target_name}"
        coverages.append(float(valid.mean()))
        medians.append(float(np.median(depth[valid])))
        if imported in {0, len(train_names) // 2, len(train_names) - 1}:
            save_preview(preview_dir / f"mvs_{imported:05d}.png", depth)
        imported += 1
    if imported != len(train_names):
        raise ValueError(f"Imported {imported} train depths but expected {len(train_names)}")
    if depth_shape is None:
        raise RuntimeError("No train depth shape was resolved")
    if heldout_frames:
        placeholder_name = "heldout_invalid.npy.gz"
        save_depth(depth_dir / placeholder_name, np.zeros(depth_shape, dtype=np.float32))
        for frame in heldout_frames:
            frame["depth_file_path"] = f"depth/{placeholder_name}"
    images = data / "images"
    if not images.is_dir():
        raise FileNotFoundError(images)
    (output / "images").symlink_to(images, target_is_directory=True)
    payload["depth_unit_scale_factor"] = 1.0
    payload["colmap_mvs_depth"] = {
        "schema_version": 1,
        "source_data": str(data),
        "source_depth_maps": str(depth_maps),
        "input_type": args.input_type,
        "split": "train_filenames_only",
        "train_depth_count": imported,
        "heldout_invalid_depth_count": len(heldout_frames),
        "coverage_mean": float(np.mean(coverages)),
        "coverage_min": float(np.min(coverages)),
        "median_camera_z_mean": float(np.mean(medians)),
        "masks": "forbidden",
    }
    (output / "transforms.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(data / "transforms.json", output / "transforms.source.json")
    print(json.dumps(payload["colmap_mvs_depth"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
