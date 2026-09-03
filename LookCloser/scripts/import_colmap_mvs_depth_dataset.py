#!/usr/bin/env python3
"""Attach train-only COLMAP MVS camera-z depths to a Nerfstudio dataset.

The source RGB dataset is immutable. Images are linked, masks are rejected,
and held-out frames receive no depth even if a stray MVS file exists.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument(
        "--colmap-model",
        type=Path,
        default=None,
        help="Optional undistorted COLMAP sparse model supplying exact per-image MVS intrinsics/sizes.",
    )
    parser.add_argument(
        "--undistorted-images",
        type=Path,
        default=None,
        help="COLMAP undistorted image root paired with --colmap-model.",
    )
    return parser.parse_args()


def read_exact(stream, size: int) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError("Truncated COLMAP binary model")
    return payload


def load_binary_pinhole_calibration(model: Path) -> dict[str, dict[str, float | int]]:
    """Read the small cameras/images subset needed from a COLMAP binary model."""

    cameras: dict[int, dict[str, float | int]] = {}
    with (model / "cameras.bin").open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(count):
            camera_id, model_id = struct.unpack("<ii", read_exact(stream, 8))
            width, height = struct.unpack("<QQ", read_exact(stream, 16))
            if model_id == 1:  # COLMAP CameraModelId::kPinhole
                fx, fy, cx, cy = struct.unpack("<4d", read_exact(stream, 32))
            elif model_id == 4:  # Some 3.13 image_undistorter outputs retain zero-distortion OPENCV cameras.
                parameters = struct.unpack("<8d", read_exact(stream, 64))
                fx, fy, cx, cy = parameters[:4]
                if not np.allclose(parameters[4:], 0.0, rtol=0.0, atol=1e-12):
                    raise ValueError("COLMAP MVS OPENCV camera retains non-zero distortion")
            else:
                raise ValueError(f"Expected PINHOLE or zero-distortion OPENCV camera, got model id {model_id}")
            cameras[camera_id] = {
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "w": int(width),
                "h": int(height),
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
            }
    result: dict[str, dict[str, float | int]] = {}
    with (model / "images.bin").open("rb") as stream:
        count = struct.unpack("<Q", read_exact(stream, 8))[0]
        for _ in range(count):
            read_exact(stream, 4 + 7 * 8)  # image id, quaternion, translation
            camera_id = struct.unpack("<i", read_exact(stream, 4))[0]
            name_bytes = bytearray()
            while True:
                value = read_exact(stream, 1)
                if value == b"\0":
                    break
                name_bytes.extend(value)
            points = struct.unpack("<Q", read_exact(stream, 8))[0]
            stream.seek(points * 24, 1)  # x, y, point3D id
            if camera_id not in cameras:
                raise ValueError(f"COLMAP image references missing camera {camera_id}")
            result[normalized_name(name_bytes.decode("utf-8"))] = dict(cameras[camera_id])
    return result


def load_undistorted_calibration(model: Path) -> dict[str, dict[str, float | int]]:
    try:
        import pycolmap
    except ImportError:
        return load_binary_pinhole_calibration(model)
    reconstruction = pycolmap.Reconstruction(str(model))
    result: dict[str, dict[str, float | int]] = {}
    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name == "PINHOLE" and len(camera.params) == 4:
            fx, fy, cx, cy = [float(value) for value in camera.params]
        elif camera.model.name == "OPENCV" and len(camera.params) == 8 and np.allclose(
            camera.params[4:], 0.0, rtol=0.0, atol=1e-12
        ):
            fx, fy, cx, cy = [float(value) for value in camera.params[:4]]
        else:
            raise ValueError(
                f"Expected PINHOLE or zero-distortion OPENCV camera for {image.name}, got {camera.model.name}"
            )
        result[normalized_name(image.name)] = {
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": int(camera.width),
            "h": int(camera.height),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }
    return result


def link_image(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def main() -> int:
    args = parse_args()
    data = args.data.expanduser().resolve()
    depth_maps = args.depth_maps.expanduser().resolve()
    output = args.output.expanduser().resolve()
    colmap_model = None if args.colmap_model is None else args.colmap_model.expanduser().resolve()
    undistorted_images = (
        None if args.undistorted_images is None else args.undistorted_images.expanduser().resolve()
    )
    if (colmap_model is None) != (undistorted_images is None):
        raise ValueError("--colmap-model and --undistorted-images must be supplied together")
    if colmap_model is not None and (not colmap_model.is_dir() or not undistorted_images.is_dir()):
        raise FileNotFoundError("COLMAP model and undistorted image roots must exist")
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
    undistorted_calibration = (
        {} if colmap_model is None else load_undistorted_calibration(colmap_model)
    )
    output.mkdir(parents=True)
    depth_dir = output / "depth"
    preview_dir = output / "depth_previews"
    preview_dir.mkdir()
    coverages: list[float] = []
    medians: list[float] = []
    heldout_frames: list[dict[str, object]] = []
    depth_shape: tuple[int, int] | None = None
    depth_shapes: set[tuple[int, int]] = set()
    imported = 0
    depth_rows: list[dict[str, object]] = []
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
        if not valid.any():
            raise ValueError(f"Depth map has no finite positive values: {source}")
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        if colmap_model is None:
            if depth_shape is None:
                depth_shape = depth.shape
            elif depth.shape != depth_shape:
                raise ValueError(f"MVS depth shapes are not uniform: {depth.shape} != {depth_shape}")
        else:
            if name not in undistorted_calibration:
                raise ValueError(f"Undistorted COLMAP model has no train image {name!r}")
            calibration = undistorted_calibration[name]
            expected_shape = (int(calibration["h"]), int(calibration["w"]))
            if depth.shape != expected_shape:
                raise ValueError(f"Depth shape {depth.shape} does not match COLMAP camera {expected_shape}: {name}")
            frame.update(calibration)
        depth_shapes.add(depth.shape)
        target_name = f"mvs_{imported:05d}.npy.gz"
        target_path = depth_dir / target_name
        save_depth(target_path, depth)
        frame["depth_file_path"] = f"depth/{target_name}"
        coverage = float(valid.mean())
        median = float(np.median(depth[valid]))
        coverages.append(coverage)
        medians.append(median)
        depth_rows.append(
            {
                "image": name,
                "physical_camera": frame.get("physical_camera"),
                "source": str(source),
                "source_sha256": sha256(source),
                "output": str(target_path),
                "output_sha256": sha256(target_path),
                "shape": list(depth.shape),
                "coverage": coverage,
                "median_camera_z": median,
            }
        )
        if imported in {0, len(train_names) // 2, len(train_names) - 1}:
            save_preview(preview_dir / f"mvs_{imported:05d}.png", depth)
        imported += 1
    if imported != len(train_names):
        raise ValueError(f"Imported {imported} train depths but expected {len(train_names)}")
    if not depth_shapes:
        raise RuntimeError("No train depth shape was resolved")
    if heldout_frames:
        placeholder_by_shape: dict[tuple[int, int], str] = {}
        for frame in heldout_frames:
            name = normalized_name(str(frame["file_path"]))
            if undistorted_images is None:
                assert depth_shape is not None
                shape = depth_shape
            else:
                with Image.open(data / name) as image:
                    shape = (image.height, image.width)
            if shape not in placeholder_by_shape:
                placeholder_name = f"heldout_invalid_{shape[1]}x{shape[0]}.npy.gz"
                save_depth(depth_dir / placeholder_name, np.zeros(shape, dtype=np.float32))
                placeholder_by_shape[shape] = placeholder_name
            frame["depth_file_path"] = f"depth/{placeholder_by_shape[shape]}"
    if undistorted_images is None:
        images = data / "images"
        if not images.is_dir():
            raise FileNotFoundError(images)
        (output / "images").symlink_to(images, target_is_directory=True)
    else:
        for frame in frames:
            name = normalized_name(str(frame["file_path"]))
            source_root = undistorted_images if name in train_names else data
            link_image(source_root / name, output / name)
    payload["depth_unit_scale_factor"] = 1.0
    payload["colmap_mvs_depth"] = {
        "schema_version": 1,
        "source_data": str(data),
        "source_depth_maps": str(depth_maps),
        "input_type": args.input_type,
        "split": "train_filenames_only",
        "train_depth_count": imported,
        "heldout_invalid_depth_count": len(heldout_frames),
        "depth_shapes": [list(shape) for shape in sorted(depth_shapes)],
        "uses_undistorted_colmap_intrinsics": colmap_model is not None,
        "colmap_model": None if colmap_model is None else str(colmap_model),
        "undistorted_images": None if undistorted_images is None else str(undistorted_images),
        "coverage_mean": float(np.mean(coverages)),
        "coverage_min": float(np.min(coverages)),
        "median_camera_z_mean": float(np.mean(medians)),
        "masks": "forbidden",
        "depth_maps": depth_rows,
    }
    (output / "transforms.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(data / "transforms.json", output / "transforms.source.json")
    print(json.dumps(payload["colmap_mvs_depth"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
