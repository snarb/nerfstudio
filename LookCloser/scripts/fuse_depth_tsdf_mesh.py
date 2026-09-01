#!/usr/bin/env python3
"""Fuse calibrated train-only camera-z depth into one Open3D TSDF mesh.

The script resolves camera transforms, intrinsics, depth units, orientation,
centering, and pose scale through Nerfstudio's dataparser. It does not consume
or create image/person masks. Unlike a union of back-projected points, TSDF
fusion contributes signed free-space observations before every measured
surface and extracts one zero crossing shared by the training cameras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--auto-scale-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--depth-unit-scale-factor", type=float, default=1.0)
    parser.add_argument("--voxel-length", type=float, default=0.002)
    parser.add_argument("--sdf-trunc", type=float, default=0.008)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument(
        "--crop-aabb",
        type=float,
        nargs=6,
        default=None,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
    )
    parser.add_argument(
        "--min-component-triangles",
        type=int,
        default=100,
        help="Remove disconnected surface islands smaller than this many triangles; zero keeps all islands.",
    )
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.data.is_dir():
        parser.error(f"Dataset does not exist: {args.data}")
    if args.output.suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    positive = (
        args.eval_interval,
        args.scale_factor,
        args.scene_scale,
        args.downscale_factor,
        args.depth_unit_scale_factor,
        args.voxel_length,
        args.sdf_trunc,
        args.depth_trunc,
    )
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
        parser.error("scale, interval, voxel, truncation, and depth values must be finite and positive")
    if args.sdf_trunc < args.voxel_length:
        parser.error("--sdf-trunc must be at least one voxel")
    if args.min_component_triangles < 0:
        parser.error("--min-component-triangles must be non-negative")
    if args.crop_aabb is not None:
        bounds = np.asarray(args.crop_aabb, dtype=np.float64).reshape(2, 3)
        if not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
            parser.error("--crop-aabb must contain finite increasing bounds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for TSDF mesh fusion") from error

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
        depth_unit_scale_factor=args.depth_unit_scale_factor,
        load_3D_points=False,
    )
    outputs = parser_config.setup().get_dataparser_outputs(split="train")
    if outputs.mask_filenames is not None:
        raise ValueError("TSDF diagnostic forbids image/person masks")
    if outputs.metadata.get("depth_filenames") is None:
        raise ValueError("Train split does not provide depth_file_path values")
    dataset = DepthDataset(outputs)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_length),
        sdf_trunc=float(args.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    cameras = outputs.cameras.to("cpu")
    rows: list[dict[str, object]] = []
    for image_index in range(len(dataset)):
        sample = dataset[image_index]
        image = sample["image"].detach().cpu().numpy()
        depth = sample.get("depth_image")
        if depth is None:
            raise RuntimeError(f"Missing depth for train image {image_index}")
        depth_array = depth.detach().cpu().numpy().squeeze(-1).astype(np.float32, copy=False)
        valid = np.isfinite(depth_array) & (depth_array > 0) & (depth_array < args.depth_trunc)
        if not valid.any():
            raise ValueError(f"Train depth {image_index} has no finite positive values below depth_trunc")
        depth_array = np.where(valid, depth_array, 0.0).astype(np.float32, copy=False)
        color_array = np.clip(np.rint(image[..., :3] * 255.0), 0, 255).astype(np.uint8)
        height, width = depth_array.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(cameras.fx[image_index].item()),
            float(cameras.fy[image_index].item()),
            float(cameras.cx[image_index].item()),
            float(cameras.cy[image_index].item()),
        )
        extrinsic = nerfstudio_c2w_to_opencv_extrinsic(cameras.camera_to_worlds[image_index].numpy())
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_array)),
            o3d.geometry.Image(np.ascontiguousarray(depth_array)),
            depth_scale=1.0,
            depth_trunc=float(args.depth_trunc),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic)
        rows.append(
            {
                "image_index": image_index,
                "image": str(outputs.image_filenames[image_index]),
                "valid_depth_fraction": float(valid.mean()),
                "valid_depth_median": float(np.median(depth_array[valid])),
            }
        )
        print(f"integrated={image_index + 1}/{len(dataset)}", flush=True)

    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    if args.crop_aabb is not None:
        bounds = np.asarray(args.crop_aabb, dtype=np.float64).reshape(2, 3)
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bounds[0], bounds[1]))
    triangles_before_components = len(mesh.triangles)
    removed_components = 0
    if args.min_component_triangles > 0 and triangles_before_components > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_array = np.asarray(labels, dtype=np.int64)
        counts_array = np.asarray(counts, dtype=np.int64)
        remove = counts_array[labels_array] < args.min_component_triangles
        removed_components = int(np.sum(counts_array < args.min_component_triangles))
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError("Cropping/component filtering removed the complete TSDF mesh")
    mesh.compute_vertex_normals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write mesh: {args.output}")

    transforms = args.data / "transforms.json"
    metadata = {
        "schema_version": 1,
        "method": "open3d_scalable_tsdf_train_only",
        "data": str(args.data),
        "source_transforms_sha256": sha256(transforms),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "masks": False,
        "train_image_count": len(dataset),
        "dataparser_scale": float(outputs.dataparser_scale),
        "dataparser_transform": outputs.dataparser_transform.tolist(),
        "parameters": {
            "eval_mode": args.eval_mode,
            "eval_interval": args.eval_interval,
            "orientation_method": args.orientation_method,
            "center_method": args.center_method,
            "auto_scale_poses": args.auto_scale_poses,
            "scale_factor": args.scale_factor,
            "scene_scale": args.scene_scale,
            "downscale_factor": args.downscale_factor,
            "depth_unit_scale_factor": args.depth_unit_scale_factor,
            "voxel_length": args.voxel_length,
            "sdf_trunc": args.sdf_trunc,
            "depth_trunc": args.depth_trunc,
            "crop_aabb": args.crop_aabb,
            "min_component_triangles": args.min_component_triangles,
        },
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "triangles_before_component_filter": triangles_before_components,
        "removed_small_components": removed_components,
        "images": rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"complete vertices={len(mesh.vertices)} triangles={len(mesh.triangles)} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
