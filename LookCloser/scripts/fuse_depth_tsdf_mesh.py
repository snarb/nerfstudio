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
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

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


def load_depth(path: Path, *, scale_factor: float) -> np.ndarray:
    """Load depth without decoding RGB or depending on Nerfstudio dataset internals."""

    suffixes = path.suffixes
    if suffixes[-2:] == [".npy", ".gz"]:
        with gzip.open(path, "rb") as stream:
            depth = np.load(stream, allow_pickle=False)
    elif path.suffix == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as image:
            depth = np.asarray(image)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected one HW depth plane, got {depth.shape}: {path}")
    return depth * float(scale_factor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--additional-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional calibrated depth dataset to integrate into the same TSDF. "
            "Normalization must match --data; this supports general multiscale PatchMatch fusion."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--center-method", default="focus")
    add_boolean_argument(parser, "--auto-scale-poses", default=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--depth-unit-scale-factor", type=float, default=1.0)
    parser.add_argument("--voxel-length", type=float, default=0.002)
    parser.add_argument("--sdf-trunc", type=float, default=0.008)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument('--tensor-full-block-integration',action='store_true',
                        help='Opt-in two-pass fusion: integrate every view into the bounded union of surface blocks, including observed free space.')
    parser.add_argument(
        "--backend",
        choices=("legacy", "tensor"),
        default="legacy",
        help="Open3D integration backend. Legacy preserves historical output; tensor enables opt-in CUDA fusion.",
    )
    parser.add_argument(
        "--device",
        default="CUDA:0",
        help="Open3D tensor device used only with --backend tensor (for example CUDA:0 or CPU:0).",
    )
    parser.add_argument(
        "--tensor-block-count",
        type=int,
        default=200000,
        help="Maximum allocated sparse blocks for the opt-in tensor backend.",
    )
    parser.add_argument(
        "--tensor-weight-threshold",
        type=float,
        default=1.0,
        help="Minimum integrated weight for tensor-backend marching cubes.",
    )
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
    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=0.0,
        help=(
            "Also remove islands smaller than this fraction of the largest connected component. "
            "Zero preserves the historical absolute-threshold behavior."
        ),
    )
    add_boolean_argument(
        parser,
        "--remove-non-manifold-edges",
        default=True,
        help=(
            "Apply Open3D's legacy non-manifold-edge cleanup before cropping. "
            "The default preserves historical fusion behavior; disable it for causal hole diagnostics."
        ),
    )
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.additional_data = [path.expanduser().resolve() for path in args.additional_data]
    args.output = args.output.expanduser().resolve()
    for data in [args.data, *args.additional_data]:
        if not data.is_dir():
            parser.error(f"Dataset does not exist: {data}")
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
        args.tensor_block_count,
        args.tensor_weight_threshold,
    )
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
        parser.error("scale, interval, voxel, truncation, and depth values must be finite and positive")
    if args.sdf_trunc < args.voxel_length:
        parser.error("--sdf-trunc must be at least one voxel")
    if args.min_component_triangles < 0:
        parser.error("--min-component-triangles must be non-negative")
    if not math.isfinite(args.min_component_fraction) or not 0.0 <= args.min_component_fraction <= 1.0:
        parser.error("--min-component-fraction must be finite and between zero and one")
    if args.crop_aabb is not None:
        bounds = np.asarray(args.crop_aabb, dtype=np.float64).reshape(2, 3)
        if not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
            parser.error("--crop-aabb must contain finite increasing bounds")
    if args.tensor_full_block_integration and (args.backend!='tensor' or args.crop_aabb is None):
        parser.error('Full-block integration requires tensor backend and explicit bounded crop')
    return args


def bounded_union_block_coordinates(blocks,crop_aabb,voxel_length,*,block_resolution=16,padding=0.):
    """Deterministic block inventory; crop affects allocation, not depth evidence."""
    bounds=np.asarray(crop_aabb,dtype=float).reshape(2,3)
    coords=np.unique(np.concatenate(blocks,axis=0),axis=0).astype(np.int32)
    size=float(voxel_length)*block_resolution
    lo=coords*size;hi=lo+size
    keep=(hi>=bounds[0]-padding).all(-1)&(lo<=bounds[1]+padding).all(-1)
    return np.ascontiguousarray(coords[keep])


def component_triangle_threshold(
    counts: np.ndarray,
    *,
    minimum_triangles: int,
    minimum_fraction: float,
) -> int:
    """Return a scale-aware island threshold without weakening the absolute gate."""

    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts <= 0):
        raise ValueError("component triangle counts must be a non-empty positive vector")
    relative = int(math.ceil(float(counts.max()) * float(minimum_fraction)))
    return max(int(minimum_triangles), relative)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for TSDF mesh fusion") from error

    groups = []
    primary_outputs = None
    for data in [args.data, *args.additional_data]:
        parser_config = NerfstudioDataParserConfig(
            data=data,
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
            raise ValueError(f"Train split does not provide depth_file_path values: {data}")
        depth_filenames = outputs.metadata["depth_filenames"]
        if len(depth_filenames) != len(outputs.image_filenames):
            raise ValueError(f"Train depth and image counts differ: {data}")
        if primary_outputs is None:
            primary_outputs = outputs
        elif (
            not math.isclose(
                float(outputs.dataparser_scale),
                float(primary_outputs.dataparser_scale),
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
            or not np.allclose(
                outputs.dataparser_transform.detach().cpu().numpy(),
                primary_outputs.dataparser_transform.detach().cpu().numpy(),
                rtol=1e-7,
                atol=1e-8,
            )
        ):
            raise ValueError("All TSDF depth datasets must resolve the same dataparser normalization")
        depth_scale = float(outputs.metadata.get("depth_unit_scale_factor", 1.0)) * float(
            outputs.dataparser_scale
        )
        groups.append((data, outputs, depth_filenames, depth_scale))
    assert primary_outputs is not None

    volume = None
    tensor_volume = None
    tensor_device = None
    trunc_voxel_multiplier = float(args.sdf_trunc / args.voxel_length)
    if args.backend == "legacy":
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(args.voxel_length),
            sdf_trunc=float(args.sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
    else:
        tensor_device = o3d.core.Device(args.device)
        if tensor_device.get_type() == o3d.core.Device.DeviceType.CUDA and not o3d.core.cuda.is_available():
            raise RuntimeError("Open3D tensor CUDA backend was requested but CUDA is unavailable")
        tensor_volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,)),
            voxel_size=float(args.voxel_length),
            block_resolution=16,
            block_count=int(args.tensor_block_count),
            device=tensor_device,
        )
    rows: list[dict[str, object]] = []
    total_images = sum(len(depth_filenames) for _, _, depth_filenames, _ in groups)
    full_block_coords=None
    full_block_stats=None
    if args.tensor_full_block_integration:
        assert tensor_volume is not None and tensor_device is not None
        candidates=[];discovered=0
        for data,outputs,depth_filenames,depth_scale in groups:
            cameras=outputs.cameras.to('cpu')
            for image_index,depth_path in enumerate(depth_filenames):
                depth=load_depth(Path(depth_path),scale_factor=depth_scale)
                valid=np.isfinite(depth)&(depth>0)&(depth<args.depth_trunc)
                if not valid.any():raise ValueError('No valid depth in full-block discovery')
                depth=np.where(valid,depth,0).astype(np.float32)
                intrinsic=np.array([[float(cameras.fx[image_index]),0,float(cameras.cx[image_index])],
                                    [0,float(cameras.fy[image_index]),float(cameras.cy[image_index])],[0,0,1]],np.float64)
                extrinsic=nerfstudio_c2w_to_opencv_extrinsic(cameras.camera_to_worlds[image_index].numpy())
                coords=tensor_volume.compute_unique_block_coordinates(
                    o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth),device=tensor_device)),
                    o3d.core.Tensor(intrinsic),o3d.core.Tensor(extrinsic),depth_scale=1.,
                    depth_max=float(args.depth_trunc),trunc_voxel_multiplier=trunc_voxel_multiplier)
                candidates.append(coords.cpu().numpy().copy())
                discovered+=1;print(f'discovered_blocks={discovered}/{total_images}',flush=True)
        union=bounded_union_block_coordinates(candidates,args.crop_aabb,args.voxel_length,padding=args.sdf_trunc)
        if not len(union):raise RuntimeError('Bounded full-block inventory is empty')
        full_block_coords=o3d.core.Tensor(union,device=tensor_device)
        full_block_stats={'allocated_block_count':len(union),'block_coordinate_sha256':hashlib.sha256(union.tobytes()).hexdigest(),
                          'allocation_crop_padding':args.sdf_trunc,'each_view_updates_entire_union':True,
                          'raw_volume_serialized':False}
        print('full_block_inventory='+json.dumps(full_block_stats),flush=True)
    integrated = 0
    for data_index, (data, outputs, depth_filenames, depth_scale) in enumerate(groups):
        cameras = outputs.cameras.to("cpu")
        for image_index, depth_path in enumerate(depth_filenames):
            depth_array = load_depth(Path(depth_path), scale_factor=depth_scale)
            valid = np.isfinite(depth_array) & (depth_array > 0) & (depth_array < args.depth_trunc)
            if not valid.any():
                raise ValueError(f"Train depth {image_index} has no finite positive values below depth_trunc")
            depth_array = np.where(valid, depth_array, 0.0).astype(np.float32, copy=False)
            height, width = depth_array.shape
            extrinsic = nerfstudio_c2w_to_opencv_extrinsic(cameras.camera_to_worlds[image_index].numpy())
            intrinsic_array = np.asarray(
                [
                    [float(cameras.fx[image_index].item()), 0.0, float(cameras.cx[image_index].item())],
                    [0.0, float(cameras.fy[image_index].item()), float(cameras.cy[image_index].item())],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            if args.backend == "legacy":
                assert volume is not None
                # The downstream renderer samples calibrated source images rather
                # than vertex colours; a neutral image avoids needless RGB I/O.
                color_array = np.zeros((height, width, 3), dtype=np.uint8)
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width,
                    height,
                    intrinsic_array[0, 0],
                    intrinsic_array[1, 1],
                    intrinsic_array[0, 2],
                    intrinsic_array[1, 2],
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(color_array)),
                    o3d.geometry.Image(np.ascontiguousarray(depth_array)),
                    depth_scale=1.0,
                    depth_trunc=float(args.depth_trunc),
                    convert_rgb_to_intensity=False,
                )
                volume.integrate(rgbd, intrinsic, extrinsic)
            else:
                assert tensor_volume is not None and tensor_device is not None
                depth_image = o3d.t.geometry.Image(
                    o3d.core.Tensor(np.ascontiguousarray(depth_array), device=tensor_device)
                )
                # Open3D's CUDA VBG API keeps camera matrices on CPU while the
                # depth image and sparse volume live on the selected device.
                intrinsic_tensor = o3d.core.Tensor(intrinsic_array)
                extrinsic_tensor = o3d.core.Tensor(extrinsic)
                block_coords = full_block_coords
                if block_coords is None:
                    block_coords = tensor_volume.compute_unique_block_coordinates(
                        depth_image,
                        intrinsic_tensor,
                        extrinsic_tensor,
                        depth_scale=1.0,
                        depth_max=float(args.depth_trunc),
                        trunc_voxel_multiplier=trunc_voxel_multiplier,
                    )
                tensor_volume.integrate(
                    block_coords,
                    depth_image,
                    intrinsic_tensor,
                    extrinsic_tensor,
                    depth_scale=1.0,
                    depth_max=float(args.depth_trunc),
                    trunc_voxel_multiplier=trunc_voxel_multiplier,
                )
            integrated += 1
            rows.append(
                {
                    "data_index": data_index,
                    "data": str(data),
                    "image_index": image_index,
                    "image": str(outputs.image_filenames[image_index]),
                    "valid_depth_fraction": float(valid.mean()),
                    "valid_depth_median": float(np.median(depth_array[valid])),
                }
            )
            print(f"integrated={integrated}/{total_images}", flush=True)

    if args.backend == "legacy":
        assert volume is not None
        mesh = volume.extract_triangle_mesh()
    else:
        assert tensor_volume is not None
        mesh = tensor_volume.extract_triangle_mesh(
            weight_threshold=float(args.tensor_weight_threshold)
        ).cpu().to_legacy()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF fusion produced an empty mesh")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    triangles_before_non_manifold_cleanup = len(mesh.triangles)
    if args.remove_non_manifold_edges:
        mesh.remove_non_manifold_edges()
    triangles_after_non_manifold_cleanup = len(mesh.triangles)
    if args.crop_aabb is not None:
        bounds = np.asarray(args.crop_aabb, dtype=np.float64).reshape(2, 3)
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bounds[0], bounds[1]))
    triangles_before_components = len(mesh.triangles)
    removed_components = 0
    effective_component_threshold = 0
    if (args.min_component_triangles > 0 or args.min_component_fraction > 0.0) and triangles_before_components > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_array = np.asarray(labels, dtype=np.int64)
        counts_array = np.asarray(counts, dtype=np.int64)
        effective_component_threshold = component_triangle_threshold(
            counts_array,
            minimum_triangles=args.min_component_triangles,
            minimum_fraction=args.min_component_fraction,
        )
        remove = counts_array[labels_array] < effective_component_threshold
        removed_components = int(np.sum(counts_array < effective_component_threshold))
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError("Cropping/component filtering removed the complete TSDF mesh")
    _, final_component_counts, _ = mesh.cluster_connected_triangles()
    final_component_triangles = sorted(
        (int(value) for value in np.asarray(final_component_counts)), reverse=True
    )
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
        "additional_data": [str(path) for path in args.additional_data],
        "additional_source_transforms_sha256": [
            sha256(path / "transforms.json") for path in args.additional_data
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "masks": False,
        "train_image_count": total_images,
        "dataparser_scale": float(primary_outputs.dataparser_scale),
        "dataparser_transform": primary_outputs.dataparser_transform.tolist(),
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
            "backend": args.backend,
            "device": args.device if args.backend == "tensor" else "CPU",
            "tensor_block_count": args.tensor_block_count if args.backend == "tensor" else None,
            "tensor_weight_threshold": args.tensor_weight_threshold if args.backend == "tensor" else None,
            "tensor_full_block_integration":args.tensor_full_block_integration,
            "full_block_inventory":full_block_stats,
            "crop_aabb": args.crop_aabb,
            "min_component_triangles": args.min_component_triangles,
            "min_component_fraction": args.min_component_fraction,
            "effective_component_triangle_threshold": effective_component_threshold,
            "remove_non_manifold_edges": args.remove_non_manifold_edges,
        },
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "triangles_before_non_manifold_cleanup": triangles_before_non_manifold_cleanup,
        "triangles_after_non_manifold_cleanup": triangles_after_non_manifold_cleanup,
        "triangles_before_component_filter": triangles_before_components,
        "removed_small_components": removed_components,
        "connected_components": len(final_component_triangles),
        "component_triangles": final_component_triangles,
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
