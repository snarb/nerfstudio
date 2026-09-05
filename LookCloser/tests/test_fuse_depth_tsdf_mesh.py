from __future__ import annotations

import importlib.util
import gzip
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "fuse_depth_tsdf_mesh.py"
SPEC = importlib.util.spec_from_file_location("fuse_depth_tsdf_mesh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_nerfstudio_identity_camera_becomes_opencv_axis_flip() -> None:
    c2w = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    result = MODULE.nerfstudio_c2w_to_opencv_extrinsic(c2w)
    np.testing.assert_allclose(result, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_camera_conversion_preserves_world_camera_roundtrip() -> None:
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    world_to_cv = MODULE.nerfstudio_c2w_to_opencv_extrinsic(c2w)
    camera_origin_world = np.linalg.inv(world_to_cv) @ np.asarray([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(camera_origin_world[:3], [1.0, 2.0, 3.0])


def test_non_manifold_cleanup_is_default_on_and_can_be_disabled(tmp_path: Path) -> None:
    default = MODULE.parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "default.ply")])
    disabled = MODULE.parse_args(
        [
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "disabled.ply"),
            "--no-remove-non-manifold-edges",
        ]
    )

    assert default.remove_non_manifold_edges is True
    assert disabled.remove_non_manifold_edges is False
    assert not default.tensor_full_block_integration


def test_full_block_inventory_is_bounded_and_order_independent():
    blocks=[np.array([[0,0,0],[2,0,0],[-1,0,0],[20,0,0]],np.int32),np.array([[0,0,0],[1,0,0]],np.int32)]
    args=([0,0,0,1,1,1],.1)
    a=MODULE.bounded_union_block_coordinates(blocks,*args,block_resolution=4)
    b=MODULE.bounded_union_block_coordinates(blocks[::-1],*args,block_resolution=4)
    np.testing.assert_array_equal(a,b)
    assert len(a)==4  # Include the block touching the crop boundary at zero.
    assert not (a[:,0]>2).any()


def test_full_block_integration_accumulates_farther_view_free_space():
    import open3d as o3d
    def volume():return o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight'),attr_dtypes=(o3d.core.float32,o3d.core.float32),attr_channels=((1,),(1,)),voxel_size=.02,block_resolution=8,block_count=2048,device=o3d.core.Device('CPU:0'))
    K=o3d.core.Tensor(np.array([[64,0,32],[0,64,32],[0,0,1]],np.float64));E=o3d.core.Tensor(np.eye(4))
    depths=[o3d.t.geometry.Image(o3d.core.Tensor(np.full((64,64),z,np.float32))) for z in [.8,1.6]]
    legacy=volume();full=volume()
    coords=[legacy.compute_unique_block_coordinates(d,K,E,1.,3.,2.).numpy().copy() for d in depths]
    union=o3d.core.Tensor(np.unique(np.concatenate(coords),axis=0).astype(np.int32))
    for index in [0,0,1,1,1,1]:
        legacy.integrate(o3d.core.Tensor(coords[index]),depths[index],K,E,1.,3.,2.)
        full.integrate(union,depths[index],K,E,1.,3.,2.)
    def at_point(v):
        xyz,indices=v.voxel_coordinates_and_flattened_indices()
        point=np.argmin(np.linalg.norm(xyz.numpy()-[0,0,.8],axis=1));idx=int(indices.numpy()[point])
        return float(v.attribute('tsdf').reshape((-1,)).numpy()[idx]),float(v.attribute('weight').reshape((-1,)).numpy()[idx])
    old,new=at_point(legacy),at_point(full)
    assert old[1]==2 and new[1]==6
    assert abs(old[0])<1e-5 and new[0]>.6


def test_load_compressed_numpy_depth_applies_scene_scale(tmp_path: Path) -> None:
    source = tmp_path / "depth.npy.gz"
    with gzip.open(source, "wb") as stream:
        np.save(stream, np.asarray([[2.0, 0.0]], dtype=np.float32), allow_pickle=False)

    np.testing.assert_allclose(MODULE.load_depth(source, scale_factor=0.25), [[0.5, 0.0]])


def test_component_threshold_combines_absolute_and_relative_gates() -> None:
    counts = np.asarray([158_081, 228], dtype=np.int64)

    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=100, minimum_fraction=0.0
    ) == 100
    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=100, minimum_fraction=0.002
    ) == 317
    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=500, minimum_fraction=0.002
    ) == 500
