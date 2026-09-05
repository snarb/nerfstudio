from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tsdf_free_space_veto import free_space_evidence, apply_veto, constrain_volume
from carve_patchmatch_mesh_free_space import free_space_evidence as numpy_evidence


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_native_evidence_matches_audited_numpy(device):
    rng = np.random.default_rng(9)
    depth = np.ones((35, 40), np.float32)
    depth[:10] = 0; depth[10:17, :20] = 1.1; depth[20:22] = np.nan
    u = rng.uniform(-5, 44, 1000).astype(np.float32)
    v = rng.uniform(-5, 39, 1000).astype(np.float32)
    z = rng.choice(np.array([.9, 1., 1.1, 0., np.nan], np.float32), 1000)
    expected, _ = numpy_evidence(depth, u, v, z)
    actual = free_space_evidence(*(torch.as_tensor(x, device=device) for x in (depth, u, v, z)))
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_veto_preserves_unknown_and_unqualified_voxels_and_all_weights():
    tsdf = torch.tensor([-.5, -.2, -.3, -.4, .1]); weight = torch.tensor([2., 2., 0., 4., 3.])
    before = weight.clone()
    result = apply_veto(tsdf, weight, torch.tensor([3, 2, 1, 0, 4]), torch.tensor([3, 9, 2, 4, 3]), 3)
    torch.testing.assert_close(tsdf, torch.tensor([1., -.2, -.3, 1., 1.]))
    torch.testing.assert_close(weight, before)
    assert result['negative_voxels_changed'] == 2


def test_pre_extraction_veto_removes_contradicted_slab_not_supported_plane():
    import open3d as o3d
    volume = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32), attr_channels=((1,), (1,)),
        voxel_size=.02, block_resolution=8, block_count=2048, device=o3d.core.Device('CPU:0'))
    K = np.array([[64, 0, 32], [0, 64, 32], [0, 0, 1]], np.float64); E = np.eye(4)
    raw = [np.full((64, 64), z, np.float32) for z in [.8, 1.6]]
    depths = [o3d.t.geometry.Image(o3d.core.Tensor(d)) for d in raw]
    coords = [volume.compute_unique_block_coordinates(d, o3d.core.Tensor(K), o3d.core.Tensor(E), 1., 3., 2.).numpy() for d in depths]
    union = o3d.core.Tensor(np.unique(np.concatenate(coords), axis=0).astype(np.int32))
    # A high multiplicity of contradictory near observations retains a false slab
    # under weighted averaging. Three far observations veto that empty space.
    for i in [0]*12+[1]*3:
        volume.integrate(union, depths[i], o3d.core.Tensor(K), o3d.core.Tensor(E), 1., 3., 2.)
    xyz, ids = volume.voxel_coordinates_and_flattened_indices()
    points = xyz.numpy(); indices = ids.numpy()
    near = int(indices[np.argmin(np.linalg.norm(points-[0, 0, .82], axis=1))])
    far = int(indices[np.argmin(np.linalg.norm(points-[0, 0, 1.62], axis=1))])
    attr = volume.attribute('tsdf').reshape((-1,)).numpy()
    assert attr[near] < 0 and attr[far] < 0
    far_before = float(attr[far])
    result = constrain_volume(volume, ((d, K, E, {'i': i}) for i, d in enumerate([raw[1]]*3)),
                              batch_size=8192)
    assert attr[near] == 1 and attr[far] == far_before
    assert result['negative_voxels_changed'] > 0
