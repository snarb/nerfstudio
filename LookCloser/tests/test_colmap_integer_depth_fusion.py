from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from colmap_integer_depth_fusion import nearest_integer_lookup_intrinsics
from audit_depth_pixel_conventions import analytic_plane_depth, synthetic_plane


def test_nearest_lookup_preserves_calibration_input():
    k=np.array([[90., 0, 32], [0, 80, 24], [0, 0, 1]])
    before=k.copy(); adjusted=nearest_integer_lookup_intrinsics(k)
    np.testing.assert_array_equal(k, before)
    np.testing.assert_array_equal(adjusted-k, [[0, 0, .5], [0, 0, .5], [0, 0, 0]])
    projection=np.array([10.49, 10.51, 11.49, 11.51])
    np.testing.assert_array_equal((projection+.5).astype(int), [10, 11, 11, 12])


@pytest.mark.parametrize('k', [np.eye(4), np.full((3,3), np.nan), np.zeros((3,3))])
def test_invalid_calibration_fails(k):
    with pytest.raises(ValueError):nearest_integer_lookup_intrinsics(k)


def test_analytic_depth_is_camera_z_not_euclidean_range():
    k=np.array([[90., 0, 32], [0, 90, 32], [0, 0, 1]])
    depth=analytic_plane_depth(k, 64, 64, [0,0,1], .7)
    np.testing.assert_allclose(depth, .7)


def test_actual_cpu_volume_nearest_lookup_removes_signed_plane_bias():
    _, old=synthetic_plane('CPU:0', False)
    _, new=synthetic_plane('CPU:0', True)
    assert old['mean_plane_residual']>.001
    assert abs(new['mean_plane_residual'])<.0002
    assert new['plane_residual_rmse']<.7*old['plane_residual_rmse']


def test_wrapper_restores_open3d_factory_when_fusion_fails(tmp_path,monkeypatch):
    import open3d as o3d
    import fuse_depth_tsdf_mesh as fusion
    from run_colmap_integer_depth_fusion import main
    (tmp_path/'transforms.json').write_text('{}')
    original=o3d.t.geometry.VoxelBlockGrid
    def fail(_):
        assert o3d.t.geometry.VoxelBlockGrid is not original
        raise RuntimeError('Synthetic fusion failure')
    monkeypatch.setattr(fusion,'main',fail)
    with pytest.raises(RuntimeError,match='Synthetic fusion failure'):
        main(['--data',str(tmp_path),'--output',str(tmp_path/'mesh.ply'),'--backend','tensor',
              '--tensor-full-block-integration','--crop-aabb','-1','-1','-1','1','1','1'])
    assert o3d.t.geometry.VoxelBlockGrid is original
    assert (tmp_path/'mesh.integer_depth_request.json').is_file()
    assert not (tmp_path/'mesh.integer_depth_manifest.json').exists()


def test_wrapper_rejects_legacy_backend_before_output(tmp_path):
    from run_colmap_integer_depth_fusion import main
    with pytest.raises(ValueError,match='requires tensor full-block'):
        main(['--data',str(tmp_path),'--output',str(tmp_path/'mesh.ply')])
    assert not (tmp_path/'mesh.integer_depth_request.json').exists()
