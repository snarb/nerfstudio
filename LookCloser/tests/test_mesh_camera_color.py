from pathlib import Path
import sys
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_camera_color import MeshCameraColor,fit_mesh_camera_gains


def mesh_fixture():
    y,x=np.mgrid[:8,:10];vertices=np.c_[x.ravel(),y.ravel(),np.zeros(80)]*.0005
    triangles=[]
    for j in range(7):
        for i in range(9):
            k=j*10+i;triangles.extend([[k,k+1,k+10],[k+1,k+11,k+10]])
    return vertices,np.array(triangles)


def test_shared_high_frequency_albedo_is_not_a_camera_gain():
    vertices,triangles=mesh_fixture();rgb=torch.randn((1,80,3),dtype=torch.float64).repeat(3,1,1)
    gains,stats=fit_mesh_camera_gains(rgb,torch.ones((3,80),dtype=torch.bool),torch.zeros(80,dtype=torch.bool),vertices,triangles,smoothness=1.)
    assert gains.abs().max()<1e-7 and stats['converged']


def test_camera_response_bias_is_recovered_without_primary_exception():
    vertices,triangles=mesh_fixture();shared=torch.randn((1,80,3),dtype=torch.float64)
    bias=torch.tensor([[-.2,.1,.1],[.2,-.1,0],[0,0,-.1]],dtype=torch.float64)
    rgb=shared+bias[:,None]
    gains,stats=fit_mesh_camera_gains(rgb,torch.ones((3,80),dtype=torch.bool),torch.arange(80)%5==0,vertices,triangles,smoothness=1.,ridge=1e-5)
    assert stats['converged'] and stats['maximum_abs_mean_camera_log_gain']<1e-10
    torch.testing.assert_close(gains.double(),-bias[:,None].expand_as(rgb),atol=1e-5,rtol=0)


def test_hidden_and_held_rgb_do_not_influence_gain_fit():
    vertices,triangles=mesh_fixture();rng=torch.Generator().manual_seed(7)
    rgb=torch.randn((3,80,3),generator=rng,dtype=torch.float64)*.1
    visible=torch.ones((3,80),dtype=torch.bool);visible[1,:20]=False
    held=torch.arange(80)%5==0
    a,_=fit_mesh_camera_gains(rgb,visible,held,vertices,triangles,smoothness=1.)
    changed=rgb.clone();changed[~visible|held[None]]=30
    b,_=fit_mesh_camera_gains(changed,visible,held,vertices,triangles,smoothness=1.)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_rigid_world_motion_preserves_camera_gain_fields():
    vertices,triangles=mesh_fixture();rgb=torch.rand((3,80,3),dtype=torch.float64)
    visible=torch.ones((3,80),dtype=torch.bool);held=torch.zeros(80,dtype=torch.bool)
    a,_=fit_mesh_camera_gains(rgb,visible,held,vertices,triangles,smoothness=1.)
    rotated=vertices[:,[2,1,0]]+[2,3,4]
    b,_=fit_mesh_camera_gains(rgb,visible,held,rotated,triangles,smoothness=1.)
    torch.testing.assert_close(a,b,atol=2e-7,rtol=0)


def test_nonfinite_solver_inputs_fail_closed():
    vertices,triangles=mesh_fixture();rgb=torch.zeros((3,80,3));rgb[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):fit_mesh_camera_gains(rgb,torch.ones((3,80),dtype=torch.bool),torch.zeros(80,dtype=torch.bool),vertices,triangles)


def test_uneven_camera_visibility_preserves_zero_mean_gauge_and_true_residual():
    vertices,triangles=mesh_fixture();rng=torch.Generator().manual_seed(19)
    rgb=torch.rand((12,80,3),generator=rng,dtype=torch.float64)*.2
    visible=torch.rand((12,80),generator=rng)>.4;held=torch.arange(80)%5==0
    gains,stats=fit_mesh_camera_gains(rgb,visible,held,vertices,triangles,smoothness=64.,iterations=4096)
    assert stats['converged'] and stats['max_relative_residual']<5e-7
    assert stats['maximum_abs_mean_camera_log_gain']<1e-12
    assert gains.mean(0).abs().max()<1e-8


@pytest.fixture
def field_files(tmp_path):
    import json
    o3d=pytest.importorskip('open3d')
    from colmap_patchmatch_tsdf_campaign_common import sha256
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[0,1,0]])
    mesh.triangles=o3d.utility.Vector3iVector([[0,1,2]])
    mesh_path=tmp_path/'mesh.ply';o3d.io.write_triangle_mesh(str(mesh_path),mesh)
    calibration=tmp_path/'calibration.json';calibration.write_text('{}')
    coefficients=tmp_path/'gains.npz'
    np.savez_compressed(coefficients,log_gains=np.array([[[0,0,0],[.2,0,0],[0,.4,0]]],np.float32))
    manifest=tmp_path/'manifest.json'
    payload=dict(uses_eval_rgb=False,uses_semantic_masks=False,output_source_averaging=False,
                 mesh_sha256=sha256(mesh_path),camera_color_calibration_sha256=sha256(calibration),
                 camera_color_model='spatial',source_cameras=['train_A'],coefficients=coefficients.name,
                 coefficients_sha256=sha256(coefficients),fit={'converged':True})
    manifest.write_text(json.dumps(payload))
    return manifest,mesh_path,calibration,payload


def test_mesh_attached_field_barycentric_sampling_distance_and_support(field_files):
    manifest,mesh,calibration,_=field_files
    field=MeshCameraColor(manifest,mesh,calibration,'spatial')
    world=np.array([[[.25,.5,0],[.25,.5,.002],[.25,.5,0]]])
    field.bind(world,np.array([[True,True,False]]))
    np.testing.assert_allclose(field.sample('train_A'),[[[.05,.2,0],[0,0,0],[0,0,0]]],atol=1e-7)
    field.bind(world,np.zeros((1,3),bool))
    np.testing.assert_array_equal(field.sample('train_A'),np.zeros((1,3,3)))


def test_mesh_color_coefficient_checksum_fails_closed(field_files):
    manifest,mesh,calibration,payload=field_files
    (manifest.parent/payload['coefficients']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        MeshCameraColor(manifest,mesh,calibration,'spatial')


@pytest.mark.parametrize('change',[
    {'uses_eval_rgb':True},{'uses_semantic_masks':True},{'output_source_averaging':True},
    {'mesh_sha256':'wrong'},{'camera_color_calibration_sha256':'wrong'},
    {'camera_color_model':'rgb'},{'source_cameras':['F004_B005_1210O9']},
    {'source_cameras':['train_A','train_A']},{'fit':{'converged':False}},
])
def test_mesh_color_provenance_inventory_and_solver_guards(field_files,change):
    import json
    manifest,mesh,calibration,payload=field_files
    manifest.write_text(json.dumps({**payload,**change}))
    with pytest.raises(ValueError):MeshCameraColor(manifest,mesh,calibration,'spatial')
