from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from canonical_surface_base import fit_surface_bases


def fixture():
    y,x=np.mgrid[:6,:8];vertices=np.c_[x.ravel(),y.ravel(),np.zeros(48)]*.0005
    triangles=[]
    for j in range(5):
        for i in range(7):
            k=j*8+i;triangles.extend([[k,k+1,k+8],[k+1,k+9,k+8]])
    return vertices,np.asarray(triangles)


def test_constant_camera_biases_map_to_common_base():
    v,t=fixture();rgb=torch.tensor([.2,.4,.6],dtype=torch.float64)[:,None,None].expand(3,48,3)
    bases,common,stats=fit_surface_bases(rgb,torch.ones((3,48),dtype=torch.bool),v,t,smoothness=1.)
    assert stats['converged']
    torch.testing.assert_close(bases,rgb.float(),rtol=0,atol=1e-6)
    torch.testing.assert_close(common,torch.full((48,3),.4),rtol=0,atol=1e-6)


def test_identical_high_frequency_sources_keep_every_detail():
    v,t=fixture();rgb=torch.rand((1,48,3),generator=torch.Generator().manual_seed(2)).repeat(3,1,1)
    bases,common,stats=fit_surface_bases(rgb,torch.ones((3,48),dtype=torch.bool),v,t,smoothness=2.)
    assert stats['converged'];torch.testing.assert_close(common[None]-bases,torch.zeros_like(bases),atol=2e-7,rtol=0)


def test_hidden_rgb_cannot_change_common_or_source_bases():
    v,t=fixture();rgb=torch.rand((3,48,3),generator=torch.Generator().manual_seed(3));visible=torch.ones((3,48),dtype=torch.bool)
    visible[1,:20]=False;a,b,_=fit_surface_bases(rgb,visible,v,t,smoothness=1.)
    changed=rgb.clone();changed[~visible]=float('nan');c,d,_=fit_surface_bases(changed,visible,v,t,smoothness=1.)
    torch.testing.assert_close(a,c,rtol=0,atol=0);torch.testing.assert_close(b,d,rtol=0,atol=0)


def test_source_permutation_keeps_the_common_base():
    v,t=fixture();rgb=torch.rand((3,48,3),generator=torch.Generator().manual_seed(4));visible=torch.ones((3,48),dtype=torch.bool)
    a,b,_=fit_surface_bases(rgb,visible,v,t,smoothness=1.);order=[2,0,1]
    c,d,_=fit_surface_bases(rgb[order],visible[order],v,t,smoothness=1.)
    torch.testing.assert_close(c,a[order],atol=1e-7,rtol=0);torch.testing.assert_close(d,b,atol=1e-7,rtol=0)


@pytest.mark.parametrize('setting',[{'smoothness':0},{'ridge':float('nan')},{'iterations':0}])
def test_bad_solver_settings_reject(setting):
    v,t=fixture()
    with pytest.raises(ValueError):fit_surface_bases(torch.ones((2,48,3))*.5,torch.ones((2,48),dtype=torch.bool),v,t,**setting)


def test_unobserved_camera_rejected():
    v,t=fixture();visible=torch.ones((2,48),dtype=torch.bool);visible[1]=False
    with pytest.raises(ValueError):fit_surface_bases(torch.ones((2,48,3))*.5,visible,v,t)


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
    coefficients=tmp_path/'bases.npz'
    np.savez_compressed(coefficients,source_bases=np.full((1,3,3),.2,np.float32),
        common_base=np.array([[.2,.2,.2],[.4,.2,.2],[.2,.6,.2]],np.float32))
    manifest=tmp_path/'manifest.json'
    payload=dict(uses_eval_rgb=False,uses_semantic_masks=False,query_dependent_base=False,
        mesh_sha256=sha256(mesh_path),camera_color_calibration_sha256=sha256(calibration),
        camera_color_model='spatial',source_cameras=['train_A'],coefficients=coefficients.name,
        coefficients_sha256=sha256(coefficients),fit={'converged':True})
    manifest.write_text(json.dumps(payload))
    return manifest,mesh_path,calibration,payload


def test_base_offset_barycentric_distance_and_support(field_files):
    from canonical_surface_base import CanonicalSurfaceBase
    manifest,mesh,calibration,_=field_files;field=CanonicalSurfaceBase(manifest,mesh,calibration)
    world=np.array([[[.25,.5,0],[.25,.5,.002],[.25,.5,0]]])
    field.bind(world,np.array([[True,True,False]]))
    np.testing.assert_allclose(field.sample_offset('train_A'),[[[.05,.2,0],[0,0,0],[0,0,0]]],atol=1e-7)
    field.bind(world,np.zeros((1,3),bool))
    np.testing.assert_array_equal(field.sample_offset('train_A'),np.zeros((1,3,3)))


def test_base_coefficient_checksum_fails_closed(field_files):
    from canonical_surface_base import CanonicalSurfaceBase
    manifest,mesh,calibration,payload=field_files
    (manifest.parent/payload['coefficients']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):CanonicalSurfaceBase(manifest,mesh,calibration)


@pytest.mark.parametrize('change',[
    {'uses_eval_rgb':True},{'uses_semantic_masks':True},{'query_dependent_base':True},
    {'mesh_sha256':'wrong'},{'camera_color_calibration_sha256':'wrong'},
    {'camera_color_model':'rgb'},{'source_cameras':['F004_B005_1210O9']},
    {'source_cameras':['train_A','train_A']},{'fit':{'converged':False}},
])
def test_base_provenance_and_solver_guards(field_files,change):
    import json
    from canonical_surface_base import CanonicalSurfaceBase
    manifest,mesh,calibration,payload=field_files
    manifest.write_text(json.dumps({**payload,**change}))
    with pytest.raises(ValueError):CanonicalSurfaceBase(manifest,mesh,calibration)
