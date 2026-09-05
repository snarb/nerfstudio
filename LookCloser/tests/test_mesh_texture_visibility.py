from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_texture_visibility import MeshVisibility


def test_bilinear_footprint_rejects_cross_layer_and_missing_taps():
    import torch
    from mesh_texture_visibility import bilinear_depth_footprint_support
    depth=torch.ones((3,3));u=v=torch.tensor([[.25]])
    assert bilinear_depth_footprint_support(depth,u,v,torch.ones_like(u)).item()
    for bad in (.7,1.4,0.,float('nan')):
        depth[1,1]=bad
        assert not bilinear_depth_footprint_support(depth,u,v,torch.ones_like(u)).item()


def test_bilinear_footprint_ignores_zero_weight_and_supports_raster_edge():
    import torch
    from mesh_texture_visibility import bilinear_depth_footprint_support
    depth=torch.ones((3,3));depth[1,1]=0
    u=v=torch.tensor([[0.,2.]])
    assert bilinear_depth_footprint_support(depth,u,v,torch.ones_like(u)).all()
    u=torch.tensor([[float('nan'),3.]])
    assert not bilinear_depth_footprint_support(depth,u,v,torch.ones_like(u)).any()


def test_bilinear_footprint_preserves_small_smooth_depth_slopes():
    import torch
    from mesh_texture_visibility import bilinear_depth_footprint_support
    depth=torch.tensor([[1.,1.001],[1.002,1.003]])
    u=v=torch.tensor([[.5]])
    assert bilinear_depth_footprint_support(depth,u,v,torch.tensor([[1.0015]])).item()


def test_stereo_support_does_not_average_zero_or_far_depth():
    import torch
    from mesh_texture_visibility import observed_depth_support
    depth=torch.ones((7,7));depth[2:5,2:5]=0
    u=v=torch.tensor([[3.]])
    supported,fraction,count=observed_depth_support(depth,u,v,torch.ones_like(u))
    assert supported.item() and count.item()==16 and fraction.item()==1
    depth[:]=2;depth[3,3]=1
    supported,fraction,count=observed_depth_support(depth,u,v,torch.ones_like(u))
    assert not supported.item() and count.item()==25
    depth[:]=0;depth[3,3]=1
    assert not observed_depth_support(depth,u,v,torch.ones_like(u))[0].item()


def test_depth_aware_rgb_matches_bilinear_on_one_surface():
    import torch
    from mesh_texture_visibility import sample_rgb_depth_aware
    from render_mesh_image_blend import grid_sample
    rgb=torch.rand((3,7,9),generator=torch.Generator().manual_seed(3))
    u=torch.tensor([[0.,2.35,8.]]);v=torch.tensor([[0.,3.76,6.]])
    sampled,valid,mass=sample_rgb_depth_aware(rgb,torch.ones((7,9)),u,v,torch.ones_like(u))
    assert valid.all()
    torch.testing.assert_close(mass,torch.ones_like(mass))
    torch.testing.assert_close(sampled,grid_sample(rgb,u,v),atol=2e-6,rtol=0)


def test_depth_aware_rgb_avoids_other_surface_without_dropping_sample():
    import torch
    from mesh_texture_visibility import sample_rgb_depth_aware
    rgb=torch.ones((3,2,2));rgb[:,1,1]=0
    depth=torch.ones((2,2));depth[1,1]=2
    u=v=torch.tensor([[.5]])
    sampled,valid,mass=sample_rgb_depth_aware(rgb,depth,u,v,torch.ones_like(u))
    assert valid.item() and mass.item()==.75
    torch.testing.assert_close(sampled,torch.ones_like(sampled))
    depth[:]=2
    sampled,valid,mass=sample_rgb_depth_aware(rgb,depth,u,v,torch.ones_like(u))
    assert not valid.item() and mass.item()==0 and not sampled.any()


def test_exact_visibility_rejects_occluder_not_same_surface(tmp_path):
    o3d=pytest.importorskip('open3d')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]]),
                                  o3d.utility.Vector3iVector([[0,1,2],[1,3,2]]))
    path=tmp_path/'plane.ply';o3d.io.write_triangle_mesh(str(path),mesh)
    scene=MeshVisibility(path)
    points=np.array([[[0,0,2],[.8,0,1],[3,0,2]]],np.float32)
    valid,stats=scene.visible(points,[0,0,0],np.ones((1,3),bool))
    np.testing.assert_array_equal(valid,[[False,True,True]])
    assert stats['occluded']==1 and stats['hit_near_target']==1 and stats['no_hit']==1
    empty,stats=scene.visible(points,[0,0,0],np.zeros((1,3),bool))
    assert not empty.any() and stats['rays']==0
