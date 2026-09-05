from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_texture_visibility import MeshVisibility


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
