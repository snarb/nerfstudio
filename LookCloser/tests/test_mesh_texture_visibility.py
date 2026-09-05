from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mesh_texture_visibility import MeshVisibility


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
