import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_mhr_local_patch_candidates import boundary_opposites, near_actual_boundary, subdivide


def test_shared_diagonal_is_not_open_even_with_boundary_vertices():
    t=np.array([[0,1,2],[0,2,3]])
    opposite=boundary_opposites(t)
    np.testing.assert_array_equal(opposite,[[True,False,True],[True,True,False]])
    assert not near_actual_boundary(np.array([[.5,0,.5]]),opposite[:1])[0]
    assert near_actual_boundary(np.array([[0,.5,.5]]),opposite[:1])[0]
    assert near_actual_boundary(np.array([[1,0,0]]),opposite[:1])[0]


def test_subdivision_preserves_shared_edges_winding_and_parents():
    v=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    t=np.array([[0,1,2],[0,2,3]])
    vv,tt,parents,rounds=subdivide(v,t,np.array([11,19]),max_edge=.8)
    assert rounds==1 and len(vv)==9 and len(tt)==8
    np.testing.assert_array_equal(parents,[11]*4+[19]*4)
    assert (np.cross(vv[tt[:,1]]-vv[tt[:,0]],vv[tt[:,2]]-vv[tt[:,0]])[:,2]>0).all()
    assert boundary_opposites(tt).sum()==8
