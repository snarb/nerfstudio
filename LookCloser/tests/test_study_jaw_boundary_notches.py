import sys
from pathlib import Path
import numpy as np
import mapbox_earcut
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_jaw_boundary_notches import propose,SETTINGS


def notched_disk():
    xy=np.array([[-.004,-.004],[.004,-.004],[.004,.004],[.0004,.004],
                 [.0004,.002],[-.0004,.002],[-.0004,.004],[-.004,.004]])
    t=mapbox_earcut.triangulate_float64(xy,np.array([len(xy)],np.uint32)).reshape(-1,3)
    return np.column_stack((xy,np.zeros(len(xy)))),t


def test_small_notch_preserves_all_old_triangles_and_vertices():
    v,t=notched_disk();saved=v.copy();settings={**SETTINGS,'max_edges':6}
    tt,notes=propose(v,t,settings)
    assert len(tt)==len(t)+2 and len(notes)==1
    assert np.array_equal(tt[:len(t)],t) and np.array_equal(v,saved)
    assert set(tt[len(t):].ravel())=={3,4,5,6}


def test_outside_head_or_oversize_is_not_completed():
    v,t=notched_disk();settings={**SETTINGS,'max_edges':6}
    moved=v.copy();moved[:,0]-=1
    tt,notes=propose(moved,t,settings)
    assert np.array_equal(tt,t) and not notes
    tt,notes=propose(v*10,t,{**settings,'min_head_x':-1})
    assert np.array_equal(tt,t) and not notes


def test_guard_integer_grid_shift_matches_explicit_rays():
    import open3d as o3d
    k=np.array([[10.,0,2.],[0,12.,1.5],[0,0,1.]])
    shifted=k.copy();shifted[:2,2]+=.5
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(shifted),o3d.core.Tensor(np.eye(4)),4,3).numpy()
    yy,xx=np.indices((3,4))
    expected=np.stack(((xx-2.)/10.,(yy-1.5)/12.,np.ones_like(xx)),-1)
    assert np.allclose(rays[...,3:],expected,atol=1e-6)
