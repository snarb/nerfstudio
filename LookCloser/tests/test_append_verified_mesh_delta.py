import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from append_verified_mesh_delta import append_delta
from curve_forearm_delta import curve_vertices


def test_transfer_preserves_target_repairs_without_restoring_carved_faces():
    base=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float);bt=np.array([[0,1,2],[1,3,2]])
    target=base[[0,1,2]];tt=np.array([[0,1,2]])
    prior=np.vstack([base,[2,1,0]]);pt=np.vstack([bt,[1,4,3]])
    v,t,r=append_delta(target,tt,base,bt,prior,pt)
    np.testing.assert_array_equal(v[:3],target);np.testing.assert_array_equal(t[:1],tt)
    assert len(t)==2 and r['transferred_triangles']==1 and r['new_vertices']==2
    assert set(map(tuple,v[t[1]]))=={(1,0,0),(2,1,0),(1,1,0)}


def test_rejects_changed_source_prefix():
    v=np.eye(3);t=np.array([[0,1,2]])
    with pytest.raises(ValueError):append_delta(v,t,v,t,v+.01,t)


def test_curvature_preserves_original_vertices_and_reference_rays():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    v=np.array([[1,2,3],[0,0,-1],[.01,0,-1]],float)
    fit=dict(reference_center=[50,50],all_camera_coefficients=[0,0,1,1,0,0])
    out,receipt=curve_vertices(v,1,camera,fit)
    np.testing.assert_array_equal(out[0],v[0])
    np.testing.assert_allclose(out[2,0]/-out[2,2],.01,atol=1e-8)
    assert out[2,2]>-1 and receipt['original_vertices_unchanged']


def test_unused_outlier_cannot_veto_or_move_a_valid_curvature_patch():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    v=np.array([[1,2,3],[.01,0,-1],[1,0,-1]],float)
    fit=dict(reference_center=[50,50],all_camera_coefficients=[0,0,1,1,0,0])
    with pytest.raises(ValueError):curve_vertices(v,1,camera,fit)
    out,_=curve_vertices(v,1,camera,fit,[1])
    np.testing.assert_array_equal(out[[0,2]],v[[0,2]])
