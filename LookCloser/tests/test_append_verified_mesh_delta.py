import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from append_verified_mesh_delta import append_delta
from curve_forearm_delta import curve_vertices,boundary_curve_vertices,semantic_faces


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


def test_boundary_condition_keeps_old_depth_ring_exact_and_feathers_interior():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    v=np.array([[1,2,3],[0,0,-1],[.01,0,-1],[.1,0,-1],[1,0,-1]],float)
    accepted=np.zeros((100,100),bool);accepted[20:80,51:80]=True
    fit=dict(reference_center=[50,50],all_camera_coefficients=[0,0,1.001,0,0,0])
    out,r=boundary_curve_vertices(v,1,camera,fit,[1,2,3],accepted)
    np.testing.assert_array_equal(out[[0,1,4]],v[[0,1,4]])
    assert 0<out[2,2]-v[2,2]<out[3,2]-v[3,2]<.002
    np.testing.assert_allclose(out[[2,3],0]/-out[[2,3],2],[.01,.1])
    assert r['boundary_ring_exact'] and r['boundary_ring_vertices']==1


def test_opt_in_axis_extent_matches_original_grid_builder():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100,fl_y=100,cx=50,cy=50)
    rows=[dict(camera,physical_camera=n) for n in ['one','two']]
    masks={n:np.ones((1080,1920),bool) for n in ['one','two']}
    v=np.array([[0,0,-1],[.0015,.0015,-1],[.0015,0,-1]])
    f=np.array([[0,1,2]])
    legacy,_=semantic_faces(v,f,rows,masks)
    axis,r=semantic_faces(v,f,rows,masks,axis_extent=True)
    assert len(legacy)==0 and len(axis)==1 and r['extent_metric']=='axis_extent_strict'


def test_matched_plane_cannot_silently_select_a_different_admission_protocol(tmp_path):
    from study_forearm_production_delta import prepare
    out=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='Matched plane requires'):
        prepare(out,'001029',matched_plane=True)
    assert not out.exists()
