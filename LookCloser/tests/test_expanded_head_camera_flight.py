import numpy as np
from test_wide_dynamic_camera_flight import rig
from diagnose_camera_grid_flight import grid_path
from expanded_head_camera_flight import expanded_path,polygon_weights,RIG_POLYGON
from repair_temporal_head_boundaries import close_loops
from local_mesh_repair import boundary_loops

def expanded_rig():
    from copy import deepcopy
    rows=rig()
    for source in list(rows):
        if source['physical_camera'].startswith('D004_') and not source['physical_camera'].startswith('D004_A'):
            row=deepcopy(source);row['physical_camera']='C'+row['physical_camera'][1:];row['transform_matrix'][1][3]+=.06;rows.append(row)
    for row in rows:row.update(fl_x=10627.,fl_y=10652.,cx=960.,cy=540.,w=1920,h=1080)
    return rows

def test_expanded_motion_uses_existing_five_anchor_hull():
    rows=expanded_rig()
    pilot,report=grid_path(rows,np.zeros(3),4,720)
    path,report=expanded_path(rows,dict(path=pilot,report=report))
    xy=np.array([p['rig_offset_xy'] for p in path]);weights=np.array([p['convex_weights'] for p in path])
    assert (np.ptp(xy,axis=0)>[8.6,3.8]).all()
    assert (weights>=0).all() and np.allclose(weights.sum(1),1)
    assert np.allclose(weights@RIG_POLYGON,xy)
    assert report['vertical_top_expansion_unavailable']
    assert len({p['fl_x'] for p in path})==1

def test_hole_completion_preserves_original_triangles():
    v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    t=np.array([[0,2,1],[0,3,2],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    loops,rejected=boundary_loops(t);assert not rejected and len(loops)==1
    vv,tt,operation=close_loops(v,t,loops)
    assert np.array_equal(vv[:len(v)],v) and np.array_equal(tt[:len(t)],t)
    assert not boundary_loops(tt)[0]
    assert operation['closed_boundary_edges']==4

def test_polygon_rejects_extrapolation():
    import pytest
    with pytest.raises(ValueError):polygon_weights(np.array([-6.,3.]))

def test_isolated_triangle_is_not_mistaken_for_a_hole():
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],float);t=np.array([[0,1,2]])
    loops,_=boundary_loops(t);vv,tt,operation=close_loops(v,t,loops)
    assert np.array_equal(vv,v) and np.array_equal(tt,t)
    assert operation['added_triangles']==0

def test_camera_workaround_still_moves_smoothly_inside_rig():
    from artifact_aware_camera_flight import avoidance_path,LIMITS
    rows=expanded_rig();pilot,report=grid_path(rows,np.zeros(3),4,720)
    path,_=avoidance_path(rows,dict(path=pilot,report=report))
    xy=np.array([p['rig_offset_xy'] for p in path]);weights=np.array([p['convex_weights'] for p in path])
    assert (np.ptp(xy,axis=0)>[4.5,2.8]).all()
    assert (xy.min(0)>=[LIMITS['left']-1e-6,LIMITS['bottom']-1e-6]).all()
    assert (xy.max(0)<=[LIMITS['right']+1e-6,LIMITS['top']+1e-6]).all()
    assert (weights>=0).all() and np.allclose(weights@RIG_POLYGON,xy)
    poses=np.array([p['transform_matrix'] for p in path]);step=np.linalg.norm(np.roll(poses[:,:3,3],-1,axis=0)-poses[:,:3,3],axis=1)
    ratio=step/np.roll(step,-1)
    assert max(ratio.max(),1/ratio.min())<1.07
    assert len({p['fl_x'] for p in path})==1

def test_elevated_workaround_retimes_spline_for_smooth_speed():
    from elevated_camera_workaround import elevated_path
    rows=expanded_rig();pilot,report=grid_path(rows,np.zeros(3),4,720)
    path,report=elevated_path(rows,dict(path=pilot,report=report))
    xy=np.array([p['rig_offset_xy'] for p in path]);poses=np.array([p['transform_matrix'] for p in path])
    assert (np.ptp(xy,axis=0)>[4.5,.7]).all() and xy[:,1].min()>=.19999
    step=np.linalg.norm(np.roll(poses[:,:3,3],-1,axis=0)-poses[:,:3,3],axis=1)
    assert step.max()/step.min()<1.03
    assert report['extrema_indices']['bottom']==int(xy[:,1].argmin())
