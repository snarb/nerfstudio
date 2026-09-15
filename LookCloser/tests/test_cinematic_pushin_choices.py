import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from cinematic_pushin_framed import timing,VARIANTS,path_for,PARENT
from render_smooth_temporal_mesh_video import verify_request

def test_velocity_envelope_and_hold():
    x=np.arange(150)/126;s=timing(x)
    assert np.all(np.diff(s)>=-1e-12)
    assert s[20]>.14 and np.all(s[126:]==1)
    eps=1e-5
    assert abs(float((timing(1)-timing(1-eps))/eps))<1e-6

def test_exact_real_hold_and_distinct_actor_times():
    parent=verify_request(PARENT)
    assert len(set(parent['ordered_frame_ids']))==150
    for variant in VARIANTS:
        rows,report=path_for(variant,parent);poses=np.array([r['transform_matrix'] for r in rows]);endpoint=np.array(report['actual_endpoint_calibration_pose'])
        np.testing.assert_allclose(poses[126:],np.repeat(endpoint[None],24,axis=0),atol=1e-10)
        assert report['radial_largest_outward_step']<1e-9
        assert report['radial_approach_fraction']>.05 and report['train_hull_max_residual']<1e-6
        assert np.allclose([r['fl_x'] for r in rows[126:]],rows[126]['fl_x'])
        if variant=='locked_arc':assert report['look_at_angular_error_max_degrees']<.001
        if variant=='soft_diagonal':assert report['center_ray_angle_extent_degrees']>18

def test_live_ending_early_rest_and_optical_disclosure():
    from cinematic_pushin_live_ending import path_for as live_path,RAW_IDS
    parent=verify_request(PARENT)
    assert len(RAW_IDS)==126 and RAW_IDS[-1]=='001149'
    for variant in VARIANTS:
        rows,report=live_path(variant,parent);poses=np.array([r['transform_matrix'] for r in rows]);endpoint=np.array(report['actual_endpoint_calibration_pose'])
        np.testing.assert_allclose(poses[118:],np.repeat(endpoint[None],32,axis=0),atol=1e-10)
        for key in ['fl_x','fl_y','cx','cy']:assert np.ptp([r[key] for r in rows[118:]])==0
        assert np.linalg.norm(poses[24,:3,3]-poses[0,:3,3])>.02
        assert not report['presentation']['final_second_is_3d']
        assert report['presentation']['dissolve_indices']==[118,125]
