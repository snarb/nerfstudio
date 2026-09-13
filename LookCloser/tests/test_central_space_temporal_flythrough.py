import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from central_space_temporal_flythrough import motion_report
from audit_smooth_temporal_video import camera_motion_samples,validate_central_containment


def test_open_arc_has_no_fake_last_to_first_velocity():
    poses=np.tile(np.eye(4),(5,1,1));poses[:,0,3]=np.arange(5)*.01
    angular,speed=camera_motion_samples(poses,30,periodic=False)
    assert len(speed)==4
    np.testing.assert_allclose(speed,.3);np.testing.assert_allclose(angular,0)
    _,closed=camera_motion_samples(poses,30,periodic=True)
    assert len(closed)==5 and closed[-1]==pytest.approx(1.2)


def test_open_motion_report_constant_translation():
    poses=np.tile(np.eye(4),(8,1,1));poses[:,1,3]=np.arange(8)*.02
    report=motion_report(poses)
    assert report['speed_max_min_ratio']==pytest.approx(1)
    assert report['acceleration_max']<1e-12


def containment_case():
    a=np.eye(4);b=np.eye(4);b[0,3]=2.
    cal={'frames':[{'physical_camera':'a','transform_matrix':a.tolist()},
                   {'physical_camera':'b','transform_matrix':b.tolist()}]}
    poses=np.tile(np.eye(4),(2,1,1));poses[:,0,3]=[.5,1.5]
    request={'camera_path_report':{'anchors':['a','b']},'inventory':[
        {'camera':{'convex_weights':[.75,.25]}},{'camera':{'convex_weights':[.25,.75]}}]}
    return request,poses,cal


def test_containment_checks_actual_pose_not_just_positive_weights():
    r,p,c=containment_case();assert validate_central_containment(r,p,c)==.25
    p[0,0,3]+=1
    with pytest.raises(ValueError,match='actual'):validate_central_containment(r,p,c)


def test_negative_hull_weights_rejected():
    r,p,c=containment_case();r['inventory'][0]['camera']['convex_weights']=[-.1,1.1]
    with pytest.raises(ValueError,match='outside'):validate_central_containment(r,p,c)
