import numpy as np
import pytest
from test_wide_dynamic_camera_flight import rig
from interior_dynamic_camera_flight import interior_path


def test_interior_margin_and_actual_motion():
    path,report=interior_path(rig(),np.zeros(3))
    xy=np.array([p['rig_offset_xy'] for p in path]);absolute=xy+[7,2]
    assert np.all(absolute[:,0]>=2) and np.all(absolute[:,0]<=11)
    assert np.all(absolute[:,1]==2)
    assert np.ptp(xy[:,0])==4 and np.ptp(xy[:,1])==0
    assert xy[0,0]==-2 and xy[75,0]==2
    poses=np.array([p['transform_matrix'] for p in path])
    assert np.ptp(poses[:,1,3])>.2 and report['maximum_pairwise_view_angle_degrees']>10
    assert np.allclose(np.linalg.det(poses[:,:3,:3]),1)
    # Position, velocity and acceleration are continuous through loop closure.
    delta=np.roll(poses[:,:3,3],-1,axis=0)-poses[:,:3,3]
    assert np.linalg.norm(delta[0])<np.linalg.norm(delta[37])*.05
    assert np.allclose(delta[-1],-delta[0])
    assert report['allowed_absolute_row_range']==[2,2]


def test_missing_inner_anchor_rejected():
    with pytest.raises(KeyError):
        interior_path([r for r in rig() if not r['physical_camera'].startswith('J004_C005')],np.zeros(3))
