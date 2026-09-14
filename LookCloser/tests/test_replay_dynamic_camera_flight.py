import numpy as np
from test_wide_dynamic_camera_flight import rig
from diagnose_camera_grid_flight import grid_path
from replay_dynamic_camera_flight import replay_path


def test_replay_full_loop_preserves_both_pose_components_and_phase():
    rows=rig();saved,report=grid_path(rows,np.zeros(3),4,720)
    replay=replay_path(dict(path=saved,report=report))
    assert len(replay)==150 and replay[-1]['pilot_sample_phase']==715.2
    # Every fifth replay sample is exactly every 24th saved pose, not its prefix.
    for i in range(0,150,5):
        assert np.allclose(replay[i]['transform_matrix'],saved[i*24//5]['transform_matrix'],atol=1e-12)
    xy=np.array([p['rig_offset_xy'] for p in replay])
    assert np.all(np.ptp(xy,axis=0)>2.87)
    assert np.linalg.norm(np.array(replay[0]['transform_matrix'])[:3,3]-np.array(replay[75]['transform_matrix'])[:3,3])>.1
