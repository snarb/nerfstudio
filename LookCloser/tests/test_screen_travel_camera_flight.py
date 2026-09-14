import numpy as np
from test_wide_dynamic_camera_flight import rig
from diagnose_camera_grid_flight import grid_path
from screen_travel_camera_flight import screen_path,optical_target,portrait_projection


def test_camera_moves_object_across_screen_without_image_transform():
    rows=rig()
    for r in rows:r.update(fl_x=10627.,fl_y=10652.,cx=960.,cy=540.,w=1920,h=1080)
    old,report=grid_path(rows,np.zeros(3),4,720)
    target=optical_target([r['transform_matrix'] for r in old])
    assert np.ptp([portrait_projection(target,r) for r in old],axis=0).max()<1e-8
    path,report=screen_path(rows,dict(path=old,report=report))
    projected=np.array([portrait_projection(target,r) for r in path])
    assert np.ptp(projected[:,0])>380 and np.ptp(projected[:,1])>150
    assert report['columns']=='D..K' and not report['image_crop']
    assert len({r['fl_x'] for r in path})==1 and len({r['cx'] for r in path})==1
    assert len({r['fl_y'] for r in path})==1 and len({r['cy'] for r in path})==1
    poses=np.array([r['transform_matrix'] for r in path]);assert np.allclose(np.linalg.det(poses[:,:3,:3]),1)
    assert np.ptp([r['rig_offset_xy'][0] for r in path])>6.7
