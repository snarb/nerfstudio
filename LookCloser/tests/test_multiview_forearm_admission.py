import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_multiview_forearm_admission import missing_in_train_views
from diffusion_mesh_repair import scene_for


def test_point_rays_use_camera_z_separation_and_actual_frustum():
    v = np.array([[-10., -10., -3.], [10., -10., -3.], [10., 10., -3.], [-10., 10., -3.]])
    scene = scene_for(v, np.array([[0, 1, 2], [0, 2, 3]]))
    rows = []
    for shift in [0., .1]:
        pose = np.eye(4);pose[0, 3] = shift
        rows.append(dict(transform_matrix=pose, fl_x=100., fl_y=100., cx=50.5, cy=40.5, w=100, h=80))
    points = np.array([[0., 0., -2.], [0., 0., -2.999], [0., 0., -2.995], [0., 0., -4.], [100., 0., -2.]])
    votes, available = missing_in_train_views(points, rows, scene)
    assert votes.tolist() == [2, 0, 2, 0, 0]
    assert available.tolist() == [2, 2, 2, 2, 0]
