"""Temporal mesh frames must share one calibration-space camera trajectory."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from render_smooth_temporal_mesh_video import calibration_pose,RECIPE
from render_patchmatch_camera_path import normalize_frame
from joint_temporal_texture import project


@pytest.mark.parametrize('scale',[.1,.7,2.])
def test_calibration_pose_roundtrip_preserves_rotation_and_translation(scale):
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();pose[:3,3]=[1,2,3]
    applied=np.eye(4);applied[:3,:3]=Rotation.from_euler('z',.3).as_matrix();applied[:3,3]=[.2,-.1,.4]
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('x',-.8).as_matrix();transform[:3,3]=[-1,.4,3]
    cal={'applied_transform':applied[:3].tolist()};meta={'dataparser_scale':scale,'dataparser_transform':transform[:3].tolist()}
    row={'transform_matrix':pose.tolist(),'physical_camera':'synthetic','fl_x':1000.,'fl_y':1000.,'cx':960.,'cy':540.}
    normalized=normalize_frame(row,cal,meta);recovered=calibration_pose(normalized,cal,meta)
    np.testing.assert_allclose(recovered['transform_matrix'],pose,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(np.array(recovered['transform_matrix'])[:3,:3]),1,atol=1e-12)


def test_different_mesh_gauges_preserve_pixel_projection():
    row={'transform_matrix':np.eye(4).tolist(),'fl_x':1000.,'fl_y':1100.,'cx':960.,'cy':540.}
    points=np.array([[.1,.2,-2.],[-.2,.1,-3.]])
    reference=project(points,[row])[0]
    for angle,scale in [(0.,.1),(.7,.25)]:
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('y',angle).as_matrix();t[:3,3]=[.3,-.4,.7]
        metadata={'dataparser_scale':scale,'dataparser_transform':t[:3].tolist()}
        camera=normalize_frame(row,{},metadata);normalized=(points@t[:3,:3].T+t[:3,3])*scale
        np.testing.assert_allclose(project(normalized,[camera])[0],reference,atol=2e-4)


def test_temporal_recipe_keeps_all_times_and_fixed_camera_response():
    assert RECIPE['frames']==150 and RECIPE['fps']==30
    assert RECIPE['fixed_profiles'] and not RECIPE['per_time_registration']
    assert RECIPE['source_count']==62 and not RECIPE['averages_rgb']
