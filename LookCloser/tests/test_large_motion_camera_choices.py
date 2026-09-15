"""Read-only regression gates for the explicitly larger opt-in camera shots."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from joint_temporal_texture import read
from large_motion_camera_choices import path_for as original_path
from visible_large_motion_choices import path_for,PARENT,VARIANTS


@pytest.mark.parametrize('variant',VARIANTS)
def test_physical_arc_and_fixed_lens(variant):
    parent=read(PARENT/'request.json');before=repr(parent)
    old,_=original_path(variant,parent);rows,report=path_for(variant,parent)
    assert repr(parent)==before
    assert len(rows)==150 and report['fps']==24
    a=np.array([r['transform_matrix'] for r in old]);b=np.array([r['transform_matrix'] for r in rows])
    np.testing.assert_allclose(a[:,:3,3],b[:,:3,3],atol=1e-9)
    assert report['center_ray_angle_extent_degrees']>45
    assert report['projected_landmark_extent_pixels'][0]>500
    assert report['projected_landmark_extent_pixels'][1]>500
    assert -.95<report['rig_parameter_min'][1]<report['rig_parameter_max'][1]<.95
    assert not report['camera_periodic'] and not report['image_crop'] and not report['lens_animation']
    for key in ['fl_x','fl_y']:
        assert len({r[key] for r in rows})==1
        np.testing.assert_allclose(rows[0][key],old[0][key]*.85)
    assert np.linalg.norm(b[-1,:3,3]-b[-3,:3,3])<1e-9


def test_refined_right_arc_keeps_large_physical_motion():
    from right_arc_visibility_refinement import path_for as refined
    parent=read(PARENT/'request.json');rows,report=refined('right_high_arc_refined',parent)
    assert len(rows)==150 and report['center_ray_angle_extent_degrees']>50
    assert report['projected_landmark_extent_pixels'][0]>500
    assert report['projected_landmark_extent_pixels'][1]>600
    assert -.95<report['rig_parameter_min'][1]<report['rig_parameter_max'][1]<.95
    assert not report['camera_periodic'] and not report['lens_animation']
