from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from measured_texture_visibility import reject_farther_source


def test_stable_farther_layer_rejected_without_mutating_eligibility():
    depth=np.ones((9,9));uv=np.array([[4,4],[4,4]],float);z=np.array([.8,.8]);valid=np.array([True,False])
    np.testing.assert_array_equal(reject_farther_source(depth,uv,z,valid),[True,False])
    np.testing.assert_array_equal(valid,[True,False])


@pytest.mark.parametrize('value',[0,np.nan,.8,.801,.5])
def test_unknown_near_or_nearer_layer_not_rejected_by_this_far_only_guard(value):
    assert not reject_farther_source(np.full((9,9),value),np.array([[4,4]]),np.array([.8]),np.array([True]))[0]


def test_single_far_tap_does_not_reject():
    depth=np.zeros((9,9));depth[4,4]=1
    assert not reject_farther_source(depth,np.array([[4,4]]),np.array([.8]),np.array([True]))[0]


def test_border_and_bad_shapes():
    assert not reject_farther_source(np.ones((9,9)),np.array([[0,0]]),np.array([.8]),np.array([True]))[0]
    with pytest.raises(ValueError):reject_farther_source(np.ones((9,9)),np.zeros((1,2)),np.ones(1),np.ones(1))


def test_raw_depth_and_display_rgb_lattices_are_distinct():
    from study_confidence_depth_prior import project_integer
    from joint_temporal_texture import project
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=500.,fl_y=500.,cx=4.,cy=4.)
    points=np.array([[.00416,0.,-.8]],np.float32)
    depth_uv,_=project_integer(camera,points)
    rgb_uv,_=project(points,[camera])
    np.testing.assert_allclose(depth_uv,rgb_uv[0]+.5,atol=1e-6)
    assert np.rint(depth_uv[0,0]) != np.rint(rgb_uv[0,0,0])
