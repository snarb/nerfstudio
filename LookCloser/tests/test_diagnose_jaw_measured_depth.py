import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_jaw_measured_depth import barycentric_samples, observed_at
from study_confidence_depth_prior import support
from guard_jaw_measured_depth import initial_admission
from review_jaw_depth_controls import normalization_matrix


def test_samples_are_bounded_and_include_centroid():
    v = np.array([[[0,0,0],[3,0,0],[0,3,0]]],float)
    samples = barycentric_samples(v)
    assert samples.shape == (1,10,3)
    np.testing.assert_array_equal(samples[0,:3],v[0])
    np.testing.assert_allclose(samples[0,-1],[1,1,0])
    assert (samples >= 0).all() and (samples.sum(2) <= 3).all()


def test_missing_and_out_of_frame_are_unknown_not_free():
    camera = dict(transform_matrix=np.eye(4).tolist(), fl_x=1,fl_y=1,cx=1,cy=1)
    depth = np.zeros((3,3));depth[1,1]=2
    points = np.array([[0,0,-2],[2,0,-2],[20,0,-2],[0,0,2]],float)
    xy,z,observed,available = observed_at(points,camera,depth)
    np.testing.assert_array_equal(available,[True,False,False,False])
    assert observed[0] == 2 and z[0] == 2


def test_observed_support_excludes_reference_and_missing_maps():
    rows = []
    for i, x in enumerate([0,.1,-.1,.2]):
        pose = np.eye(4); pose[0,3] = x
        rows.append(dict(physical_camera=str(i), transform_matrix=pose.tolist(),
                         fl_x=100,fl_y=100,cx=50,cy=50))
    points = np.array([[0,0,-2]],float)
    depths = [np.full((1080,1920),2,dtype=np.float32) for _ in rows]
    count,free = support(points, rows[0], rows, depths)
    np.testing.assert_array_equal(count,[3]); np.testing.assert_array_equal(free,[0])
    for depth in depths[1:]: depth.fill(0)
    count,free = support(points, rows[0], rows, depths)
    np.testing.assert_array_equal(count,[0]); np.testing.assert_array_equal(free,[0])


def test_admission_requires_anchors_and_no_measured_free_space():
    votes = np.full((5,10),2,dtype=np.uint8)
    votes[1,:3]=[0,0,5]
    votes[2,3:]=0
    free = np.zeros((62,5,10),np.uint8);free[10,3,8]=1
    outside=np.array([0,0,0,0,1]); masks=np.full(5,62)
    np.testing.assert_array_equal(initial_admission(votes,free,masks,outside),[True,False,False,False,False])


def test_controls_use_exact_shared_normalization_not_relaxed_comparison():
    a=dict(dataparser_transform=[[1,0,0,3],[0,1,0,2],[0,0,1,1]],dataparser_scale=.1)
    b=dict(dataparser_transform=[[0,-1,0,1],[1,0,0,4],[0,0,1,2]],dataparser_scale=.2)
    point=np.array([2,3,4,1.])
    mapping=normalization_matrix(a)@np.linalg.inv(normalization_matrix(b))
    np.testing.assert_allclose(mapping@normalization_matrix(b)@point,normalization_matrix(a)@point,atol=1e-14)
