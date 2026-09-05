from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_fixed_camera_feature_geometry import mutual_matches,holdout,epipolar_distance,temporal_similarity


def test_mutual_descriptors_are_one_to_one():
    a=np.eye(8,dtype=np.float32);b=a[::-1].copy()
    pairs=mutual_matches(a,b)
    np.testing.assert_array_equal(pairs[:,1],7-pairs[:,0])


def test_block_holdout_does_not_split_local_neighbors():
    points=np.array([[130,260],[140,270],[512,768]])
    assert holdout(points)[0]==holdout(points)[1]


def test_signed_distance_in_image_pixels():
    f=np.array([[0,0,0],[0,0,-1],[0,1,0.]])
    np.testing.assert_allclose(epipolar_distance(f,np.array([[50,30],[2,3]]),np.array([[80,32],[20,3]])),[-2,0])


def test_temporal_cluster_recovers_small_shift_not_large_moving_component():
    a=np.random.default_rng(2).random((100,2))*1000;b=a+[.3,-.2];b[50:]+=20
    result,accepted=temporal_similarity(a,b)
    assert accepted.sum()==50 and not accepted[50:].any()
    np.testing.assert_allclose(result['median_displacement_xy'],[.3,-.2],atol=1e-7)
