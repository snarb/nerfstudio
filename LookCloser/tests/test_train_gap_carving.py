import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_train_gap_carving import negative_samples,removable


def test_gap_sampling_requires_whole_footprint_and_depth_layer():
    mask=np.ones((5,5),bool);mask[2,2]=False
    xy=np.array([[0.2,0.2],[1.2,1.2],[0.2,0.2],[0.2,0.2],[np.nan,1],[4,4],[-.1,0]])
    z=np.array([1.,1.,.4,2.,1.,1.,1.])
    np.testing.assert_array_equal(negative_samples(mask,xy,z,[.9,1.1]),[True,False,False,False,False,False,False])


def test_same_three_views_must_reject_every_triangle_sample():
    neg=np.ones((4,4),bool);samples=np.array([[0,1,2,3]])
    assert removable(neg,samples)[0]
    neg[0,0]=False
    assert removable(neg,samples)[0]
    # Each point still has >=3 votes, but not the same three cameras.
    neg[1,1]=False
    assert (neg.sum(0)>=3).all()
    assert not removable(neg,samples)[0]


def test_invalid_shapes_and_depth_slab_fail():
    with pytest.raises(ValueError): negative_samples(np.ones((3,3),bool),np.zeros((1,2)),np.ones(1),[2,1])
    with pytest.raises(ValueError): removable(np.ones((4,5),bool),np.zeros((2,3),int))
