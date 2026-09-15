import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_original_crown_support import classify,verify_original_surface,display_pixels
import pytest


def test_labels_do_not_treat_unknown_depth_as_mask_contradiction():
    votes=np.array([[0,0,0,0],[2,3,2,3],[0,1,1,1],[1,2,2,3]])
    low,multi,both=classify(votes,np.array([0,20,3,0]))
    np.testing.assert_array_equal(low,[True,False,True,False])
    np.testing.assert_array_equal(multi,[False,True,True,False])
    np.testing.assert_array_equal(both,[False,False,True,False])


def test_zero_distance_topology_split_preserves_surface_not_vertex_ids():
    old=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    new=np.concatenate([old,old[:1]])
    verify_original_surface(new,np.array([[3,1,2]]),old,np.array([[0,1,2]]))
    new[3,0]=.01
    with pytest.raises(AssertionError):
        verify_original_surface(new,np.array([[3,1,2]]),old,np.array([[0,1,2]]))


def test_display_witness_is_already_uint8():
    rgb=np.array([[[17,112,255]]],np.uint8)
    np.testing.assert_array_equal(display_pixels(rgb),rgb)
    with pytest.raises(ValueError):display_pixels(rgb.astype(float)/255)
