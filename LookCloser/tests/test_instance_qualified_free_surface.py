"""Fail-closed mask lookup, including OpenCV's remap dimension limit."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_instance_qualified_free_surface import outside_mask


def test_unknown_and_boundary_are_not_negative_evidence():
    sdf=np.full((10,10),-3,np.float32);sdf[3,3]=-1
    outside,valid=outside_mask(sdf,[[4,4],[3,3],[-1,4],[10,4],[np.nan,4]])
    np.testing.assert_array_equal(outside,[True,False,False,False,False])
    np.testing.assert_array_equal(valid,[True,True,False,False,False])


def test_more_than_32767_queries_remains_identical():
    sdf=np.full((10,10),-3,np.float32)
    q=np.tile([[4.25,5.25],[15,5]],(40000,1))
    outside,valid=outside_mask(sdf,q)
    np.testing.assert_array_equal(outside,np.tile([True,False],40000))
    np.testing.assert_array_equal(valid,outside)
