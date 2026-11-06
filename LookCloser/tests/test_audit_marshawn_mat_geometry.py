"""Exact bounded-ray checks protect the proposed occlusion interval convention."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_mat_geometry import ray_bounded_slab


def test_bounded_surface_interval_stays_before_behind_surface_shoe():
    hull=ConvexHull(np.array([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]]))
    interval=ray_bounded_slab(np.array([0.,0.,2.]),np.array([0.,0.,-1.]),
        np.zeros(3),np.array([0.,0.,1.]),np.array([[1.,0.,0.],[0.,1.,0.]]),hull,.1)
    assert np.allclose(interval,[1.9,2.1])
    assert interval[1] < 3.  # shoe at z=-1 must remain behind the occluder


def test_no_plane_extrapolation_outside_observed_footprint():
    hull=ConvexHull(np.array([[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]]))
    interval=ray_bounded_slab(np.array([2.,0.,2.]),np.array([0.,0.,-1.]),
        np.zeros(3),np.array([0.,0.,1.]),np.array([[1.,0.,0.],[0.,1.,0.]]),hull,.1)
    assert interval is None
