from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from regularized_depth_displacement import solve_displacement


def test_uniform_measured_shift_and_no_evidence_identity():
    h=np.tile(np.eye(3),(4,1,1));g=np.tile([.001,-.0003,.0007],(4,1));edges=[[0,1],[1,2],[2,3]]
    d,s=solve_displacement(h,g,edges,prior_weight=1e-6)
    np.testing.assert_allclose(d,g/(1+1e-6),atol=1e-11,rtol=0)
    assert s['unconstrained_energy_after']<0
    zero,_=solve_displacement(np.zeros_like(h),np.zeros_like(g),edges)
    np.testing.assert_array_equal(zero,np.zeros((4,3)))


def test_norm_bound_graph_order_and_invalid_index():
    h=np.tile(np.eye(3),(2,1,1));g=np.array([[.2,.1,0],[0,.1,.2]])
    a,s=solve_displacement(h,g,[[0,1]],maximum_step=.002)
    b,_=solve_displacement(h,g,[[1,0],[0,1]],maximum_step=.002)
    np.testing.assert_array_equal(a,b);assert (np.linalg.norm(a,axis=1)<=.002+1e-15).all()
    assert s['clipped_vertices']==2
    with pytest.raises(ValueError):solve_displacement(h,g,[[0,2]])
