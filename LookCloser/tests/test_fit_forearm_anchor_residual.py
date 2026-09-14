import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_forearm_anchor_residual import anchor_targets,solve_residual


def test_repeated_camera_samples_do_not_dominate_median():
    uv=np.array([[0.,0.]]);depth=np.array([1.])
    samples=np.zeros((102,2));z=np.r_[np.full(100,1.005),.999,1.001];sources=np.r_[np.zeros(100),1,2]
    target,weight,count,_=anchor_targets(uv,depth,samples,z,sources)
    np.testing.assert_allclose(target,[.001]);assert weight[0]==1 and count[0]==3


def test_surface_residual_preserves_pins_and_spreads_local_constraint():
    delta,r=solve_residual(5,[[0,1],[1,2],[2,3],[3,4]],np.array([0,0,.003,0,0]),np.array([0,0,1,0,0]),np.array([1,0,0,0,1],bool))
    assert delta[0]==delta[4]==0 and 0<delta[1]<delta[2]<.003
    np.testing.assert_allclose(delta[1],delta[3])


def test_displacement_bound_and_empty_constraints():
    delta,r=solve_residual(2,[[0,1]],np.array([0,.1]),np.array([0,1]),np.array([1,0],bool))
    assert delta[0]==0 and delta[1]==.006 and r['clipped_nodes']==1
    with pytest.raises(ValueError,match='No movable'):
        solve_residual(2,[[0,1]],np.zeros(2),np.zeros(2),np.array([1,0],bool))


def test_distant_anchor_does_not_influence_patch():
    _,weight,_,_=anchor_targets(np.array([[0.,0.]]),np.ones(1),np.array([[2.,0.]]),np.ones(1),np.zeros(1))
    assert weight[0]==0
