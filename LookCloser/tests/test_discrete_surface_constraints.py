import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from discrete_surface_constraints import solve


def test_disjoint_feasible_values_never_filled_between():
    domain=np.ones((1,3),bool);options=np.array([[-1.,1.],[-1.,1.],[-1.,1.]])
    out,stats=solve(domain,options,np.ones((3,2),bool),np.array([True,False,True]),np.array([.3,0,.3]))
    assert out[0,1]==1 and out[0,0]==out[0,2]==.3
    assert stats['pins_exact'] and all(a>=b-1e-12 for a,b in zip(stats['energy_history'],stats['energy_history'][1:]))


def test_checkerboard_smoothing_retains_constraints_and_exact_pins():
    domain=np.ones((3,3),bool);options=np.tile([-.002,0,.002],(9,1));valid=np.ones((9,3),bool)
    valid[4,:2]=False;pins=np.zeros(9,bool);pins[0]=True;pin=np.zeros(9);pin[0]=.0007
    out,stats=solve(domain,options,valid,pins,pin)
    assert out[0,0]==.0007 and out[1,1]==.002 and stats['coordinatewise_converged']
    assert np.isin(out.ravel()[1:],options[0]).all()


def test_empty_feasible_node_fails_closed():
    with pytest.raises(ValueError,match='feasible'):
        solve(np.ones((1,1),bool),np.zeros((1,2)),np.zeros((1,2),bool),np.zeros(1,bool),np.zeros(1))


def test_disconnected_grid_components_do_not_create_neighbors():
    domain=np.array([[True,False,True]]);options=np.array([[0,.1],[0,.1]])
    out,_=solve(domain,options,np.ones((2,2),bool),np.array([True,False]),np.array([.1,0]))
    np.testing.assert_array_equal(out,[[.1,0,0]])
