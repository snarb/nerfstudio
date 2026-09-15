import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if not Path('/home/brans/lookcloser_temp/clarabel_correction_20260915/clarabel').is_dir():
    pytest.skip('Optional private Clarabel dependency absent',allow_module_level=True)
from conic_surface_step import solve,certificate


def test_ball_projection_and_offset():
    rhs=np.array([.002,.003,.001]);offset=np.array([[.0002,-.0001,.0003]])
    x,record=solve(sparse.eye(3),rhs,sparse.csc_matrix((0,3)),[],offset,.001)
    direction=rhs+offset.ravel();expected=.001*direction/np.linalg.norm(direction)-offset.ravel()
    np.testing.assert_allclose(x,expected,atol=1e-9)
    assert record['maximum_displacement']<=.001+1e-11


def test_linear_bound_and_ball():
    x,_=solve(sparse.eye(3),[-.003,.003,0],sparse.csc_matrix([[1.,0,0]]),[.0002],np.zeros((1,3)),.001)
    np.testing.assert_allclose(x,[.0002,np.sqrt(.001**2-.0002**2),0],atol=1e-9)


def test_feasible_stationary_noncomplementary_is_rejected():
    # min .5(x-1)^2, x>=0: x=2,z=1 has stationarity but is not optimal.
    with pytest.raises(ValueError,match='complementarity'):
        certificate(sparse.eye(1),np.array([-1.]),sparse.csc_matrix([[-1.]]),
                    np.zeros(1),np.array([2.]),np.array([1.]),1)


def test_dual_outside_lorentz_cone_is_rejected():
    with pytest.raises(ValueError,match='dual'):
        certificate(sparse.eye(3),np.zeros(3),sparse.csc_matrix((4,3)),
                    np.array([1.,0,0,0]),np.zeros(3),np.array([0.,1.,0,0]),0)
