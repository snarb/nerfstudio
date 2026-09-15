import sys
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from constrained_surface_step import lower_bound_lsq,oriented_area_constraints


def test_known_bound_solution_and_kkt():
    m=sparse.eye(2);rhs=np.array([-2.,3.]);a=sparse.csr_matrix([[1.,0.],[0.,-1.]])
    x,r=lower_bound_lsq(m,rhs,a,np.array([0.,-1.]))
    np.testing.assert_allclose(x,[0.,1.],atol=1e-12)
    assert min(r['multipliers'])>=0


def test_inactive_bounds_preserve_unconstrained_solution():
    x,r=lower_bound_lsq(sparse.eye(2),[1.,2.],sparse.eye(2),[-1.,-1.])
    np.testing.assert_allclose(x,[1.,2.]);assert not r['active']


def test_coupled_bound_and_redundancy():
    a=sparse.csr_matrix([[1.,1.],[1.,0.],[2.,2.]])
    x,_=lower_bound_lsq(sparse.eye(2),[0.,0.],a,[2.,.2,3.])
    np.testing.assert_allclose(x,[1.,1.],atol=1e-10)


def test_signed_area_gradient_matches_finite_difference():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    t=np.array([[0,1,2]]);n=np.array([[0.,0.,1.]])
    a,b,selected=oriented_area_constraints(v,t,n,np.array([True]),np.arange(3),np.array([.01]))
    direction=np.array([[.2,.1,.4],[-.3,.3,.1],[.3,-.5,.2]])
    eps=1e-7;trial=v+eps*direction
    derivative=(np.cross(trial[1]-trial[0],trial[2]-trial[0])[2]-1)/eps
    # Unnormalized gradient has length 2 on this triangle.
    np.testing.assert_allclose((a@direction.ravel())[0]*2,derivative,atol=1e-6)
    np.testing.assert_allclose(b,[-.495]);np.testing.assert_array_equal(selected,[0])
