import sys
from pathlib import Path
import numpy as np
from scipy import sparse
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if not Path('/home/brans/lookcloser_temp/osqp_correction_20260915/osqp').is_dir():
    pytest.skip('Private optional OSQP experiment dependency not installed',allow_module_level=True)
from osqp_surface_step import lower_bound_lsq
from constrained_surface_step import lower_bound_lsq as reference


def test_osqp_matches_reference_on_fixed_random_qps():
    rng=np.random.default_rng(29)
    for _ in range(12):
        m=sparse.csc_matrix(rng.normal(size=(30,10)));rhs=rng.normal(size=30)
        a=sparse.csc_matrix(rng.normal(size=(16,10)));b=-rng.uniform(.01,.1,size=16)
        expected,_=reference(m,rhs,a,b);actual,r=lower_bound_lsq(m,rhs,a,b)
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-7)
        assert r['minimum_slack']>=-1e-11


def test_osqp_redundant_supporting_planes():
    a=sparse.csr_matrix([[1.,1.],[1.,0.],[2.,2.]])
    x,_=lower_bound_lsq(sparse.eye(2),[0.,0.],a,[2.,.2,3.])
    np.testing.assert_allclose(x,[1.,1.],atol=1e-8)
