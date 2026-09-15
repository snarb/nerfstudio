import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if not Path('/home/brans/lookcloser_temp/clarabel_correction_20260915/clarabel').is_dir():
    pytest.skip('Optional private Clarabel dependency absent',allow_module_level=True)
from certified_conic_surface_step import certificate,solve
import certified_conic_surface_step as backend
from types import SimpleNamespace


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_certificate_rejected(value):
    with pytest.raises(ValueError,match='Nonfinite'):
        certificate(sparse.eye(1),np.array([value]),sparse.csc_matrix([[-1.]]),
                    np.zeros(1),np.array([2.]),np.array([1.]),1)


def test_certificate_cannot_excuse_noncomplementarity():
    with pytest.raises(ValueError,match='complementarity'):
        certificate(sparse.eye(1),np.array([-1.]),sparse.csc_matrix([[-1.]]),
                    np.zeros(1),np.array([2.]),np.array([1.]),1)


def test_same_constrained_solution():
    x,r=solve(sparse.eye(3),[-.003,.003,0],sparse.csc_matrix([[1.,0,0]]),[.0002],np.zeros((1,3)),.001)
    np.testing.assert_allclose(x,[.0002,np.sqrt(.001**2-.0002**2),0],atol=1e-9)
    assert r['certificate']['complementarity']<=1e-7


@pytest.mark.parametrize('status,position,accepted',[
    ('AlmostSolved',[0.,0,0],True),('AlmostSolved',[2.,0,0],False),
    ('MaxIterations',[0.,0,0],False)])
def test_status_does_not_bypass_certificate(monkeypatch,status,position,accepted):
    result=SimpleNamespace(status=status,x=position,z=[0.,0,0,0],iterations=1)
    monkeypatch.setattr(backend.original.clarabel,'DefaultSolver',lambda *args: SimpleNamespace(solve=lambda:result))
    args=(sparse.eye(3),np.zeros(3),sparse.csc_matrix((0,3)),[],np.zeros((1,3)),.001)
    if accepted:
        x,r=solve(*args);np.testing.assert_array_equal(x,np.zeros(3));assert r['status']=='AlmostSolved'
    else:
        with pytest.raises(ValueError):solve(*args)
