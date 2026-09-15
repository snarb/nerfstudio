import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import certified_conic_surface_step as original
import refined_conic_surface_step as refined


def test_only_internal_refinement_settings_change():
    baseline=original.solve.__globals__['SETTINGS']
    changed={k for k in refined.SETTINGS if refined.SETTINGS[k]!=baseline.get(k)}
    assert changed=={'iterative_refinement_reltol','iterative_refinement_abstol','iterative_refinement_max_iter'}
    assert refined.solve.__code__ is original.solve.__code__
    assert refined.solve.__globals__['certificate'] is original.solve.__globals__['certificate']


def test_exact_ball_control_passes_unchanged_certificate():
    x,r=refined.solve(sparse.eye(3),np.array([.002,0,0]),sparse.csr_matrix((0,3)),
                      np.zeros(0),np.zeros((1,3)),.001)
    np.testing.assert_allclose(x,[.001,0,0],atol=1e-9)
    assert r['certificate']['primal_violation']<=1e-8
