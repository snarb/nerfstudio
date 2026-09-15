"""Discrete correction safeguards, not anatomical-quality tests."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_mhr_guarded_correction import StepGuard, SETTINGS


def guard_fixture():
    v=np.array([[0.,0.,0.],[.001,0.,0.],[0.,.001,0.],[2.,2.,2.]])
    return v,StepGuard(v,np.array([[0,1,2]]),v)


def test_translation_is_backtracked_and_unselected_vertex_preserved():
    v,g=guard_fixture();ids=np.array([0,1,2]);step=np.tile([.002,0.,0.],(3,1))
    result,applied=g.update(v,ids,step)
    assert g.history[-1]['factor']==.5
    np.testing.assert_array_equal(result[3],v[3])
    np.testing.assert_allclose(result[ids]-v[ids],applied)
    assert g.check(result)[0]


def test_collapse_is_rejected():
    v,g=guard_fixture();trial=v.copy();trial[2]=trial[0]
    ok,record=g.check(trial)
    assert not ok and record['reason']=='area'


def test_normal_reversal_is_rejected():
    v,_=guard_fixture();v[2]=[0.,.0001,0.]
    g=StepGuard(v,np.array([[0,1,2]]),v);trial=v.copy();trial[2]=[0.,-.0001,0.]
    ok,record=g.check(trial)
    assert not ok and record['reason']=='normal_rotation'


def test_no_admissible_step_returns_original_and_stall(monkeypatch):
    v,g=guard_fixture()
    monkeypatch.setattr(g,'check',lambda trial:(False,{'reason':'test_rejection'}))
    result,step=g.update(v,np.array([0]),np.array([[.0001,0.,0.]]))
    np.testing.assert_array_equal(result,v);assert not step.any() and g.stalled
    assert len(g.history[-1]['rejected'])==SETTINGS['maximum_halvings']+1
