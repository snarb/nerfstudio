import inspect
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fit_mhr_silhouette_conformance as original
from continue_mhr_silhouette_convergence import instrument,STOP


def test_instrumentation_only_adds_observer():
    source=inspect.getsource(original.optimize)
    added='\n        if continuation_observe(locals()):\n            break'
    changed=instrument(source)
    assert changed.count(added)==1
    assert changed.replace(added,'')==source


def test_convergence_protocol_is_bounded_not_trust_cap_success():
    assert STOP['maximum_outer_iterations']==100
    assert STOP['consecutive_small_steps']==3
    assert STOP['unconstrained_step_tolerance'] < original.RECIPE['maximum_step']
    assert STOP['collapsed_area_ratio']>0
