import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from transfer_mhr_radius_seed_control import adapted_sources, replace_once


def test_exact_adapter_preserves_numerical_selection_and_guard():
    run, audit = adapted_sources()
    assert "frame=spec['frame']" in run
    assert 'tree.query_ball_point(v[qid], .003)' in run
    assert 'tolerance=.0005' in run and 'admission.native_guard' in run
    assert 'distances = np.linalg.norm(seeds - v[qid], axis=1)' in audit
    assert 'admission.measured_pixel_veto' in audit
    assert "source / 'admission/certified_conic" not in run + audit
    assert "source / 'candidates/certified_conic" not in run + audit
    assert "dest = out / 'admission/certified_conic'" in run


def test_adapter_fails_on_missing_or_ambiguous_frozen_source():
    for text in ['none', 'old old']:
        with pytest.raises(ValueError): replace_once(text, 'old', 'new')
