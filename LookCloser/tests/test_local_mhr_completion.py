import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_local_mhr_completion as runner


def test_spec_rejects_frame_and_gate_overrides(tmp_path):
    spec = {key: str(tmp_path) for key in runner.FIELDS}
    spec['frame'] = '../001195'
    with pytest.raises(ValueError, match='six-digit'):
        runner.validate_spec(spec, tmp_path / 'output')
    spec['frame'] = '001195'; spec['depth_tolerance'] = .01
    with pytest.raises(ValueError, match='fields'):
        runner.validate_spec(spec, tmp_path / 'output')


def test_output_overlap_rejected(tmp_path):
    spec = {key: str(tmp_path) for key in runner.FIELDS}
    spec['frame'] = '001195'
    with pytest.raises(ValueError, match='overlaps'):
        runner.validate_spec(spec, tmp_path / 'output')


def test_seal_detects_changed_input(tmp_path):
    p = tmp_path / 'input.txt'; p.write_text('original')
    runner.save(tmp_path / 'final_seal.json', dict(status='passed', inventory={'input.txt': runner.sha(p)}, checked_bindings={}))
    checked = {}; runner.check_seal(tmp_path, checked)
    assert str(p) in checked
    p.write_text('changed')
    with pytest.raises(ValueError, match='Changed sealed'):
        runner.check_seal(tmp_path, {})


def test_adapters_only_rebind_paths_and_frame():
    import build_mhr_silhouette_patch_candidates as builder
    replacement = [("frame='001193'", 'frame=FRAME', 1)]
    _, proof = runner.adapt(builder, 'build', replacement, {'FRAME': '001197'})
    assert proof['generated_source'].replace('frame=FRAME', "frame='001193'") == proof['original_source']
    for guard in ['max_edge=.00075', 'distance<=.002', 'bd<=.003', 'dot>=.25', 'gap>=.00002']:
        assert guard in proof['generated_source']
    with pytest.raises(ValueError, match='adapter mismatch'):
        runner.adapt(builder, 'build', [("frame='001193'", 'frame=FRAME', 2)], {})


def test_fixed_recipe_rejects_zero_margin():
    import hashlib, json
    q = runner.read('/mnt/data/dec5_mhr_transfer_001195/silhouette100/protocol.json')['recipe']
    digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
    assert digest(q) == runner.RECIPE_SHA256
    q['boundary_tolerance_pixels'] = 0
    assert digest(q) != runner.RECIPE_SHA256
