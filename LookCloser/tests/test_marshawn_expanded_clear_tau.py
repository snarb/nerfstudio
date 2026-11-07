import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import marshawn_expanded_clear_tau as expanded
from marshawn_clear_tau import QualifiedClearTau as Original


def targets(n):
    return {'targets': [dict(origin=[0., 0., .5], direction=[0., 0., -1.], near=.1, far=.3) for _ in range(n)]}


def test_small_draw_is_exact_original_and_expansion_preserves_global_rng():
    model = SimpleNamespace(device=torch.device('cpu'))
    old = Original(model, targets(243), 'a'*64, .001)
    small = expanded.QualifiedClearTau(model, targets(243), 'a'*64, .001)
    large = expanded.QualifiedClearTau(model, targets(7369), 'b'*64, .001)
    before = torch.get_rng_state().clone()
    for _ in range(4):
        oi, oj, oh = old.draw(); si, sj, sh = small.draw(); li, lj, _ = large.draw()
        assert oh == sh and torch.equal(oi, si) and torch.equal(oj, sj)
        assert torch.equal(sj, lj) and int(li.max()) < 7369
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(small.generator.get_state(), large.generator.get_state())


def test_expanded_resume_preserves_stream_and_rejects_population_change():
    model = SimpleNamespace(device=torch.device('cpu'))
    a = expanded.QualifiedClearTau(model, targets(7369), 'b'*64, .001); a.draw()
    state = copy.deepcopy(a.state_dict()); expected = a.draw()
    b = expanded.QualifiedClearTau(model, targets(7369), 'b'*64, .001); b.load_state_dict(state)
    actual = b.draw(); assert expected[2] == actual[2] and a.chain == b.chain
    wrong = copy.deepcopy(state); wrong['population'] = 243
    with pytest.raises(ValueError, match='population'): b.load_state_dict(wrong)
    for n, coefficient in [(100, .001), (7369, 0.), (243, .0001)]:
        with pytest.raises(ValueError): expanded.QualifiedClearTau(model, targets(n), 'b'*64, coefficient)


def test_geometry_rejects_floor_contact_or_invalid_interval():
    frame = dict(transform_matrix=np.eye(4).tolist(), cx=.5, cy=.5, fl_x=100., fl_y=100.)
    frame['transform_matrix'][2][3] = .8
    box = np.array([[-.5, -.5, .2], [.5, .5, .6]])
    o, d, near, far, floor = expanded.ray_geometry(frame, box, 1, 1)
    assert floor > far > near and (o+d*far)[2] > .1
    bad = box.copy(); bad[0, 2] = .09
    with pytest.raises(ValueError, match='floor'): expanded.ray_geometry(frame, bad, 1, 1)
    with pytest.raises(ValueError, match='floor'): expanded.ray_geometry(frame, box, 1000, 1000)


def test_population_and_source_digest_fail_closed_before_source_reads(tmp_path):
    p = tmp_path/'targets.json'; p.write_text('{}')
    for population, sha in [('small243', '0'*64), ('unknown', '0'*64), ('expanded7369', '0'*64)]:
        with pytest.raises(ValueError): expanded.validate_targets(p, sha, tmp_path, tmp_path/'absent', population)
