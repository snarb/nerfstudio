"""Scope guards, not evidence of visual/anatomical correctness."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_mhr_production_patch_control as control
from review_mhr_production_patch_control import check_recipe
from localize_mhr_production_patch_side_effects import side_effect_masks


def test_new_black_is_distinct_from_old_black_and_new_uncolored_surface():
    before = np.array([[[0]*3, [2]*3, [0]*3, [3]*3]], np.uint8)
    after = np.array([[[0]*3, [0]*3, [0]*3, [3]*3]], np.uint8)
    old_depth = np.array([[0., 1., 1., 1.]])
    new_depth = np.array([[1., 1., 1., 0.]])
    result = side_effect_masks(before, old_depth, after, new_depth)
    np.testing.assert_array_equal(result['new_geometry_without_rgb'], [[True, False, False, False]])
    np.testing.assert_array_equal(result['newly_black_rgb'], [[False, True, False, False]])
    np.testing.assert_array_equal(result['lost_geometry'], [[False, False, False, True]])


@pytest.mark.parametrize('key,value', [
    ('source_incidence_power', 8), ('static_registration', True),
    ('pixel_fallback_angle_prior', False), ('texture_source_prior', 'late'),
])
def test_old_texture_policy_is_not_a_current_recipe_control(key, value):
    recipe = dict(source_incidence_power=2, static_registration=False,
                  pixel_fallback_angle_prior=True,
                  texture_source_prior='target_angle_before_incidence_clip')
    check_recipe(dict(recipe=recipe))
    recipe[key] = value
    with pytest.raises(AssertionError):
        check_recipe(dict(recipe=recipe))


@pytest.mark.parametrize('fail', [False, True])
def test_rebinding_is_explicit_copied_and_restored(monkeypatch, tmp_path, fail):
    proof = dict(production_mesh='production.ply', production_mesh_sha256='production-hash')
    original = dict(original_mesh='raw.ply', original_mesh_sha256='raw-hash', unchanged=[1])
    before = deepcopy(original)
    writes = []
    def reader(path):
        return original
    def writer(path, data):
        writes.append((path, data))
    def builder(dest):
        rebound = control.builder.read(control.builder.PRIOR/'protocol.json')
        assert rebound['original_mesh'] == 'production.ply'
        assert rebound['original_mesh_sha256'] == 'production-hash'
        assert rebound['unchanged'] == [1]
        rebound['unchanged'].append(2)
        assert control.builder.read(tmp_path/'unrelated.json') is original
        control.builder.save(dest/'request.json', dict(candidate=True))
        if fail:
            raise RuntimeError('simulated builder failure')
    monkeypatch.setattr(control, 'binding', lambda: proof)
    monkeypatch.setattr(control.builder, 'read', reader)
    monkeypatch.setattr(control.builder, 'save', writer)
    monkeypatch.setattr(control.builder, 'build', builder)
    if fail:
        with pytest.raises(RuntimeError, match='simulated'):
            control.build(tmp_path)
    else:
        control.build(tmp_path)
    assert control.builder.read is reader
    assert control.builder.save is writer
    assert original == before
    assert writes == [(tmp_path/'request.json', dict(candidate=True, production_base_binding=proof))]
