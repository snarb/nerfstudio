import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1]/'scripts/probe_marshawn_occupancy.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def test_beer_lambert_trace():
    a, t, w = probe.integrate([0, 1, 2], [1, 2, 3], [1, 2, 0])
    np.testing.assert_allclose(a, [1-np.exp(-1), 1-np.exp(-2), 0])
    np.testing.assert_allclose(t, [1, np.exp(-1), np.exp(-3)])
    assert abs(w.sum()-(1-np.exp(-3))) < 1e-12


def test_trace_empty_and_thin_surface():
    assert all(len(x) == 0 for x in probe.integrate([], [], []))
    a, t, w = probe.integrate([0, .001, .002], [.001, .002, .003], [0, 10000, 0])
    assert w[1] > .999 and t[2] < .0001


@pytest.mark.parametrize('args', [([0], [1], [-1]), ([0], [float('inf')], [1]), ([1, 0], [2, 1], [1, 1]), ([0, .5], [1, 2], [1, 1])])
def test_invalid_trace_rejected(args):
    with pytest.raises(ValueError): probe.integrate(*args)


def test_all_live_bounds_must_agree():
    from types import SimpleNamespace as NS
    box = np.array([[-1., -2., -3.], [1., 2., 3.]])
    model = NS(scene_aabb=box.flatten(), scene_box=NS(aabb=box), field=NS(aabb=box.copy()), collider=NS(scene_box=NS(aabb=box)), occupancy_grid=NS(aabbs=box.reshape(1, 6)))
    report = probe.assert_consistent_bounds(model)
    assert len(report) == 5
    model.field.aabb[0, 0] -= 1e-4
    with pytest.raises(ValueError, match='field.aabb'): probe.assert_consistent_bounds(model)
    model.field.aabb = box+5e-7
    assert probe.assert_consistent_bounds(model)
    model.collider.scene_box.aabb = box+1e-3
    with pytest.raises(ValueError, match='collider'): probe.assert_consistent_bounds(model)


def test_trained_halo_is_not_dilated_twice():
    import sys
    sys.path.insert(0, str(Path(probe.__file__).parent))
    from marshawn_occupancy_audit import dilate_one
    values = np.zeros((1, 7, 7, 7)); values[0, 3, 3, 3] = 1
    core = values > 0; native = dilate_one(core)
    profiles, policy = probe.occupancy_profiles(native, values, 1, 5000, 4096)
    assert profiles['base'].sum() == 1 and profiles['halo1'].sum() == 27
    assert np.array_equal(profiles['halo1'], native) and policy['native_matches_declared_policy']
    with pytest.raises(ValueError, match='disagrees'):
        probe.occupancy_profiles(dilate_one(native), values, 1, 5000, 4096)


def test_binary_warmup_has_no_claimed_one_cell_comparison():
    profiles, policy = probe.occupancy_profiles(np.ones((1, 3, 3, 3), bool), np.zeros(27), 1, 100, 4096)
    assert set(profiles) == {'native', 'all_cells'} and not policy['halo1_available']
