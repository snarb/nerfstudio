import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mhr_surface_silhouette as surface
import run_mhr_surface_sampling_control as control


def test_association_preserves_constant_and_samples_fixed_vertices():
    a = surface.association(np.array([[0, 1, 2], [2, 3, 4]]), [True, False, False, False, False])
    assert a.shape == (4, 5)
    np.testing.assert_allclose(a.sum(1), 1)
    np.testing.assert_allclose(a.toarray()[:, :3], surface.BARY)


def test_barycentric_chain_rule_with_fixed_vertex(monkeypatch):
    v = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    t = np.array([[0, 1, 2]]); active = np.array([True, False, True])
    gradient = np.array([2., 3., 4.])

    def samples(points, rows, sdfs):
        return [(np.arange(len(points)), points @ gradient + 5., np.tile(gradient, (len(points), 1)))]

    monkeypatch.setattr(surface, 'silhouette_samples', samples)
    a, rhs = surface.linearize(v, t, active, [None], [None])
    assoc = surface.association(t, active)
    values = assoc @ v @ gradient + 5.
    frozen_weights = -rhs / values
    direction = np.array([[.1, .2, -.1], [.3, -.2, .5]])
    trial = v.copy(); trial[active] += 1e-7 * direction
    finite = ((assoc @ trial @ gradient + 5.) - values) / 1e-7
    np.testing.assert_allclose(a @ direction.ravel(), finite * frozen_weights, rtol=1e-7, atol=1e-8)


def test_interior_excursion_is_seen_when_vertices_are_inside(monkeypatch):
    v = np.array([[-2., -2, 0], [2., -2, 0], [0., 4, 0]])
    assert (1 - np.sum(v**2, axis=1) < 0).all()

    def samples(points, rows, sdfs):
        return [(np.arange(len(points)), 1 - np.sum(points**2, axis=1), -2 * points)]

    monkeypatch.setattr(surface, 'silhouette_samples', samples)
    a, rhs = surface.linearize(v, [[0, 1, 2]], [True] * 3, [None], [None])
    assert len(rhs) == 1 and rhs[0] < 0
    # At the exact symmetric interior maximum the Jacobian is zero: detection
    # alone cannot promise a useful descent direction or repair.
    np.testing.assert_allclose(a.toarray(), 0)


def test_invalid_weight_fails_closed():
    with pytest.raises(ValueError):
        surface.linearize(np.zeros((3, 3)), [[0, 1, 2]], [True] * 3, [None], [None], weight=np.nan)


def test_actual_projection_jacobian_matches_finite_difference():
    from fit_mhr_silhouette_conformance import project_jacobian, sample_sdf
    v = np.array([[1., 0, -4.], [2., 0, -4.], [1., 1., -4.]])
    t = np.array([[0, 1, 2]]); active = np.array([True, True, False])
    row = dict(transform_matrix=np.eye(4).tolist(), fl_x=16., fl_y=16., cx=32., cy=32.)
    y, x = np.mgrid[:64, :64]; sdf = (x + .3*y - 20).astype(np.float32)
    a, rhs = surface.linearize(v, t, active, [row], [sdf])
    mapping = surface.association(t, active)
    uv, _, _ = project_jacobian(mapping @ v, row)
    values, _ = sample_sdf(sdf, uv)
    weights = -rhs / values
    direction = np.array([[.13, -.2, .04], [-.17, .11, -.05]])
    trial = v.copy(); trial[active] += direction * 1e-6
    uv_trial, _, _ = project_jacobian(mapping @ trial, row)
    new_values, _ = sample_sdf(sdf, uv_trial)
    np.testing.assert_allclose(a @ direction.ravel(), weights * (new_values-values)/1e-6,
                               atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('arm', ['surface', 'vertex32'])
def test_adapter_freezes_both_controls(arm):
    source, driver = control.sources(arm)
    assert 'surface_sampling=' in source
    assert f"dec5_mhr_sampling_{arm}" in driver
    assert 'SURFACE_WRAPPER,SURFACE_HELPER' in driver
    if arm == 'surface':
        assert 'surface_helper.augment_optimizer(source)' in source
        assert "ns['surface_silhouette']=surface_helper.linearize" in source
    else:
        assert '32.*robust/' in source and '16.*robust/' not in source
