import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_surface_certificate import certify


def test_nearest_cap_can_discard_surrounding_measured_support():
    rng = np.random.default_rng(17)
    near = np.column_stack([rng.uniform(-.0008, -.0001, 24), rng.uniform(-.0005, .0005, 24), np.zeros(24)])
    angle = np.arange(12)*2*np.pi/12
    ring = np.column_stack([.002*np.cos(angle), .002*np.sin(angle), np.zeros(12)])
    seeds = np.concatenate([near, ring]); q = np.zeros(3); normal = np.array([0., 0., 1.])
    _, ids = cKDTree(seeds).query(q, k=24)
    ok, reason = certify(q, normal, seeds[ids])
    assert not ok and reason['reason'] == 'outside_seed_hull'
    ok, note = certify(q, normal, seeds[np.linalg.norm(seeds-q, axis=1) <= .003])
    assert ok and note['predicted_offset'] == note['loo_p90'] == 0


def test_more_seeds_does_not_relax_depth_offset():
    x, y = np.meshgrid(np.linspace(-.001, .001, 7), np.linspace(-.001, .001, 7))
    seeds = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .0006)])
    ok, note = certify(np.zeros(3), np.array([0., 0., 1.]), seeds, tolerance=.0005)
    assert not ok and note['predicted_offset'] > .0005
