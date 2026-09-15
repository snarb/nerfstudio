import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from body_surface_neighborhood import select_seeds
from local_surface_certificate import certify


def test_balances_crowded_edge_but_requires_actual_enclosing_support():
    angles = np.linspace(-np.pi, np.pi, 32, endpoint=False) + .02
    ring = np.column_stack([.004 * np.cos(angles), .004 * np.sin(angles), np.zeros(32)])
    edge = np.column_stack([np.full(80, .002), np.linspace(-.0002, .0002, 80), np.zeros(80)])
    points = np.concatenate([edge, ring])
    normals = np.tile([0, 0, 1.], (len(points), 1))
    ids = select_seeds([0, 0, 0], [0, 0, 1], points, normals)
    assert len(ids) <= 24 and len(np.unique(ids)) == len(ids)
    assert np.any(points[ids, 0] < 0) and np.any(points[ids, 0] > 0)
    assert certify([0, 0, 0], [0, 0, 1], points[ids])[0]
    one_side = points[points[:, 0] > 0]
    selected = select_seeds([0, 0, 0], [0, 0, 1], one_side, np.tile([0, 0, 1], (len(one_side), 1)))
    assert not certify([0, 0, 0], [0, 0, 1], one_side[selected])[0]


def test_radius_normal_filter_and_determinism():
    points = np.array([[.001, 0, 0], [.001, 0, 0], [.007, 0, 0], [0, .001, 0], [np.nan, 0, 0]])
    normals = np.tile([0., 0, 1], (len(points), 1)); normals[3] *= -1
    a = select_seeds([0, 0, 0], [0, 0, 1], points, normals, per_sector=1)
    np.testing.assert_array_equal(a, [0])
    np.testing.assert_array_equal(a, select_seeds([0, 0, 0], [0, 0, 1], points, normals, per_sector=1))
    with pytest.raises(ValueError):
        select_seeds([0, 0, 0], [0, 0, 0], points, normals)
