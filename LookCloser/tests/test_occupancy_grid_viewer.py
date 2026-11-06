"""Protect XYZ indexing and exact cell alignment in the manual inspector."""
import base64
import importlib.util
from pathlib import Path

import numpy as np
import pytest

module_spec = importlib.util.spec_from_file_location(
    'grid_viewer', Path(__file__).parents[1] / 'scripts/export_occupancy_grid_viewer.py')
viewer = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(viewer)


def fixture_spec(tmp_path, second_bounds=None):
    shape = (3, 5, 7)  # Unequal axes expose accidental transposes.
    a = np.zeros(shape, dtype=bool)
    a[0, 4, 6] = a[2, 1, 0] = a[1, 2, 3] = True
    bounds = np.array([[-2., 4., -1.], [1., 9., 6.]])
    np.savez(tmp_path / 'a.npz', active=a[None], cascade_aabbs=bounds.reshape(1, 6))
    np.savez(tmp_path / 'b.npz', active=~a, aabb=bounds if second_bounds is None else second_bounds)
    return a, {'coordinate_frame': 'floor', 'layers': [
        {'name': 'A', 'archive': str(tmp_path / 'a.npz'), 'key': 'active'},
        {'name': 'B', 'archive': str(tmp_path / 'b.npz'), 'key': 'active', 'bounds_key': 'aabb'}]}


def test_browser_bit_addressing_xyz_and_padding(tmp_path):
    a, spec = fixture_spec(tmp_path)
    result = viewer.load_layers(spec)
    for layer, truth in zip(result['layers'], (a, ~a)):
        data = base64.b64decode(layer['bits'])
        for x, y, z in np.ndindex(a.shape):
            index = (x * a.shape[1] + y) * a.shape[2] + z
            assert bool((data[index >> 3] >> (7 - (index & 7))) & 1) == truth[x, y, z]
        assert layer['occupied'] == int(truth.sum())


def test_refuses_even_tiny_boundary_misalignment(tmp_path):
    bounds = np.array([[-2., 4., -1.], [1., 9., 6.]])
    bounds[0, 0] += 1e-12
    _, spec = fixture_spec(tmp_path, bounds)
    with pytest.raises(ValueError, match='Cell boundaries differ'):
        viewer.load_layers(spec)


def test_refuses_ambiguous_frame(tmp_path):
    _, spec = fixture_spec(tmp_path)
    spec['layers'][1]['coordinate_frame'] = 'camera'
    with pytest.raises(ValueError, match='Coordinate frames differ'):
        viewer.load_layers(spec)


def test_refuses_multiple_cascades(tmp_path):
    _, spec = fixture_spec(tmp_path)
    np.savez(tmp_path / 'a.npz', active=np.zeros((2, 3, 5, 7), dtype=bool),
             cascade_aabbs=np.zeros((2, 6)))
    with pytest.raises(ValueError, match='single cascade'):
        viewer.load_layers(spec)


def test_landmark_coordinate_mapping_and_outside_flag(tmp_path):
    _, spec = fixture_spec(tmp_path)
    spec['landmarks'] = [
        {'label': 'interior', 'xyz': [-.5, 6.5, 2.5]},
        {'label': 'upper boundary', 'xyz': [1, 9, 6]},
        {'label': 'outside', 'xyz': [1.001, 9, 6]}]
    points = viewer.load_layers(spec)['landmarks']
    assert points[0]['index'] == [1, 2, 3]
    assert points[1]['index'] == [2, 4, 6]
    assert points[2]['inside'] is False and points[2]['index'] is None
