import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_semantic_node_cell_bridge import LogitNodeCellBridge, fresh_dilation


def bridge(nodes, bounds=((0, 0, 0), (1, 1, 1)), **kw):
    return LogitNodeCellBridge(nodes, bounds, world_to_box=np.eye(4), selector=kw.pop('selector', 'hard_theta05'), **kw)


def test_knot_peak_inside_native_cell_not_visible_at_eight_corners():
    nodes = np.full((5, 5, 5), -10.); nodes[2, 2, 2] = 10.
    gate = bridge(nodes, coordinate_guard=0, numerical_logit_guard=0)
    assert gate.cells_supported([[.25]*3], [[.75]*3])[0]
    # All eight cell corners are negative nodes, but interior centre is positive.
    assert (nodes[np.ix_([1, 3], [1, 3], [1, 3])] < 0).all()


@pytest.mark.parametrize('align', [True, False])
@pytest.mark.parametrize('selector', ['hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05'])
def test_independent_torch_trilinear_samples_never_rejected(align, selector):
    rng = np.random.default_rng(812); nodes = rng.normal(-2.5, 1.8, (7, 9, 11))
    bounds = np.array([[-.84, -.21, -.07], [.15, .42, .63]])
    gate = LogitNodeCellBridge(nodes, bounds, world_to_box=np.eye(4), selector=selector, align_corners=align)
    target = np.array([[-.90, -.25, -.12], [.21, .49, .69]])
    cells = gate.map_cascades([target], [13, 17, 19])[0]
    points = rng.uniform(target[0], target[1], (10000, 3))
    unit = (points-bounds[0])/(bounds[1]-bounds[0]); inside = ((unit >= 0) & (unit <= 1)).all(1)
    coords = torch.tensor((2*unit-1)[:, [2, 1, 0]], dtype=torch.float64).reshape(1, 1, 1, -1, 3)
    p = F.grid_sample(torch.tensor(nodes)[None, None], coords, align_corners=align, padding_mode='border').reshape(-1).sigmoid().numpy()*inside
    selected = p > .1 if selector.startswith('exactzero') else p >= (.5 if selector == 'hard_theta05' else .1)
    ids = np.floor((points-target[0])/(target[1]-target[0])*[13, 17, 19]).astype(int)
    assert cells[tuple(ids[selected].T)].all()
    assert not torch.cuda.is_initialized()


def test_threshold_equality_is_active_hard_and_zero_smooth_without_slack():
    threshold = np.log(.1)-np.log1p(-.1); nodes = np.full((3, 4, 5), threshold)
    kw = dict(coordinate_guard=0, numerical_logit_guard=0)
    assert bridge(nodes, selector='hard_theta01', **kw).cells_supported([[0]*3], [[1]*3])[0]
    assert not bridge(nodes, selector='exactzero_smoothstep01_05', **kw).cells_supported([[0]*3], [[1]*3])[0]
    assert bridge(nodes, selector='exactzero_smoothstep01_05', numerical_logit_guard=1e-5).cells_supported([[0]*3], [[1]*3])[0]


def test_soft_small_probability_has_no_finite_density_bound():
    gate = bridge(np.full((2, 3, 4), -70.), selector='soft_probability')
    assert gate.cells_supported([[.1]*3], [[.9]*3])[0]
    assert not gate.cells_supported([[2]*3], [[3]*3])[0]
    assert gate.cells_supported([[1]*3], [[1]*3])[0]


def test_native128_node127_pitch_and_fresh_halo_are_distinct():
    nodes = np.full((128, 128, 128), -10., dtype=np.float32); nodes[64, 64, 64] = 10.
    bounds = np.array([[-.83999997, -.21000001, -.06999999], [.15000004, .41999999, .63]])
    gate = bridge(nodes, bounds)
    core = gate.map_cascades([bounds], 128, radius=0)[0]
    halo = gate.map_cascades([bounds], 128, radius=1)[0]
    assert core.sum() > 1
    assert np.array_equal(halo, fresh_dilation(core, 1))
    assert halo.sum() > core.sum()
    assert np.array_equal(gate.map_cascades([bounds], 128, radius=0)[0], core)


def test_axes_border_partial_overlap_and_independent_cascades():
    nodes = np.full((3, 5, 7), -10.); nodes[0, 4, 6] = 10.
    gate = bridge(nodes, [[1, 2, 3], [3, 8, 17]])
    assert gate.cells_supported([[.9, 7.9, 16.9]], [[1.1, 8.1, 17.1]])[0]
    assert not gate.cells_supported([[2.9, 2., 3.]], [[3., 2.1, 3.1]])[0]
    boxes = np.array([[[1, 2, 3], [3, 8, 17]], [[10, 20, 30], [12, 26, 44]]])
    out = gate.map_cascades(boxes, [4, 6, 8])
    assert out[0].any() and not out[1].any()


def test_nonfinite_transform_and_invalid_geometry_rejected():
    nodes = np.zeros((3, 3, 3)); nodes[0, 0, 0] = np.nan
    with pytest.raises(ValueError): bridge(nodes)
    with pytest.raises(ValueError): LogitNodeCellBridge(np.zeros((3, 3, 3)), [[0]*3, [1]*3], selector='hard_theta05', world_to_box=np.diag([-1, 1, 1, 1]))
    with pytest.raises(ValueError): bridge(np.zeros((3, 3, 3))).cells_supported([[1]*3], [[0]*3])


def test_threshold_guard_and_class_value_monotonicity():
    rng = np.random.default_rng(7); nodes = rng.normal(-2, 2, (5, 7, 9))
    a = bridge(nodes, selector='hard_theta05').map_cascades([[[0]*3, [1]*3]], 17)
    b = bridge(nodes, selector='hard_theta01').map_cascades([[[0]*3, [1]*3]], 17)
    c = bridge(nodes+3, selector='hard_theta01').map_cascades([[[0]*3, [1]*3]], 17)
    assert (a <= b).all() and (b <= c).all()


def test_fp32_nextafter_edges_and_node_planes_have_no_positive_rejection():
    nodes = np.full((9, 11, 13), -6., dtype=np.float32); nodes[4, 5, 6] = 6.
    gate = bridge(nodes)
    cells = gate.map_cascades([[[0]*3, [1]*3]], [16, 18, 20])[0]
    q = np.linspace(0, 1, 65, dtype=np.float32)
    q = np.concatenate([q, np.nextafter(q, np.float32(-np.inf)), np.nextafter(q, np.float32(np.inf))])
    points = np.stack([q, np.full_like(q, .5), np.full_like(q, .5)], 1)
    points = points[((points >= 0) & (points <= 1)).all(1)]
    coords = torch.from_numpy((2*points-1)[:, [2, 1, 0]].copy()).reshape(1, 1, 1, -1, 3)
    p = F.grid_sample(torch.from_numpy(nodes)[None, None], coords, align_corners=True, padding_mode='border').reshape(-1).sigmoid().numpy()
    ids = np.minimum(np.floor(points*[16, 18, 20]).astype(int), [15, 17, 19])
    assert cells[tuple(ids[p >= .5].T)].all()
