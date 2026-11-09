import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1]/'scripts/marshawn_occupancy_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    grid = np.zeros((2, 2, 3, 4), bool)
    grid[0, 1, 2, 3] = True
    grid[1, 0, 0, 0] = True
    boxes = np.array([[0, 0, 0, 2, 3, 4], [-2, -3, -4, 2, 3, 4]], float)
    return grid, boxes


def test_lookup_each_cascade_non_cubic_and_half_open():
    grid, boxes = fixture()
    result = audit.lookup(grid, boxes, [1.5, 2.5, 3.5])
    assert result[0]['index_xyz'] == [1, 2, 3] and result[0]['occupied']
    assert result[1]['index_xyz'] == [1, 2, 3] and not result[1]['occupied']
    assert not any(x['inside'] for x in audit.lookup(grid, boxes, [2, 3, 4]))
    with pytest.raises(ValueError): audit.lookup(grid, boxes, [float('nan'), 0, 0])


def test_ray_all_cells_exact_intervals_and_reverse():
    grid, boxes = fixture()
    ray = audit.ray_intervals(grid, boxes, [-1, 2.5, 3.5], [1, 0, 0])[0]
    assert (ray['t_enter'], ray['t_exit']) == (1, 3)
    assert [r['index_xyz'] for r in ray['intervals']] == [[0, 2, 3], [1, 2, 3]]
    assert [r['occupied'] for r in ray['intervals']] == [False, True]
    assert [(r['t_start'], r['t_end']) for r in ray['intervals']] == [(1, 2), (2, 3)]
    reverse = audit.ray_intervals(grid, boxes, [3, 2.5, 3.5], [-1, 0, 0])[0]
    assert [r['occupied'] for r in reverse['intervals']] == [True, False]


def test_parallel_misses_inside_near_clip_and_zero_direction():
    grid, boxes = fixture()
    assert not audit.ray_intervals(grid, boxes, [-1, 3, 0], [1, 0, 0])[0]['hit']
    ray = audit.ray_intervals(grid, boxes, [0.5, 0.5, 0.5], [2, 0, 0], near=.1, far=.5)[0]
    assert ray['t_enter'] == .1 and ray['t_exit'] == .5
    assert len(ray['intervals']) == 2
    with pytest.raises(ValueError): audit.ray_intervals(grid, boxes, [0, 0, 0], [0, 0, 0])


def test_dilation_boundary_no_wrap_or_cross_cascade():
    grid = np.zeros((2, 4, 4, 4), bool); grid[0, 0, 0, 0] = True
    out = audit.dilate_one(grid)
    assert out[0].sum() == 8 and out[1].sum() == 0 and not out[0, 3].any()
    assert grid.sum() == 1


def test_every_slice_has_atlas_page(tmp_path):
    grid, boxes = fixture()
    pages = audit.write_atlas(grid, boxes, tmp_path, 1)
    assert len(pages) == 6 and (tmp_path/'index.html').is_file()
    for level in range(2):
        for axis, size in zip('xyz', grid.shape[1:]):
            selected = [p for p in pages if p['level'] == level and p['axis'] == axis]
            assert [i for p in selected for i in range(p['slice_start'], p['slice_end']+1)] == list(range(size))
            assert all((tmp_path/p['path']).is_file() for p in selected)


def test_invalid_box_and_grid():
    grid, boxes = fixture()
    with pytest.raises(ValueError): audit.validate_grid(grid.astype(float), boxes)
    boxes[0, 3] = 0
    with pytest.raises(ValueError): audit.validate_grid(grid, boxes)


def test_cli_npz_queries_and_dilation(tmp_path):
    import subprocess
    import sys
    import json
    grid, boxes = fixture()
    source = tmp_path/'grid.npz'; np.savez_compressed(source, occupied=grid, aabbs=boxes)
    queries = tmp_path/'queries.json'
    queries.write_text(json.dumps({'points': [[1.5, 2.5, 3.5]], 'rays': [{'origin': [-1, 2.5, 3.5], 'direction': [1, 0, 0]}]}))
    output = tmp_path/'review'
    result = subprocess.run([sys.executable, str(Path(audit.__file__)), '--grid', str(source), '--output', str(output), '--queries', str(queries), '--dilation-radius', '1'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/'audit.json').read_text())
    assert report['dilation_radius'] == 1
    assert len(report['source_sha256']) == 64 and len(report['pages']) == 6
    assert report['points'][0]['levels'][0]['occupied']
    assert report['rays'][0]['levels'][0]['hit']


def test_cell_index_roundtrip_and_bounds():
    grid, boxes = fixture()
    record = audit.cell_record(grid, boxes, 0, [1, 2, 3])
    assert record['center_xyz'] == [1.5, 2.5, 3.5] and record['occupied']
    assert record['bounds'] == [1, 2, 3, 2, 3, 4]
    with pytest.raises(ValueError): audit.cell_record(grid, boxes, 0, [2, 0, 0])
    with pytest.raises(ValueError): audit.cell_record(grid, boxes, 0, [1.5, 0, 0])


def test_core_guard_and_unknown_admission_are_distinct():
    values = np.zeros((1, 5, 5, 5), np.float32); values[0, 2, 2, 2] = 1
    core = values > .001
    grid = audit.dilate_one(core); grid[0, 0, 0, 0] = True
    labels, threshold = audit.classify_cells(grid, values, 1)
    assert threshold == pytest.approx(.008)
    assert (labels == 1).sum() == 1 and (labels == 2).sum() == 26
    assert labels[0, 0, 0, 0] == 3 and labels[0, 4, 4, 4] == 0
    grid[0, 2, 2, 2] = False
    labels, _ = audit.classify_cells(grid, values, 1)
    assert labels[0, 2, 2, 2] == 4
    assert audit.classify_cells(grid, None, 1) == (None, None)


def test_checkpoint_shared_grid_aliases(tmp_path):
    torch = pytest.importorskip('torch')
    grid, boxes = fixture()
    state = {}
    for prefix in ['_model.', '_model.adaptive_sampler.']:
        state[prefix+'occupancy_grid.binaries'] = torch.from_numpy(grid)
        state[prefix+'occupancy_grid.aabbs'] = torch.from_numpy(boxes)
        state[prefix+'occupancy_grid.occs'] = torch.zeros(grid.size)
    path = tmp_path/'state.ckpt'; torch.save({'pipeline': state}, path)
    actual, _, values = audit.load_grid_details(path)
    assert np.array_equal(actual, grid) and values.shape == grid.shape
    state['_model.adaptive_sampler.occupancy_grid.binaries'] = ~torch.from_numpy(grid)
    torch.save({'pipeline': state}, path)
    with pytest.raises(ValueError, match='Conflicting'): audit.load_grid_details(path)
