from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_marshawn_tuning_checkpoint import fixed_mask_pixels, ray_for_pixel, witness_membership, summarize_traces, actual_profiles


def test_gpu_overlap_requires_explicit_opt_in(monkeypatch):
    import audit_marshawn_tuning_checkpoint as audit
    def query(command, **kwargs):
        return '0, GPU-test\n' if '--query-gpu=index,uuid' in command else 'GPU-test, 123\n'
    monkeypatch.setattr(audit.subprocess, 'check_output', query)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'original')
    with pytest.raises(RuntimeError, match='occupied'):
        audit.require_idle_gpu(0)
    assert audit.os.environ['CUDA_VISIBLE_DEVICES'] == 'original'
    assert audit.require_idle_gpu(0, allow_readonly_overlap=True) == 'GPU-test'


def test_shirt_mass_retains_transmission_instead_of_conditioning_it_away(tmp_path):
    from compare_marshawn_shirt_traces import masses
    np.savez(tmp_path/'ray.npz', t_start=np.array([.09,.49]), t_end=np.array([.11,.51]),
             weights=np.array([.25,.5]))
    row = dict(profiles=dict(uniform2x=dict(trace='ray.npz')),
               pixel=dict(witness_xyz=[0.,0.,.5]), origin=[0.,0.,0.], direction=[0.,0.,1.])
    result, _, _, _ = masses(tmp_path, row, 'uniform2x')
    assert result == pytest.approx([.25,.5,0.,.25])


def test_fixed_queries_separate_boundaries_and_unknown_exterior():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 35:65] = True
    first, second = fixed_mask_pixels(mask), fixed_mask_pixels(mask)
    assert first == second
    assert len(first) == 8
    for row in first:
        assert bool(mask[row['y'], row['x']]) == (row['semantic'] != 'outside_actor_unknown_geometry')
    with pytest.raises(ValueError):
        fixed_mask_pixels(np.zeros_like(mask))


def test_prepared_opengl_camera_center_ray():
    frame = dict(transform_matrix=np.eye(4).tolist(), fl_x=100., fl_y=100., cx=50.5, cy=50.5)
    ray = ray_for_pixel(frame, 50, 50)
    assert ray['origin'] == [0., 0., 0.]
    assert ray['direction'] == [0., 0., -1.]
    ray = ray_for_pixel(frame, 60, 40)
    assert ray['direction'][0] > 0 and ray['direction'][1] > 0
    assert np.linalg.norm(ray['direction']) == pytest.approx(1.)


def test_witnesses_never_label_unobserved_cells_free():
    core = np.zeros((1, 2, 2, 2), bool)
    core[0, 0, 0, 0] = True
    active = core.copy()
    active[0, 1, 1, 1] = True
    boxes = np.array([[0, 0, 0, 2, 2, 2.]])
    points = np.array([[.5, .5, .5], [1.5, 1.5, 1.5], [.5, 1.5, .5], [2, 2, 2.]])
    result = witness_membership(points, core, active, boxes)
    assert result['active_missed_indices'] == [2]
    assert result['admitted_noncore_indices'] == [1]
    assert result['outside'] == 1
    assert not any('free' in key for key in result)


def test_multilayer_diagnostic_is_independent_of_occupancy_rgb_error(tmp_path):
    import json
    edges = np.linspace(0, 1, 16385)
    mid = (edges[:-1]+edges[1:])*.5
    weights = np.exp(-((mid-.2)/.02)**2) + np.exp(-((mid-.8)/.02)**2)
    weights /= weights.sum()
    np.savez(tmp_path/'uniform.npz', t_start=edges[:-1], t_end=edges[1:], weights=weights)
    profile = dict(trace='uniform.npz', integrated_foreground_rgb=[.2, .3, .4], native_culled_weight_per_cascade=[0.])
    record = dict(rows=[dict(pixel={}, near=0., far=1., profiles=dict(native=profile, halo1=profile, uniform2x=profile),
               uniform_convergence=dict(opacity_abs_difference=0., foreground_rgb_max_abs_difference=0.))])
    (tmp_path/'audit.json').write_text(json.dumps(record))
    result = summarize_traces(tmp_path)
    assert result['supplemental_geometry_trigger']
    assert result['rows'][0]['native_halo_rgb_max_difference'] == 0
    assert len(result['rows'][0]['significant_peak_bins']) == 2


def test_actual_profile_never_reconstructs_threshold_from_cpu_ema(monkeypatch):
    import marshawn_tuning_occupancy
    core = np.zeros((1, 3, 3, 3), bool)
    core[0, 1, 1, 1] = True
    ema = np.zeros_like(core, dtype=float)  # CPU reconstruction would erase actual core.
    boxes = np.array([[0, 0, 0, 3, 3, 3.]])
    record = dict(radius=0, checkpoint_step=5000, binary_warmup=False)
    monkeypatch.setattr(marshawn_tuning_occupancy, 'load_snapshot', lambda *a: (core, core.copy(), ema, boxes, record))
    profiles, receipt = actual_profiles(core.copy(), ema, 0, 5000, 4096, receipt='unused', checkpoint='unused')
    assert profiles['base'].sum() == 1 and profiles['halo1'].sum() == 27
    assert receipt['cpu_reconstruction_used'] is False
