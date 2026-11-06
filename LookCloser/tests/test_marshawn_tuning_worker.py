"""Durability/selection contracts independent of an expensive training run."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_marshawn_tuning import build_config, choose_checkpoint, finish, inventory_and_prune, protected_checkpoints, validate_spec


def record(step, lpips, psnr):
    return dict(step=step, masked_lpips=lpips, psnr=psnr,
                metrics=dict(lpips=lpips, psnr=psnr, ssim=.8, validation_loss=.1),
                by_actor={}, opacity_diagnostics=[dict(trigger=False)])


def test_frontier_keeps_tomorrows_tie_winner():
    old = [record(1, .1, 20), record(2, .1009, 30)]
    assert choose_checkpoint(old)['step'] == 2
    assert {1, 2}.issubset(protected_checkpoints(old, 2))
    new = old + [record(3, .0995, 10)]
    assert choose_checkpoint(new)['step'] == 1
    assert protected_checkpoints(new, 3) == {1, 3, 75940}


def test_dominated_checkpoint_cannot_win_later():
    rows = [record(1, .1005, 20), record(2, .1, 21), record(3, .09, 10)]
    assert 1 not in protected_checkpoints(rows[:2], 2)
    assert protected_checkpoints(rows, 3) == {3, 75940}


def test_real_checkpoint_inventory_sizes_and_missing_prior_ledger(tmp_path):
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    old = directory / 'step-000000001.ckpt'
    new = directory / 'step-000000002.ckpt'
    unrelated = directory / 'unrelated.bin'
    old.write_bytes(b'old state')
    new.write_bytes(b'new state is larger')
    unrelated.write_bytes(b'leave me')
    result = inventory_and_prune(tmp_path, new, 2, [record(1, .2, 20), record(2, .1, 21)])
    assert result['2']['bytes'] == len(b'new state is larger')
    assert result['1']['bytes'] == len(b'old state') and not result['1']['retained']
    assert result['2']['retained'] and new.exists() and not old.exists()
    assert unrelated.read_bytes() == b'leave me'
    assert json.loads((tmp_path / 'checkpoint_inventory.json').read_text()) == result


def test_terminal_checkpoint_can_refinalize(tmp_path):
    selected = record(106316, .1, 30)
    state = dict(evaluations=[selected], plateau=dict(stopped=True), spec_sha256='a', data_sha256='b')
    finish(tmp_path, {'seed': 42}, state, 126316, None, {})
    first = json.loads((tmp_path / 'complete.json').read_text())
    (tmp_path / 'complete.json').unlink()
    finish(tmp_path, {'seed': 42}, state, 126316, None, {})
    assert json.loads((tmp_path / 'complete.json').read_text()) == first
    assert first['selection']['masked_lpips'] == .1
    assert not first['visual_review_complete']


def test_baseline_config_does_not_silently_import_newer_recipe(tmp_path):
    spec = dict(data=str(tmp_path / 'data'), output=str(tmp_path / 'run'), objective='full',
                sampling='baseline', width=64, radius=0, coarse_step=.00625,
                distortion_final=.01, seed=42)
    validate_spec(spec)
    c = build_config(spec)
    m, p, d = c.pipeline.model, c.pipeline, c.pipeline.datamanager
    assert d.train_num_rays_per_batch == 4096
    assert (m.density_activation, m.correct_sh_directions, p.frequency_projection_mode) == ('softplus', False, 'legacy')
    assert (m.log2_hashmap_size, m.max_res, m.adaptive_warmup_steps) == (23, 8192, 4096)
    assert not m.corrected_arm_allocator and m.stable_occupancy_reduction
    assert (p.feature_reweighting_switch_step, p.feature_reweighting_after_switch) == (75941, .3)
    assert c.optimizers['fields']['scheduler'].max_steps == 200000
    assert c.load_checkpoint is None and c.mixed_precision
    assert not d.dataparser.auto_scale_poses and d.dataparser.downscale_factor == 1
    assert d.dataparser.orientation_method == d.dataparser.center_method == 'none'
    with pytest.raises(ValueError):
        validate_spec(dict(spec, radius=2))
