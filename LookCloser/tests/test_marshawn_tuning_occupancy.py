import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import marshawn_tuning_occupancy as audit


class Model:
    def __init__(self, radius=1):
        self.config = SimpleNamespace(occupancy_thre_clamp_mult=1., geometry_support_enabled=False,
                                      occupancy_dilation_radius=radius, occupancy_binary_warmup_steps=4096,
                                      occupancy_update_interval=16)
        core = torch.zeros((1, 5, 5, 5), dtype=torch.bool)
        core[0, 2, 2, 2] = True
        self.occupancy_grid = SimpleNamespace(binaries=core, occs=torch.zeros(125),
                                             aabbs=torch.tensor([[0., 0., 0., 5., 5., 5.]]))
        self.calls = 0

    def _postprocess_occupancy_grid(self, step=None):
        self.calls += 1
        grid = self.occupancy_grid
        if self.config.occupancy_dilation_radius:
            grid.binaries = torch.nn.functional.max_pool3d(grid.binaries.float()[:, None], 3, 1, 1)[:, 0] > 0
        if step < self.config.occupancy_binary_warmup_steps:
            grid.binaries.fill_(True)


def test_actual_core_export_never_reconstructs_ema(tmp_path):
    model = Model()
    remove = audit.install_capture_hook(model)
    state_before = torch.random.get_rng_state().clone()
    model._postprocess_occupancy_grid(4096)
    assert torch.equal(state_before, torch.random.get_rng_state())
    checkpoint = tmp_path / 'model.ckpt'
    checkpoint.write_bytes(b'checkpoint')
    receipt = audit.export_snapshot(model, checkpoint, 4100, tmp_path)
    core, active, ema, boxes, record = audit.load_snapshot(receipt, checkpoint)
    assert core.sum() == 1 and active.sum() == 27 and ema.sum() == 0
    assert record['cpu_reconstruction_used'] is False and model.calls == 1
    report = audit.audit_snapshot(receipt, tmp_path / 'review', checkpoint=checkpoint,
                                  queries={'rays': [{'origin': [-1, 2.5, 2.5], 'direction': [2, 0, 0]}]})
    assert report['all_cells_scanned'] == 125 and len(report['pages']) == 3
    assert report['counts']['skipped_unknown_geometry'] == 98
    assert len(report['rays'][0]['active'][0]['intervals']) == 5
    assert not report['complete_manual_review']
    remove()
    assert model._tuning_capture_installed is False


def test_policy_rejects_double_halo_stale_and_modified(tmp_path):
    model = Model()
    audit.install_capture_hook(model)
    with pytest.raises(ValueError, match='already installed'):
        audit.install_capture_hook(model)
    model._postprocess_occupancy_grid(4096)
    checkpoint = tmp_path / 'm'
    checkpoint.write_bytes(b'x')
    with pytest.raises(ValueError, match='stale'):
        audit.export_snapshot(model, checkpoint, 4112, tmp_path)
    model.occupancy_grid.binaries.fill_(True)
    with pytest.raises(ValueError, match='changed'):
        audit.export_snapshot(model, checkpoint, 4096, tmp_path)
    core = np.zeros((1, 7, 7, 7), bool)
    core[0, 3, 3, 3] = True
    with pytest.raises(ValueError, match='noncumulative'):
        audit.validate_policy(core, audit.dilate_one(audit.dilate_one(core)), 1)


def test_warmup_explicit_and_hash_checked(tmp_path):
    model = Model(0)
    audit.install_capture_hook(model)
    model._postprocess_occupancy_grid(0)
    checkpoint = tmp_path / 'm'
    checkpoint.write_bytes(b'x')
    receipt = audit.export_snapshot(model, checkpoint, 0, tmp_path)
    core, active, _, _, record = audit.load_snapshot(receipt, checkpoint)
    assert core.sum() == 1 and active.all() and record['binary_warmup']
    with pytest.raises(FileExistsError):
        audit.export_snapshot(model, checkpoint, 0, tmp_path)
    checkpoint.write_bytes(b'other')
    with pytest.raises(ValueError, match='Checkpoint hash'):
        audit.load_snapshot(receipt, checkpoint)
    receipt.with_suffix('.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError, match='archive hash'):
        audit.load_snapshot(receipt)


def test_subcell_scan_detects_center_missed_density():
    active = np.zeros((2, 2, 3, 4), bool)
    boxes = np.array([[0, 0, 0, 2, 3, 4], [0, 0, 0, 4, 6, 8]], float)
    def density(xyz):
        return np.where(np.linalg.norm(xyz - [.25, .25, .25], axis=1) < .01, 10., 0.)
    maximum, report = audit.scan_subcell_density(active.copy(), active, boxes, density, chunk_size=7)
    assert maximum[0, 0, 0, 0] == 10
    assert maximum[1].sum() == 0
    assert report['probes'] == active.size * 9 and not report['geometry_certificate']
    assert report['skipped_sigma_max'] == 10
    with pytest.raises(ValueError, match='nonnegative'):
        audit.scan_subcell_density(active.copy(), active, boxes, lambda xyz: np.full(len(xyz), -1))


def test_baseline_override_rejected():
    model = Model()
    model.config.geometry_support_enabled = True
    with pytest.raises(ValueError, match='baseline'):
        audit.install_capture_hook(model)
