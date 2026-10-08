from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_luster_opacity import density_geometry_fingerprint, freeze_density_parameters


def model():
    field = torch.nn.Module()
    field.density_geometry_encoding = torch.nn.Linear(3, 4)
    field.density_geometry_mlp = torch.nn.Linear(4, 1)
    field.encoding = torch.nn.Linear(3, 4)
    field.mlp_color = torch.nn.Linear(4, 3)
    field.aabb = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    field.feature_reweighting_strength = .3
    field.enable_feature_reweighting = True
    field.density_activation = 'softplus'
    return SimpleNamespace(field=field, freq_grid=SimpleNamespace(grid=torch.zeros(2, 2, 2)))


def test_color_adaptation_keeps_density_and_optimizer_layout():
    m = model()
    optimizer = torch.optim.Adam(m.field.parameters(), lr=.1)
    parameter_ids = [id(p) for p in optimizer.param_groups[0]['params']]
    before = density_geometry_fingerprint(m)
    freeze_density_parameters(m)
    p = torch.randn(16, 3)
    rgb = m.field.mlp_color(m.field.encoding(p))
    rgb.square().mean().backward()
    assert all(p.grad is None for p in m.field.density_geometry_encoding.parameters())
    assert all(p.grad is None for p in m.field.density_geometry_mlp.parameters())
    assert all(p.grad is not None for p in m.field.encoding.parameters())
    optimizer.step()
    assert density_geometry_fingerprint(m) == before
    assert [id(p) for p in optimizer.param_groups[0]['params']] == parameter_ids


@pytest.mark.parametrize('which', ['density', 'frequency', 'reweighting', 'aabb'])
def test_audit_detects_changes_to_each_density_input(which):
    m = model()
    before = density_geometry_fingerprint(m)
    with torch.no_grad():
        if which == 'density':
            m.field.density_geometry_mlp.weight.add_(1)
        elif which == 'frequency':
            m.freq_grid.grid[0, 0, 0] = 2
        elif which == 'reweighting':
            m.field.feature_reweighting_strength = .4
        else:
            m.field.aabb[0, 0] = -2
    assert density_geometry_fingerprint(m) != before


def test_shared_density_cannot_be_frozen_as_independent_geometry():
    m = model()
    m.field.density_geometry_encoding = None
    with pytest.raises(ValueError, match='independent density'):
        freeze_density_parameters(m)
