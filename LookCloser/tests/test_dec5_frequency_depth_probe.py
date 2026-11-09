from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_frequency_depth_probe import install


def test_projection_probe_changes_only_grid_depth_and_restores_renderer_on_failure():
    result=dict(depth=torch.tensor([[.5]]),rgb=torch.ones(1,3),
        packed_spacing_starts=torch.tensor([0.,.5]),packed_spacing_ends=torch.tensor([.5,1.]),
        packed_weights=torch.tensor([.5,.5]),packed_ray_indices=torch.tensor([0,0]))
    rays=SimpleNamespace(nears=torch.tensor([[0.]]),fars=torch.tensor([[1.]]))
    original=lambda rays:result
    model=SimpleNamespace(get_outputs=original)
    def update(step):
        outputs=model.get_outputs(rays)
        assert 0<float(outputs['depth'])<.1
        assert outputs['rgb'] is result['rgb']
        raise RuntimeError('interrupted grid update')
    pipe=SimpleNamespace(model=model,_update_frequency_grid=update)
    stats=install(pipe,.1)
    with pytest.raises(RuntimeError,match='interrupted'):pipe._update_frequency_grid(16384)
    assert model.get_outputs is original
    assert float(model.get_outputs(rays)['depth'])==.5
    assert stats['valid_rays']==1 and stats['additional_density_queries']==0


def test_relaxed_frequency_gate_never_changes_normal_opacity_or_rendered_rgb():
    result=dict(depth=torch.tensor([[.5]]),rgb=torch.ones(1,3),accumulation=torch.tensor([[.4]]))
    original=lambda rays:result;model=SimpleNamespace(get_outputs=original)
    def update(step):
        changed=model.get_outputs(None)
        assert float(changed['accumulation'])==1.
        assert changed['rgb'] is result['rgb'] and changed['depth'] is result['depth']
    pipe=SimpleNamespace(model=model,_update_frequency_grid=update,
        config=SimpleNamespace(frequency_projection_mode='normalized_jacobian'),dec5_frequency_sampler=True)
    stats=install(pipe,minimum_opacity=.1);pipe._update_frequency_grid(16384)
    assert model.get_outputs is original and float(model.get_outputs(None)['accumulation'])==pytest.approx(.4)
    assert stats['opaque_rays']==0 and stats['relaxed_gate_rays']==1
