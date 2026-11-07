"""CPU-only adversarial adapter proofs; no nerfstudio/CUDA extensions needed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_lc_semantic_capture import capture_lc_native, packed_density_weights


class Rays:
    def __init__(self):
        self.origins = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.]])
        self.directions = torch.tensor([[0.,0.,1.]]).expand(3,3)

    def __len__(self):
        return len(self.origins)


class Field(torch.nn.Module):
    def forward(self, samples):
        return {'density':torch.tensor([[2.], [3.], [4.]]),
                'rgb':torch.tensor([[.1,.2,.3], [.9,.8,.7], [.3,.5,.6]])}


class Model(torch.nn.Module):
    def __init__(self, defect=None):
        super().__init__()
        self.field = Field()
        self.adaptive_sampler = torch.nn.Identity()
        self.collider = None
        self.config = SimpleNamespace(background_color='black', corrected_arm_allocator=False, max_steps_per_ray=1024)
        self.defect = defect
        self.eval().requires_grad_(False)

    def _resolved_ray_sampling_mode(self):
        return 'adaptive'

    def renderer_rgb(self, rgb, weights, ray_indices, num_rays):
        return rgb.new_zeros((num_rays,3)).index_add(0, ray_indices.long(), weights*rgb)

    def get_outputs(self, rays):
        ids = torch.tensor([0,0,2])
        starts = torch.tensor([[2.], [3.], [8.]])
        ends = starts+.25
        f = SimpleNamespace(origins=rays.origins[ids], directions=rays.directions[ids], starts=starts, ends=ends)
        f.get_positions = lambda: f.origins+f.directions*((starts+ends)*.5)
        samples = SimpleNamespace(frustums=f, deltas=ends-starts,
            spacing_starts=(starts-2)/10, spacing_ends=(ends-2)/10)
        if self.defect == 'raise':
            self.field(samples)
            raise RuntimeError('native failure')
        if self.defect == 'empty':
            ids = ids[:0]
            return dict(packed_ray_indices=ids, packed_weights=starts[:0], rgb=torch.zeros(3,3),
                accumulation=torch.zeros(3,1), num_samples_per_ray=torch.zeros(3,dtype=torch.long),
                packed_spacing_starts=starts[:0], packed_spacing_ends=ends[:0])
        field = self.field(samples)
        if self.defect == 'two_calls':
            self.field(samples)
        sigma = field['density'].reshape(-1)
        weights = packed_density_weights(starts.reshape(-1), ends.reshape(-1), sigma, ids, len(rays)).float()
        rgb = self.renderer_rgb(field['rgb'], weights[:,None], ids, len(rays))
        alpha = torch.zeros(3).index_add(0,ids,weights)[:,None]
        result = dict(packed_ray_indices=ids, packed_weights=weights[:,None], rgb=rgb,
            accumulation=alpha, packed_spacing_starts=samples.spacing_starts.clone(),
            packed_spacing_ends=samples.spacing_ends.clone(), num_samples_per_ray=torch.tensor([2,0,1]))
        if self.defect == 'weight': result['packed_weights'][1] += .02
        if self.defect == 'rgb': result['rgb'][0,0] += .02
        if self.defect == 'spacing': result['packed_spacing_starts'][0] += .02
        if self.defect == 'ray_order': result['packed_ray_indices'] = torch.tensor([0,2,2])
        if self.defect == 'physical_delta': samples.deltas[0] += .02
        return result


def capture(model):
    return capture_lc_native(model, Rays(), density_key='density', rgb_key='rgb')


def test_physical_intervals_packed_empty_ray_and_restoration():
    model = Model()
    prior = model.field.forward
    row = capture(model)
    assert 'forward' not in model.field.__dict__
    assert model.field.forward == prior
    assert row['field_forward_calls'] == 1
    assert row['starts'].tolist() == [2.,3.,8.]
    assert not torch.allclose(row['starts'], row['spacing_starts'])
    assert row['native_alpha'][1] == 0
    assert row['xyz'][-1].tolist() == [2.,0.,8.125]
    assert max(row['proof'].values()) < 1e-6


@pytest.mark.parametrize('defect', ['two_calls','weight','rgb','spacing','ray_order','physical_delta'])
def test_rejects_silent_alignment_or_render_drift(defect):
    model = Model(defect)
    with pytest.raises(ValueError): capture(model)
    assert 'forward' not in model.field.__dict__


def test_exception_restores_instance_override_exactly():
    model = Model('raise')
    override = model.field.forward
    model.field.forward = override
    with pytest.raises(RuntimeError, match='native failure'): capture(model)
    assert model.field.__dict__['forward'] is override


def test_empty_capture_never_calls_field():
    row = capture(Model('empty'))
    assert row['field_forward_calls'] == 0
    assert row['xyz'].shape == (0,3)
    assert row['native_alpha'].tolist() == [0.,0.,0.]


def test_segmented_weights_match_independent_serial_reference():
    ids = torch.tensor([0,0,0,3,3,5])
    starts = torch.tensor([1.,1.1,2.,4.,4.2,10.])
    ends = starts+torch.tensor([.1,.2,.1,.15,.3,.2])
    sigma = torch.tensor([0.,1.,100.,2.,3.,1e6])
    actual = packed_density_weights(starts,ends,sigma,ids,7)
    transmittance = torch.ones(7,dtype=torch.float64)
    expected = []
    for start,end,density,i in zip(starts,ends,sigma,ids):
        optical_depth = density.double()*(end.double()-start.double())
        expected.append(transmittance[i]*(-torch.expm1(-optical_depth)))
        transmittance[i] *= torch.exp(-optical_depth)
    assert torch.allclose(actual,torch.stack(expected),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('unfrozen', ['train','requires_grad','sampler_train'])
def test_frozen_gate_before_field(unfrozen):
    model = Model()
    if unfrozen == 'train': model.train()
    if unfrozen == 'requires_grad': model.field.register_parameter('unsafe',torch.nn.Parameter(torch.ones(1)))
    if unfrozen == 'sampler_train': model.adaptive_sampler.train()
    with pytest.raises(ValueError): capture(model)
