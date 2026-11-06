from pathlib import Path
import sys
from types import SimpleNamespace
import dataclasses

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_softplus_prefix_candidate import SoftplusPrefixCandidate, PrefixSamplingAdapter
from test_marshawn_softplus_prefix_candidate import geometry, BOX, CAMS, Field


class Sampler:
    def __init__(self):
        self.step = 0
    def set_step(self, step):
        self.step = step
    def state_dict(self):
        return dict(step=self.step)
    def load_state_dict(self, state):
        self.step = state['step']
    def audit_state(self):
        return self.state_dict()


def adapter():
    field = Field()
    pipe = SimpleNamespace(model=SimpleNamespace(field=field),
        get_train_loss_dict=lambda step: ({'sentinel': step}, {'rgb_loss': field.raw.square()}, {}))
    objective = SoftplusPrefixCandidate(geometry(), BOX, CAMS)
    return pipe, PrefixSamplingAdapter(pipe, Sampler(), objective, {'sha256': 'sealed'})


def test_checkpoint_roundtrip_preserves_private_rng_and_loss_composition(tmp_path):
    pipe, first = adapter()
    first.set_step(15189)
    rng = torch.random.get_rng_state().clone()
    before = first.objective.sample(15189)
    original_parameters = list(pipe.model.field.parameters())
    outputs, losses, _ = pipe.get_train_loss_dict(15189)
    assert set(losses) == {'rgb_loss', 'train_clothing_prefix_tau'}
    assert losses['rgb_loss'].item() == 4. and outputs == {'sentinel': 15189}
    checkpoint = tmp_path/'technical.ckpt'
    torch.save({'sampling_state': first.state_dict(), 'pipeline': pipe.model.field.state_dict()}, checkpoint)
    new_pipe, resumed = adapter()
    state = torch.load(checkpoint, weights_only=False)
    new_pipe.model.field.load_state_dict(state['pipeline'])
    resumed.load_state_dict(state['sampling_state'])
    after = resumed.objective.sample(resumed.step)
    assert torch.equal(before['points'], after['points'])
    assert torch.equal(before['indices'], after['indices'])
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(new_pipe.get_train_loss_dict(15189)[1]['train_clothing_prefix_tau'],
                               losses['train_clothing_prefix_tau'])
    assert list(pipe.model.field.parameters()) == original_parameters
    state['sampling_state']['weight'] = .003
    with pytest.raises(ValueError, match='resume provenance'):
        resumed.load_state_dict(state['sampling_state'])


def test_candidate_config_preserves_entire_recipe_except_parser_module(tmp_path):
    import train_marshawn_tuning as baseline
    import train_marshawn_prefix as candidate
    spec = dict(data=str(tmp_path/'data'), output=str(tmp_path/'run'), objective='full',
        sampling='body80', width=64, radius=0, coarse_step=.00625, distortion_final=.01, seed=42,
        prefix_candidate=dict(kind='reviewed_clothing_prefix_only', weight=.01))
    candidate.validate_spec(spec)
    a, b = baseline.build_config(spec), candidate.build_config(spec)
    # Parser implementations are byte-identical apart from their serialization module.
    b.pipeline.datamanager.dataparser._target = a.pipeline.datamanager.dataparser._target
    assert dataclasses.asdict(a) == dataclasses.asdict(b)
    with pytest.raises(ValueError, match='preserve full/body80'):
        candidate.validate_spec(dict(spec, objective='actor'))
