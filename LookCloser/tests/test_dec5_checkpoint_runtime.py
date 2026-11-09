from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_checkpoint_runtime import restore


def model():
    return SimpleNamespace(config=SimpleNamespace(feature_reweighting_strength=1.),
                           field=SimpleNamespace(feature_reweighting_strength=1.))


def test_late_sequence_reload_restores_fr03_without_changing_early_or_legacy_checkpoints():
    m=model();restore(m,dict(step=75940),dict(fr_switch_after=75940));assert m.field.feature_reweighting_strength==1.
    restore(m,dict(step=75941),dict(fr_switch_after=75940));assert m.field.feature_reweighting_strength==.3
    m.config.feature_reweighting_strength=m.field.feature_reweighting_strength=.7
    assert restore(m,dict(step=90000),{}) is None
    assert m.field.feature_reweighting_strength==.7


def test_explicit_runtime_phase_mismatch_is_rejected_instead_of_silent_wrong_render():
    with pytest.raises(ValueError,match='disagrees'):
        restore(model(),dict(step=90000,runtime=dict(feature_reweighting_strength=1.)),dict(fr_switch_after=75940))


def test_public_pipeline_reload_restores_late_and_early_phase_from_real_tensor_state():
    import torch
    from torch import nn
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    class FieldModel(nn.Module):
        def __init__(self):
            super().__init__();self.device_indicator_param=nn.Parameter(torch.empty(0))
            self.config=SimpleNamespace(feature_reweighting_strength=1.)
            self.field=SimpleNamespace(feature_reweighting_strength=1.)
        @property
        def device(self):return self.device_indicator_param.device
        def update_to_step(self,step):self.step=step
    pipe=LookCloserPipeline.__new__(LookCloserPipeline);nn.Module.__init__(pipe)
    pipe._model=FieldModel();pipe.world_size=1;pipe.dynamic_num_rays_per_batch=4096
    pipe.config=SimpleNamespace(feature_reweighting_switch_step=75941,feature_reweighting_after_switch=.3,target_num_samples_per_batch=0)
    pipe._initial_feature_reweighting_strength=1.
    pipe.datamanager=SimpleNamespace(train_pixel_sampler=SimpleNamespace(sample_count=0))
    pipe._sync_tcnn_network_jit_to_step=lambda step:None
    for name,value in dict(cumulative_point_samples=torch.tensor(0),dynamic_samples_per_ray_ema=torch.tensor(0.),
                           dynamic_rays_per_batch_state=torch.tensor(4096),fas_sample_count_state=torch.tensor(0)).items():
        pipe.register_buffer(name,value)
    state=pipe.state_dict()
    pipe.load_pipeline(state,90000)
    assert pipe.model.step==90000 and pipe.model.field.feature_reweighting_strength==.3
    pipe.load_pipeline(state,15000)
    assert pipe.model.field.feature_reweighting_strength==1. and not pipe._feature_reweighting_switch_applied
