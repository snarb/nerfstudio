import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_crash_capture import TrainingCrashCapture


class Field(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mlp_base=torch.nn.Linear(3,1);self.mlp_head=torch.nn.Linear(1,3)
    def forward(self,rays):
        raw=self.mlp_base(rays);return {'density':raw.exp(),'rgb':self.mlp_head(raw)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.field=Field();self.renderer_depth=torch.nn.Identity()
    def forward(self,rays):
        out=self.field(rays);self.renderer_depth(out['density']);return out


def test_failure_packet_state_counter_and_hook_cleanup(tmp_path):
    model=Model();optimizer=torch.optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    manager=SimpleNamespace(next_train=lambda step:(torch.ones(2,3),{'image':torch.zeros(2,3),'indices':torch.tensor([4,9])}))
    pipeline=torch.nn.Module();pipeline.model=model;pipeline.datamanager=manager
    trainer=SimpleNamespace(pipeline=pipeline,
        optimizers=SimpleNamespace(optimizers={'fields':optimizer},schedulers={'fields':scheduler}),
        grad_scaler=SimpleNamespace(state_dict=lambda:{'scale':65536.}))
    def iteration(step):
        rays,batch=manager.next_train(step);out=model(rays)
        if step==14002:raise FloatingPointError('injected nonfinite forward failure')
        return out['density'].mean()
    trainer.train_iteration=iteration;original=trainer.train_iteration
    before={k:v.clone()for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    capture=TrainingCrashCapture(trainer,tmp_path,phase_start=14001,prior_attempts=17170).install()
    trainer.train_iteration(14001)
    assert capture.packet=={} and optimizer.state=={}
    with pytest.raises(FloatingPointError,match='injected'):
        trainer.train_iteration(14002)
    path=next(tmp_path.glob('*.pt'));payload=torch.load(path,weights_only=False)
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert metadata['phase_attempts_exact']==2 and metadata['ancestry_attempts_exact_plus_declared_prior']==17172
    assert metadata['global_step']==14002 and not metadata['valid_resume_checkpoint']
    assert not metadata['eligible_for_selection'] and metadata['errors']=={}
    packet=payload['batch_rays_main_field_packet']
    # A diagnostic must retain restorable RNG structure, even though its model
    # snapshot may be mid-update and is explicitly ineligible for resuming.
    import random
    import numpy as np
    saved_rng=payload['rng_before_iteration']
    assert isinstance(saved_rng['python'],tuple)
    assert isinstance(saved_rng['numpy'],tuple)
    random.setstate(saved_rng['python'])
    np.random.set_state(saved_rng['numpy'])
    assert torch.equal(saved_rng['torch'],rng)
    assert all(k in packet for k in ['actual_train_batch_and_rays','main_ray_bundle','main_raw_mlp_base','main_raw_mlp_head','main_field_outputs','main_packed_depth_inputs'])
    assert torch.equal(packet['actual_train_batch_and_rays'][1]['indices'],torch.tensor([4,9]))
    assert torch.equal(torch.get_rng_state(),rng)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    capture.close();assert trainer.train_iteration is original
    for module in model.modules():assert not module._forward_hooks and not module._forward_pre_hooks
