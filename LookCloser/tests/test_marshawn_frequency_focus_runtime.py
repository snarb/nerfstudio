import sys,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from marshawn_frequency_focus_runtime import FocusRuntime

def make():
 class Pipe:
  config=SimpleNamespace(grid_update_batch_size=2048,grid_update_interval=1024,frequency_projection_mode='legacy',independent_rng_streams=False,datamanager=SimpleNamespace(cpu_fas_prefetch=False))
  cached_freq_maps={0:None,1:None}
  def _update_frequency_grid(self,step):
   ids=torch.randint(2,(2048,));torch.rand(2048);torch.rand(2048);self.last=self.dec5_frequency_sampler(ids)
 pipe=Pipe();rows=[dict(core=np.ones((8,9),bool),pools=dict(core=np.arange(12),band=np.arange(20,30),valid=np.arange(72)))for _ in range(2)]
 adapter=SimpleNamespace(rows=rows,source_records=[dict(camera='AA',mask_sha256='a'),dict(camera='AB',mask_sha256='b')])
 cfg=dict(budget=2048,core_count=1229,band_count=410,uniform_count=409,private_seed=4201024)
 return pipe,adapter,cfg

def test_checkpoint_restores_nonempty_private_stream_and_next_actual_hook():
 pipe,a,c=make();r=FocusRuntime(pipe,a,c);torch.manual_seed(12);pipe._update_frequency_grid(1024);state=copy.deepcopy(r.state_dict());rng=torch.get_rng_state()
 pipe._update_frequency_grid(2048);expected=pipe.last;expectedstate=r.state_dict();globalend=torch.get_rng_state()
 pipe2,a2,c2=make();r2=FocusRuntime(pipe2,a2,c2);r2.load_state_dict(state);torch.set_rng_state(rng);pipe2._update_frequency_grid(2048)
 assert expectedstate==r2.state_dict()and torch.equal(globalend,torch.get_rng_state())
 assert all(torch.equal(x,y)for x,y in zip(expected,pipe2.last))

def test_policy_or_source_change_rejected_on_resume():
 pipe,a,c=make();r=FocusRuntime(pipe,a,c);s=r.state_dict();s['source_sha256']='changed'
 with pytest.raises(ValueError,match='source/policy'):r.load_state_dict(s)
 pipe,a,c=make();pipe.config=copy.deepcopy(pipe.config);pipe.config.frequency_projection_mode='normalized_jacobian'
 with pytest.raises(ValueError,match='policy'):FocusRuntime(pipe,a,c)
