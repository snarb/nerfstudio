import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
sp=importlib.util.spec_from_file_location('proposal',Path(__file__).parents[1]/'scripts/marshawn_frequency_sampling_proposal.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def fixture():
 return m.FrequencyProposal([[8,9],[8,9]],[dict(core=np.arange(5,12),band=np.arange(20,25),valid=np.arange(72))for _ in range(2)])
def original(n=2048):
 ids=torch.randint(2,(n,));y=(torch.rand(n)*8).long();x=(torch.rand(n)*9).long();return ids,y,x

def test_preserves_exact_global_draws_and_uniform_pixels():
 torch.manual_seed(9);before=torch.get_rng_state();p=fixture();ids=p.prepare([0,1],2048,before);actual,y,x=original();after=torch.get_rng_state();yy,xx,v=p.hook(actual)
 assert torch.equal(ids,actual)and torch.equal(after,torch.get_rng_state())
 uniform=p.last['labels']==2
 assert torch.equal(y[uniform],yy[uniform])and torch.equal(x[uniform],xx[uniform])
 assert p.last['quotas']==[1229,410,409]and v.all()
 for k,name in [(0,'core'),(1,'band')]:assert np.isin((yy*9+xx).numpy()[p.last['labels']==k],p.pools[0][name]).all()

def test_private_and_global_resume_reproduces_next_update():
 p=fixture();torch.manual_seed(13);p.prepare([0,1],2048,torch.get_rng_state());p.hook(original()[0]);state=p.state_dict();rng=torch.get_rng_state()
 p.prepare([0,1],2048,rng);a=p.hook(original()[0]);after=torch.get_rng_state()
 q=fixture();q.load_state_dict(state);torch.set_rng_state(rng);q.prepare([0,1],2048,rng);b=q.hook(original()[0])
 assert all(torch.equal(x,y)for x,y in zip(a,b))and torch.equal(after,torch.get_rng_state())and p.state_dict()==q.state_dict()

def test_rejects_changed_global_draws_or_camera_ids():
 p=fixture();p.prepare([0,1],2048,torch.get_rng_state());ids=original()[0];torch.rand(1)
 with pytest.raises(ValueError,match='RNG'):p.hook(ids)
 p=fixture();p.prepare([0,1],2048,torch.get_rng_state());ids=original()[0]
 with pytest.raises(ValueError,match='Camera'):p.hook(1-ids)
 with pytest.raises(ValueError,match='during'):p.state_dict()
