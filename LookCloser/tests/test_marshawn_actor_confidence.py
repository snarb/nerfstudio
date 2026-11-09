import importlib.util,itertools
from pathlib import Path
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('actor_confidence',Path(__file__).resolve().parents[1]/'scripts/marshawn_actor_confidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_continuous_monotone_profile_and_veto():
 d=np.linspace(0,64,129);w=m.outside_profile(d,64);assert w[0]==1 and w[64]==np.float32(.1)and w[-1]==0 and(np.diff(w)<=0).all();fg=np.zeros((101,101),bool);fg[40:61,40:61]=1;v=np.zeros_like(fg);v[50,50]=1;v[25:35,45:55]=1;a,r=m.native_confidence(fg,v,32);assert a[50,50]==0 and(a[v]==0).all()and a[45,45]==1 and a[0,0]==0;assert r['RGB_target_unchanged']

def test_no_gradient_outside_band_and_target_untouched():
 fg=np.zeros((25,25),bool);fg[10:15,10:15]=1;w,_=m.native_confidence(fg,np.zeros_like(fg),4);p=torch.ones((25,25,3),requires_grad=True);gt=torch.rand((25,25,3));before=gt.clone();loss=m.weighted_rgb_mse(p,gt,torch.from_numpy(w),proposal_mean_weight=float(w[w>0].mean()));loss.backward();assert torch.equal(gt,before)and(p.grad[torch.from_numpy(w==0)]==0).all()and(p.grad[torch.from_numpy(w>0)].abs().sum()>0)

def test_uniform_proposal_once_weighted_expectation_is_exact():
 w=torch.tensor([1.,.4,.1,.02]);p=torch.tensor([[1.,0,0],[0,2.,0],[0,0,3.],[1.,1.,1.]]);gt=torch.zeros_like(p);target=((p-gt).square().mean(-1)*w).sum()/w.sum();vals=[]
 for ids in itertools.combinations(range(4),2):
  ix=torch.tensor(ids);vals.append(m.weighted_rgb_mse(p[ix],gt[ix],w[ix],proposal_mean_weight=float(w.mean())))
 torch.testing.assert_close(torch.stack(vals).mean(),target,rtol=1e-6,atol=1e-7)


def test_common_proposal_same_for_hard_and_soft_policy():
 fg=np.zeros((41,41),bool);fg[15:26,15:26]=1;veto=np.zeros_like(fg);soft,_=m.native_confidence(fg,veto,8);common=soft>0;hard,r=m.native_confidence(fg,veto,0,common_proposal=common);assert r['eligible_pixels']==int(common.sum()) and r['outside_sampling_fraction_uniform_proposal']>0;assert r['proposal_mean_weight']==float(hard.sum(dtype=np.float64)/common.sum());assert(hard[common&~fg]==0).all()
