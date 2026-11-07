"""CPU negative/identity gates; these do not certify actual CUDA traversal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

S=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(S))
from marshawn_semantic_grid_replay import temporary_binaries,grid_membership,adversarial_positions,missed_reference_mass,numeric_rgba_parity,packed_selected
from run_marshawn_semantic_grid_replay import validate


def digest(state):
    return tuple((k,str(v.dtype),tuple(v.shape),hashlib.sha256(v.numpy().tobytes()).hexdigest())for k,v in sorted(state.items()))


class Pipe:
    def __init__(self):
        self.model=SimpleNamespace(training=False,config=SimpleNamespace(occupancy_eval_dilation_radius=0),_eval_occupancy_backup=None,
            occupancy_grid=SimpleNamespace(binaries=torch.ones((1,2,2,2),dtype=torch.bool)),semantic_admission=torch.ones((1,2,2,2),dtype=torch.bool))
        self.model.adaptive_sampler=SimpleNamespace(occupancy_grid=self.model.occupancy_grid)
        self.ema=torch.arange(8,dtype=torch.float32);self.core=torch.ones((1,2,2,2),dtype=torch.bool);self.assertions=0
        self.original=digest(self.state_dict())
    def state_dict(self):return{'_model.occupancy_grid.binaries':self.model.occupancy_grid.binaries,'_model.adaptive_sampler.occupancy_grid.binaries':self.model.adaptive_sampler.occupancy_grid.binaries,'EMA':self.ema,'core':self.core}
    def source_assert(self):
        self.assertions+=1
        assert bool(self.model.occupancy_grid.binaries.all()),'Original grid must be restored BEFORE source assertion'
        assert digest(self.state_dict())==self.original


def test_binary_only_and_restoration():
    p=Pipe();mask=torch.zeros_like(p.core);mask[0,0,1,0]=True;original=p.model.occupancy_grid.binaries
    with temporary_binaries(p,mask,digest,p.source_assert)as check:
        assert torch.equal(p.model.occupancy_grid.binaries,mask);check()
        assert torch.equal(p.core,torch.ones_like(p.core))
    assert p.model.occupancy_grid.binaries is original;p.source_assert();assert p.assertions==3


def test_exception_restores_before_assert():
    p=Pipe()
    with pytest.raises(RuntimeError,match='render failure'):
        with temporary_binaries(p,torch.zeros_like(p.core),digest,p.source_assert):raise RuntimeError('render failure')
    p.source_assert()


def test_nonbinary_mutation_detected_not_silently_repaired():
    p=Pipe()
    with pytest.raises(AssertionError):
        with temporary_binaries(p,torch.zeros_like(p.core),digest,p.source_assert)as check:
            p.ema[0]=99
            with pytest.raises(ValueError,match='Non-binary'):check()
    assert p.model.occupancy_grid.binaries.all();assert p.ema[0]==99


@pytest.mark.parametrize('bad',['radius','backup','training','admission','dtype','shape'])
def test_reject_implicit_grid_or_bad_masks(bad):
    p=Pipe();mask=torch.zeros_like(p.core)
    if bad=='radius':p.model.config.occupancy_eval_dilation_radius=1
    if bad=='backup':p.model._eval_occupancy_backup=p.core
    if bad=='training':p.model.training=True
    if bad=='admission':p.model.semantic_admission.zero_();mask.fill_(True)
    if bad=='dtype':mask=mask.float()
    if bad=='shape':mask=mask[:,0]
    with pytest.raises(ValueError):
        with temporary_binaries(p,mask,digest,p.source_assert):pass


def test_actual_cell_axes_endpoints_outside():
    g=torch.zeros((1,2,2,2),dtype=torch.bool);g[0,1,0,1]=True
    points=torch.tensor([[.75,.25,.75],[.25,.75,.75],[1.,0.,1.],[1.01,0.,1.]])
    assert grid_membership(points,g,torch.tensor([[0.,0.,0.,1.,1.,1.]])).tolist()==[True,False,True,False]


def test_adversarial_private_rng_and_109_prefix():
    np.random.seed(123);before=np.random.get_state();w=np.zeros((109,3),np.float32)
    a,proof=adversarial_positions([[0,0,0],[1,1,1]],[[0,0,0,1,1,1]],w)
    after=np.random.get_state();assert all(np.array_equal(x,y)for x,y in zip(before,after));assert np.array_equal(a[:109],w)
    b,other=adversarial_positions([[0,0,0],[1,1,1]],[[0,0,0,1,1,1]],w);assert np.array_equal(a,b);assert proof==other


def test_missing_selected_mass_gap_and_tail_not_raw_mass():
    dense={'starts':np.array([0.,1.,2.,3.]),'ends':np.array([1.,2.,3.,4.]),'selected_weight':np.array([.1,.2,.3,.1])}
    native={'starts':np.array([0.,2.]),'ends':np.array([1.,3.])}
    r=missed_reference_mass(dense,native,{'candidate':np.array([1,0,1,0],bool)})
    assert r['native_interval_uncovered_selected_mass']==pytest.approx(.3)
    assert r['selected_mass_beyond_native_last_end']==pytest.approx(.1)
    assert r['rejected_grid_selected_mass']['candidate']==pytest.approx(.3)
    assert not r['physical_free_space_claim']


def test_rgba_identity_and_reject_visible_drift():
    a=np.zeros((20,20,4),np.float32);assert numeric_rgba_parity(a,a)['float_max_error']==0
    b=a.copy();b[0,0,0]=.1
    with pytest.raises(ValueError):numeric_rgba_parity(b,a)


class FakeNerfacc:
    @staticmethod
    def pack_info(ids,n):return(ids,n)
    @staticmethod
    def render_weight_from_density(t_starts,t_ends,sigmas,packed_info):
        tau=(t_ends-t_starts)*sigmas;prefix=torch.cat((torch.zeros(1),torch.cumsum(tau[:-1],0)));return(torch.exp(-prefix)*(-torch.expm1(-tau)),)
    @staticmethod
    def accumulate_along_rays(weights,values,ray_indices,n_rays):
        if values is None:return weights.sum().reshape(1,1)
        return(weights[:,None]*values).sum(0,keepdim=True)


def test_selected_density_recomposes_revealed_background():
    row=dict(starts=torch.tensor([0.,1.]),ends=torch.tensor([1.,2.]),sigma=torch.tensor([100.,1.]),rgb=torch.tensor([[1.,0.,0.],[0.,1.,0.]]),ray_indices=torch.zeros(2,dtype=torch.long),num_samples_per_ray=torch.tensor([2]))
    stable=lambda s,e,sigma,ids,n:FakeNerfacc.render_weight_from_density(s,e,sigma,(ids,n))[0].double()
    r=packed_selected(row,torch.tensor([0.,1.]),'hard_theta01',lambda p:{'hard_theta01':(p>=.1).float()},FakeNerfacc,stable)
    assert r['selected_RGBA'][0,1]> .63;assert r['selected_RGBA'][0,0]==0;assert row['sigma'][0]==100


def base_plan():
    return dict(kind='FROZEN_LC_P01_THETA01_GRID_R0_R1_V1',selector='hard_theta01',phases=['source_grid','semantic_r0','semantic_r1'],
        source_checkpoint_sha256='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641',head_sha256='bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc',native_cap=1024,chunk=128,source_label_regime='two_reviewed_UNKNOWN_40470_40622',uniform_samples=[8192,16384],bound_files={})


@pytest.mark.parametrize('key,value',[('head_sha256','wrong'),('source_checkpoint_sha256','wrong'),('selector','hard_theta05'),('source_label_regime','4UNKNOWN'),('native_cap',2048),('uniform_samples',[4096,8192])])
def test_wrong_source_or_math_reject_before_cuda(tmp_path,key,value):
    p=base_plan();p[key]=value;path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):validate(path,False)


def test_false_release_cannot_launch(tmp_path):
    p=base_plan();p['root_release']=str(tmp_path/'release.json');path=tmp_path/'plan.json';path.write_text(json.dumps(p));Path(p['root_release']).write_text(json.dumps(dict(reviewer='root',readonly_grid_comparison_authorized=False,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    with pytest.raises(ValueError,match='release'):validate(path,True)


def test_independent_grid_alias_storage_rejected():
    p=Pipe();p.model.adaptive_sampler.occupancy_grid=SimpleNamespace(binaries=p.core.clone())
    with pytest.raises(ValueError,match='same registered grid object'):
        with temporary_binaries(p,torch.zeros_like(p.core),digest,p.source_assert):pass


def test_cloned_state_alias_rejected():
    p=Pipe();original_state=p.state_dict
    def cloned():
        state=original_state();state['_model.adaptive_sampler.occupancy_grid.binaries']=state['_model.adaptive_sampler.occupancy_grid.binaries'].clone();return state
    p.state_dict=cloned
    with pytest.raises(ValueError,match='same source storage|exact source storage'):
        with temporary_binaries(p,torch.zeros_like(p.core),digest,p.source_assert):pass


def test_unrelated_grid_like_tensor_remains_protected():
    p=Pipe();unrelated=torch.ones_like(p.core);original_state=p.state_dict
    def extra():return dict(original_state(),unrelated_grid_like_tensor=unrelated)
    p.state_dict=extra;p.original=digest(p.state_dict())
    with pytest.raises(AssertionError):
        with temporary_binaries(p,torch.zeros_like(p.core),digest,p.source_assert) as check:
            unrelated.zero_()
            with pytest.raises(ValueError,match='Non-binary'):check()
    assert p.model.occupancy_grid.binaries.all()
