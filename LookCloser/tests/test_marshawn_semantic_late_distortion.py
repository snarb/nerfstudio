import copy
import json
import importlib.util
from pathlib import Path
import random
import pytest
import torch


def module():
    p = Path(__file__).resolve().parents[1]/'scripts/train_marshawn_semantic_late_distortion.py'
    s = importlib.util.spec_from_file_location('late_distortion', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def state():
    return {'pipeline': {'scalar':torch.tensor(7), 'weight':torch.arange(4.)},
            'optimizers': {'fields':{'state':{0:{'step':torch.tensor(123.),'exp_avg':torch.ones(4),'exp_avg_sq':torch.ones(4)*2}},'param_groups':[{'lr':.002,'params':[0]}]}},
            'schedulers':{'fields':{'last_epoch':123}},'scalers':{'scale':16384.},'rng_state':{'torch_cpu':torch.get_rng_state(),'python':random.getstate()},'tuning_runtime':{'sampling_state':{'step':30376}},'actual_occupancy_capture':{'core':torch.ones(1,2,2,2,dtype=torch.bool)}}


def test_exact_state_copy_and_noncontiguous_scalar_hashing():
    m = module(); expected = state(); assert len(m.verify_state_equality(expected, copy.deepcopy(expected))) == 7
    assert m.tree_digest(torch.tensor(7)) == m.tree_digest(torch.tensor(7))
    a = torch.arange(12).reshape(3,4).T
    assert m.tree_digest(a) == m.tree_digest(a.contiguous())


@pytest.mark.parametrize('key', ['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'])
def test_every_training_state_component_is_guarded(key):
    m = module(); a = state(); b = copy.deepcopy(a); b[key] = {'wrong':1}
    with pytest.raises(ValueError, match='Restored state mismatch'):
        m.verify_state_equality(a,b)


def test_empty_adam_cannot_appear_to_resume():
    m = module(); a = state(); a['optimizers']['fields']['state'] = {}
    with pytest.raises(ValueError, match='Nonempty'):
        m.verify_state_equality(a,copy.deepcopy(a))


def test_cache_order_preserved_without_consuming_python_rng():
    m = module(); before = random.getstate(); ids = random.Random(42).sample(range(63),63)
    m.verify_cache_ids(ids,42); assert random.getstate() == before
    with pytest.raises(ValueError, match='cache order'):
        m.verify_cache_ids(ids[::-1],42)


def test_actual_adam_scheduler_roundtrip_continues_moments():
    # CPU proof exercises real optimizer state rather than a mirrored fake counter.
    torch.manual_seed(12); p = torch.nn.Parameter(torch.tensor([.7,-.3])); opt = torch.optim.Adam([p],lr=.01,eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.99)
    for _ in range(3): opt.zero_grad(); p.square().sum().backward(); opt.step(); scheduler.step()
    checkpoint = copy.deepcopy((p.detach(), opt.state_dict(), scheduler.state_dict()))
    q = torch.nn.Parameter(checkpoint[0].clone()); resumed = torch.optim.Adam([q],lr=123.); resumed.load_state_dict(checkpoint[1]); sched = torch.optim.lr_scheduler.ExponentialLR(resumed,gamma=.99); sched.load_state_dict(checkpoint[2])
    for parameter, optimizer, schedule in [(p,opt,scheduler),(q,resumed,sched)]:
        optimizer.zero_grad(); parameter.square().sum().backward(); optimizer.step(); schedule.step()
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert module().tree_digest(opt.state_dict()) == module().tree_digest(resumed.state_dict())
    assert float(resumed.state[q]['step']) == 4


def test_unreleased_fork_stops_before_importing_training_code(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'train_marshawn_semantic_admission',None)
    release=tmp_path/'release.json';release.write_text('{}')
    spec={'output':str(tmp_path/'out'),'parent_checkpoint':str(tmp_path/'sources/p.ckpt'),'parent_checkpoint_sha256':'unused',
          'parent_spec':str(tmp_path/'sources/spec.json'),'parent_spec_sha256':'unused','anchor_audit':str(tmp_path/'sources/anchor.json'),'anchor_audit_sha256':'unused',
          'mode':'zero','distortion_coefficient':.01,'release':str(release),'worker_sha256':'unused','frozen_source_hashes':{},'source_render_sha256':{c+s:'bound' for c in ('DC','HC','KA')for s in ('.png','_alpha.png')}}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='source-bound fork release'):
        module().run(path)
    spec['distortion_coefficient']=.1
    with pytest.raises(ValueError,match='reviewed'):
        module().validate_spec(spec)


def test_initial_rays_must_match_but_later_adaptive_divergence_is_reported():
    p=Path(__file__).resolve().parents[1]/'scripts/run_marshawn_semantic_late_technical.py'
    s=importlib.util.spec_from_file_location('technical_pair',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    segment={'first_preupdate_ray_indices_sha256':'same-initial','ray_indices_sha256':'all-rays'}
    a={'first32':{'sampling_segment':copy.deepcopy(segment)},'final':{'sampling_segment':copy.deepcopy(segment)},'real_resume_nonempty_Adam':True};b=copy.deepcopy(a)
    b['final']['sampling_segment']['ray_indices_sha256']='different-later'
    r=m.audit_pair(a,b);assert r['initial_preupdate_rays_equal'] and r['later_divergence_requires_interpretation']
    b['first32']['sampling_segment']['first_preupdate_ray_indices_sha256']='different-initial'
    with pytest.raises(ValueError,match='Initial paired'):
        m.audit_pair(a,b)

@pytest.mark.parametrize('damage',[None,'support','learned_core','selector'])
def test_nonpersistent_support_and_actual_core_are_checked(monkeypatch,damage):
    import sys,types,hashlib
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    m=module();model=type('SemanticAdmissionModel',(),{})()
    model.semantic_admission=torch.ones(2,2,2,dtype=torch.bool)
    core=torch.zeros(1,2,2,2,dtype=torch.bool);core[0,0,0,0]=True
    model.semantic_learned_core=core.clone();model.occupancy_grid=types.SimpleNamespace(binaries=core.clone())
    model.semantic_lookup=object();model._tuning_occupancy_capture={'core':core.clone(),'active':core.clone(),'radius':0,'binary_warmup':False}
    receipt={'mapped_bool_sha256':hashlib.sha256(model.semantic_admission.numpy().tobytes()).hexdigest()}
    model.semantic_receipt=receipt.copy();anchor={'semantic_capture_metadata':{'semantic_receipt':receipt}}
    if damage=='support':model.semantic_admission[1,1,1]=False
    if damage=='learned_core':model.semantic_learned_core.zero_()
    if damage=='selector':del model.semantic_lookup
    if damage:
        with pytest.raises(ValueError):m.verify_semantic_model(model,{},anchor)
    else:assert m.verify_semantic_model(model,{},anchor)['learned_core_preserved']
