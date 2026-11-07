import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

S=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(S))
import marshawn_quarter_clear_tau as q
import marshawn_quarter_tau_factory as f
from marshawn_expanded_clear_tau import QualifiedClearTau as Previous

def targets():return {'targets':[dict(origin=[0.,0.,.5],direction=[0.,0.,-1.],near=.1,far=.3) for _ in range(10481)]}
def create(pop,coef=.001):return q.QualifiedClearTau(SimpleNamespace(device=torch.device('cpu')),targets(),'a'*64,coef,pop)

def test_packets_same_jitter_cursors_and_sealed_old_randint_64_attempts():
    a=create('old7369');b=create('seven10481');c=create('seven10481',0.)
    old=Previous(SimpleNamespace(device=torch.device('cpu')),{'targets':targets()['targets'][:7369]},'a'*64,.001)
    global_before=torch.get_rng_state().clone()
    for _ in range(64):
        ai,aj,ah=a.draw();bi,bj,bh=b.draw();ci,cj,ch=c.draw();oi,oj,oh=old.draw()
        assert torch.equal(ai,oi) and torch.equal(aj,oj) and ah==oh
        assert torch.equal(bi,ci) and bh==ch
        assert torch.equal(aj,bj) and torch.equal(bj,cj)
        assert a.packet_state_before==b.packet_state_before==c.packet_state_before
        assert a.packet_state_after==b.packet_state_after==c.packet_state_after
    assert a.jitter_chain==b.jitter_chain==c.jitter_chain
    assert a.ids_chain!=b.ids_chain and b.ids_chain==c.ids_chain
    assert torch.equal(global_before,torch.get_rng_state())

@pytest.mark.parametrize('population',['old7369','seven10481'])
def test_private_roundtrip_and_negative_population(population):
    a=create(population)
    for _ in range(32):a.draw()
    state=copy.deepcopy(a.state_dict());expected=a.draw();b=create(population);b.load_state_dict(state)
    actual=b.draw();assert actual[2]==expected[2] and a.jitter_chain==b.jitter_chain and a.ids_chain==b.ids_chain
    bad=copy.deepcopy(state);bad['population']=10481 if population=='old7369' else 7369
    with pytest.raises(ValueError):b.load_state_dict(bad)

def anchor():
    p=dict(f.RECIPE,semantic_admission=dict(identity_control=False,manifest_sha256='a'*64))
    a=dict(anchor_status='EVALUATED_INTERMEDIATE_QUARTER',producer_continues=True,step=91128,
           fullstate_hashes={k:'b'*64 for k in f.STATE_KEYS},semantic_grid_checks={k:True for k in ['core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline']},
           semantic_capture_metadata=dict(step=91120,radius=0,binary_warmup=False,semantic_receipt=dict(manifest_sha256='a'*64)),source_eval=dict(step=91128))
    return p,a

def test_intermediate_is_not_falsely_completed_and_wrong_recipes_rejected():
    p,a=anchor();assert f.validate_anchor(p,a,.001) is p
    for key,value in [('coarse_step',.00625),('distortion_final',.1),('radius',1),('width',32),('frequency_projection_mode','normalized_jacobian')]:
        wrong=copy.deepcopy(p);wrong[key]=value
        with pytest.raises(ValueError):f.validate_anchor(wrong,a,.001)
    for key,value in [('step',111316),('anchor_status','COMPLETED_SELECTED_R0'),('producer_continues',None)]:
        wrong=copy.deepcopy(a);wrong[key]=value
        with pytest.raises(ValueError):f.validate_anchor(p,wrong,.001)

def test_worker_keeps_original_train_iteration_frequency_boundary_and_leases():
    text=(S/'train_marshawn_quarter_clear_tau.py').read_text();tree=ast.parse(text)
    assert len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='train_iteration'])==1
    assert 'result = native_frequency_update(step)' in text and "e['step'] == 91136" in text
    assert 91136%1024==0 and 91136-91128==8
    assert '(500,1000,2000,4000,8000,12000)' in text
    assert text.index('require_controller_lease(os.environ')<text.index('import torch\n    isolation')<text.index('loaded = torch.load')
    assert 'grad_scaler.update(' not in text and "aux.load_state_dict(loaded['fork_runtime']['auxiliary_state'])" in text

def test_bad_target_hash_rejects_before_source_reads(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{}')
    with pytest.raises(ValueError):q.validate_targets(p,'0'*64,tmp_path,tmp_path/'absent','seven10481')
