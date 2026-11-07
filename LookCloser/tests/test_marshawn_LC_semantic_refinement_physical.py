"""Typed constructor checks only; synthetic zero-stage states are NOT fitted results."""
import copy,json,os,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_LC_semantic_refinement_contract import construct_head,checkpoint_metadata,BASE_SHA,LABEL_SHA
from marshawn_LC_semantic_refinement_physical import physical_refinement_head
from marshawn_frozen_semantic_head import tensor_digest
from prepare_marshawn_frozen_semantic_cpu import sha
C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
BASE=C/'frozen_LC_semantic_head_archives/LA_four_UNKNOWN_20261107_v1/runs/prior001_fresh2064/head.pt'
STAGE=C/'frozen_LC_semantic_matched_refinement_plan_v1'
BOUNDS=[[-.84,-.21,-.07],[.15,.42,.63]]

@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    if not BASE.exists():pytest.skip('Actual immutable coarse checkpoint unavailable')
    assert sha(BASE)==BASE_SHA
    torch.set_num_threads(2)
    training=json.loads((STAGE/'specs/residual256_fresh2064.json').read_text())
    d=tmp_path_factory.mktemp('typed_physical_synthetic')
    tp=d/'training.json';tp.write_text(json.dumps(training))
    base=torch.load(BASE,map_location='cpu',weights_only=False)
    head=construct_head(base,training,BOUNDS)
    metadata=dict(head_type=training['head_type'],head_schema_version=1,refinement_family=training['refinement_family'],
        base_checkpoint_sha256=BASE_SHA,qualified_UNKNOWN_count=4,iteration=2064)
    # Declared2064 here is deliberately synthetic validation input, not a run receipt.
    trained=dict(**metadata,head=head.state_dict(),head_digest=tensor_digest(head),spec_sha256=sha(tp),label_qualification_sha256=LABEL_SHA)
    completion=dict(**metadata,actual_head_updates=2064)
    spec=dict(refinement_training_spec=str(tp),head_training_spec_sha256=sha(tp),expected_head_type=training['head_type'],refinement_base_checkpoint=str(BASE))
    return head,trained,completion,spec

def test_explicit_typed_roundtrip_preserves_zero_base_function(synthetic):
    head,trained,completion,spec=synthetic
    loaded=physical_refinement_head(trained,completion,spec,BOUNDS)
    xyz=torch.tensor([[.05794519186,.05855029821,.01855337620],[0.,0.,0.]])
    assert torch.equal(loaded(xyz),head(xyz))
    assert tensor_digest(loaded)==trained['head_digest']
    assert not any(p.requires_grad for p in loaded.parameters())
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('key,value',[('head_type','FrozenActorGrid'),('head_schema_version',0),('iteration',32),('base_checkpoint_sha256','other')])
def test_legacy_or_mismatched_constructor_identity_rejected(synthetic,key,value):
    _,trained,completion,spec=synthetic
    cur=dict(trained);cur[key]=value
    with pytest.raises(ValueError):physical_refinement_head(cur,completion,spec,BOUNDS)

def test_changed_frozen_base_rejected_even_with_updated_digest(synthetic):
    _,trained,completion,spec=synthetic
    cur=dict(trained);cur['head']={k:v.clone()for k,v in trained['head'].items()}
    cur['head']['coarse_logits'][0,0,0,0,0]+=1
    class State:
        def state_dict(self):return cur['head']
    cur['head_digest']=tensor_digest(State())
    with pytest.raises(ValueError,match='exact frozen learned128'):physical_refinement_head(cur,completion,spec,BOUNDS)
