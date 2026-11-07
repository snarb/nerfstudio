"""Refinement stage provenance and parameterization fail-closed guards."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_LC_semantic_refinement_contract import validate_spec,FAMILY,BASE_SHA,LABEL_SHA,ARMS

def spec():
    return dict(refinement_family=FAMILY,arm='residual256',head_type=ARMS['residual256'],head_schema_version=1,
        seed=42,learning_rate=.03,base_checkpoint_sha256=BASE_SHA,label_qualification_sha256=LABEL_SHA,
        native_model_updates=0,stage_optimizer_policy='EXPLICIT_NEW_STAGE_RESET_ADAM0_AND_SAMPLER_SEED42',
        capacity_policy='KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND',qualified_UNKNOWN_count=4,
        updates=32,packet_history_reference='fixed.json')

@pytest.mark.parametrize('arm',ARMS)
def test_both_explicit_typed_arms(arm):
    s=spec();s.update(arm=arm,head_type=ARMS[arm]);assert validate_spec(s)==ARMS[arm]

@pytest.mark.parametrize('key,value',[
    ('head_type','FrozenActorGrid'),('head_schema_version',0),('base_checkpoint_sha256','other'),
    ('label_qualification_sha256','other'),('stage_optimizer_policy','restore_old_Adam2064'),
    ('qualified_UNKNOWN_count',19),('native_model_updates',1),('learning_rate',.01),
    ('packet_history_reference',None),('updates',10000)])
def test_unmatched_or_ambiguous_stage_rejected(key,value):
    s=spec();s[key]=value
    with pytest.raises(ValueError):validate_spec(s)

def test_resume_cannot_mean_new_fresh2064_or_wrong_parent():
    s=spec();s.update(resume='parent.pt',resume_initial_iteration=32);validate_spec(s)
    for key,value in [('updates',2064),('resume_initial_iteration',2064)]:
        cur=dict(s);cur[key]=value
        with pytest.raises(ValueError):validate_spec(cur)
