import importlib.util
from pathlib import Path
import copy
import pytest
s=importlib.util.spec_from_file_location('semanticfork',Path(__file__).parents[1]/'scripts/marshawn_semantic_late_factory.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def source():
 p=dict(m.RECIPE,semantic_admission={'identity_control':False,'manifest_sha256':'same'})
 a=dict(anchor_status='EVALUATED_INTERMEDIATE',step=30376,producer_continues=True,fullstate_hashes=dict.fromkeys(m.STATE_KEYS,'bound'),semantic_grid_checks=dict.fromkeys(['core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline'],True),semantic_capture_metadata={'radius':0,'binary_warmup':False,'step':30368,'semantic_receipt':{'manifest_sha256':'same'}},source_eval={'step':30376})
 return p,a
@pytest.mark.parametrize('key,value',[('objective','full'),('radius',1),('width',32),('sampling','baseline'),('frequency_projection_mode','normalized_jacobian'),('density_activation','softplus')])
def test_no_cross_ancestry_reuse(key,value):
 p,a=source();p[key]=value
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)
def test_evaluated_intermediate_not_fake_completed_parent():
 p,a=source();m.validate_anchor(p,a,.01);m.validate_anchor(p,a,1.);a['anchor_status']='complete'
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)
@pytest.mark.parametrize('mutation',['core','state','eval','stale','identity'])
def test_bound_semantic_evidence_required(mutation):
 p,a=source()
 if mutation=='core':a['semantic_grid_checks']['learned_core_pipeline']=False
 if mutation=='state':a['fullstate_hashes'].pop('rng_state')
 if mutation=='eval':a['source_eval']['step']=60752
 if mutation=='stale':a['semantic_capture_metadata']['step']=30300
 if mutation=='identity':p['semantic_admission']['identity_control']=True
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)
