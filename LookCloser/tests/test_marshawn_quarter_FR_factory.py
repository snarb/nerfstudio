import importlib.util
from pathlib import Path
import copy
import pytest
s=importlib.util.spec_from_file_location('semanticfork',Path(__file__).parents[1]/'scripts/marshawn_quarter_FR_factory.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def source():
 p=dict(m.RECIPE,semantic_admission={'identity_control':False,'manifest_sha256':'same'})
 a=dict(anchor_status='EVALUATED_INTERMEDIATE',step=75940,producer_continues=True,fullstate_hashes=dict.fromkeys(m.STATE_KEYS,'bound'),semantic_grid_checks=dict.fromkeys(['core_capture','active_capture','active_pipeline','EMA_pipeline','learned_core_pipeline'],True),semantic_capture_metadata={'radius':0,'binary_warmup':False,'step':75936,'semantic_receipt':{'manifest_sha256':'same'}},source_eval={'step':75940})
 return p,a
@pytest.mark.parametrize('key,value',[('objective','full'),('radius',1),('width',32),('sampling','baseline'),('frequency_projection_mode','normalized_jacobian'),('density_activation','softplus')])
def test_no_cross_ancestry_reuse(key,value):
 p,a=source();p[key]=value
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)
def test_evaluated_intermediate_not_fake_completed_parent():
 p,a=source();m.validate_anchor(p,a,.3);m.validate_anchor(p,a,1.);a['anchor_status']='complete'
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)
@pytest.mark.parametrize('mutation',['core','state','eval','stale','identity'])
def test_bound_semantic_evidence_required(mutation):
 p,a=source()
 if mutation=='core':a['semantic_grid_checks']['learned_core_pipeline']=False
 if mutation=='state':a['fullstate_hashes'].pop('rng_state')
 if mutation=='eval':a['source_eval']['step']=60752
 if mutation=='stale':a['semantic_capture_metadata']['step']=75800
 if mutation=='identity':p['semantic_admission']['identity_control']=True
 with pytest.raises(ValueError):m.validate_anchor(p,a,1.)

from types import SimpleNamespace as N
@pytest.mark.parametrize('step,after,applied,strength',[(75940,.3,False,1.),(75941,.3,True,.3),(75972,1.,True,1.)])
def test_python_FR_state_and_LUT_phase(step,after,applied,strength):
 field=N(feature_reweighting_strength=strength,_feature_weight_lut=N(numel=lambda:32),_feature_weight_lut_strength=strength)
 pipe=N(_feature_reweighting_switch_applied=applied,model=N(config=N(feature_reweighting_strength=strength),field=field))
 assert m.verify_FR_runtime(pipe,step,after)['field_strength']==strength
 field._feature_weight_lut_strength=.7
 with pytest.raises(ValueError):m.verify_FR_runtime(pipe,step,after)
 field._feature_weight_lut=N(numel=lambda:0)
 assert m.verify_FR_runtime(pipe,step,after)['LUT_values']==0
 pipe._feature_reweighting_switch_applied=not applied
 with pytest.raises(ValueError):m.verify_FR_runtime(pipe,step,after)
@pytest.mark.parametrize('value',[.00625,.003125])
def test_quarter_only(value):
 p,a=source();p['coarse_step']=value
 with pytest.raises(ValueError):m.validate_anchor(p,a,.3)
