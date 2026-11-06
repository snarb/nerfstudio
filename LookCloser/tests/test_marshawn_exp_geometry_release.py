import copy,json,hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_marshawn_exp_geometry_release import eligible,precedence_ready,release_values,once,validate_policy

DOMAIN='fixed GT actor crop; identical RGB0.5 exterior; AlexNet LPIPS without resize'
def fixture():
    policy=dict(expected_parent_config={'density_activation':'trunc_exp'},maximum_actor_lpips=.06,
        reference_actor_lpips={'black':.06560206164916356,'grey':.07376009970903397},maximum_actor_ratio=1.03,
        precedence_jobs=['a','b','c','d','e'])
    parent=dict(config=policy['expected_parent_config'],technical_smoke=False,final_step=126316,plateau={'stopped':True},
        selection=dict(metrics={'psnr':28.,'ssim':.8,'lpips':.05},by_actor={'black':{'lpips':.05},'grey':{'lpips':.05}},
            domain=DOMAIN,aggregation='equal actors per camera, then equal cameras',
            records=[dict(view=v,actor=a,domain=DOMAIN)for v in ('DC','HC','KA')for a in ('black','grey')]))
    return parent,policy

def test_completed_parent_thresholds_and_exact_config():
    parent,policy=fixture();assert eligible(parent,policy)
    for key,value in [('technical_smoke',True),('final_step',106315),('plateau',{'stopped':False})]:
        x=copy.deepcopy(parent);x[key]=value;assert not eligible(x,policy)
    x=copy.deepcopy(parent);x['selection']['metrics']['lpips']=.06000001;assert not eligible(x,policy)
    for actor in ('black','grey'):
        x=copy.deepcopy(parent);x['selection']['by_actor'][actor]['lpips']=policy['reference_actor_lpips'][actor]*1.031;assert not eligible(x,policy)
    x=copy.deepcopy(parent);x['config']={'density_activation':'softplus'}
    with pytest.raises(ValueError,match='drift'):eligible(x,policy)

def test_wrong_protocol_missing_pair_and_nonfinite_rejected():
    parent,policy=fixture()
    for kind in ('nonfinite','domain','pair'):
        x=copy.deepcopy(parent)
        if kind=='nonfinite':x['selection']['by_actor']['black']['lpips']=float('nan')
        elif kind=='domain':x['selection']['domain']='whole image'
        else:x['selection']['records'].pop()
        with pytest.raises(ValueError):eligible(x,policy)

def test_existing_queue_precedence_requires_all_five_assignments():
    _,policy=fixture();state={'jobs':{n:{'status':'running','host':'silver','gpu':i}for i,n in enumerate(policy['precedence_jobs'])}}
    assert precedence_ready(state,policy)
    for status in ('pending','unknown','orphaned'):
        x=copy.deepcopy(state);x['jobs']['e']['status']=status;assert not precedence_ready(x,policy)
    x=copy.deepcopy(state);x['jobs']['e']={'status':'failed'};assert precedence_ready(x,policy)
    x=copy.deepcopy(state);x['jobs']['e']={'status':'dispatching'};assert not precedence_ready(x,policy)

def test_release_binds_real_parent_and_does_not_grant_geometry_quality(tmp_path):
    for arm in ('P','PB'):(tmp_path/f'release_{arm}.pending.json').write_text(json.dumps({'approved_for_training':False,'band_weight':0 if arm=='P'else.001}))
    raw=b'{"actual":"completed parent bytes"}';policy=dict(reviewer='root',reviewed_at='now')
    releases=release_values(tmp_path,policy,raw,'policyhash')
    for value in releases.values():
        assert value['approved_for_training']is True and value['geometry_quality_approved']is False and value['model_promotion_authorized']is False
        assert value['completed_parent_receipt_sha256']==hashlib.sha256(raw).hexdigest()
    path=tmp_path/'release.json';once(path,releases['P']);once(path,releases['P'])
    with pytest.raises(ValueError,match='immutable'):once(path,releases['PB'])

def test_authorization_and_hash_drift_fail_closed(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/watch_marshawn_exp_geometry_release.py';(tmp_path/script.name).write_bytes(script.read_bytes())
    _,policy=fixture();policy.update(conditional_experiment_launch_authorized=False,model_promotion_authorized=False,reviewer='root',reviewed_at='now',reviewed_artifacts={script.name:hashlib.sha256(script.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='authorization'):validate_policy(tmp_path,policy)
    policy['conditional_experiment_launch_authorized']=True;validate_policy(tmp_path,policy)
    (tmp_path/script.name).write_text('drift')
    with pytest.raises(ValueError,match='drift'):validate_policy(tmp_path,policy)
