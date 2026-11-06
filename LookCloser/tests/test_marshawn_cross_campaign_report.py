import copy
import importlib.util
import json
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/report_marshawn_cross_campaign.py'
spec=importlib.util.spec_from_file_location('cross_report',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def metric(lpips=.1,psnr=30,step=100):
    return {'step':step,'metrics':{'lpips':lpips,'psnr':psnr,'ssim':.9},'domain':m.DOMAIN,'aggregation':m.AGGREGATION,
            'by_actor':{a:{'lpips':lpips} for a in ['black','grey']},
            'records':[{'view':v,'actor':a,'domain':m.DOMAIN} for v in ['DC','HC','KA'] for a in ['black','grey']]}


def fixture(tmp_path, cases):
    registry={'control_id':'main/control','sources':[]};records={}
    def add(key,value):
        raw=json.dumps(value);records[key]={'raw':raw,'value':copy.deepcopy(value),'sha256':m.digest(raw.encode()),'mtime':1,'read_at':2,'path':key}
    for sid,rows in cases.items():
        registry['sources'].append({'id':sid,'controller_root':str(tmp_path/sid),'protocol':m.PROTOCOL})
        jobs=[];states={}
        for name,status,lpips in rows:
            jobs.append({'name':name,'metrics_file':'{job_root}/run/complete.json','command':['python','train.py']})
            result={'selection':metric(lpips),'config':{'seed':42},'final_step':200}
            states[name]={'status':status,'host':'local','metrics':result}
            add(sid+'/'+name+'/complete',result)
            add(sid+'/'+name+'/progress',{'step':200})
            add(sid+'/'+name+'/online',[metric(lpips/2,step=150)])
        plan={'jobs':jobs,'hosts':{'local':{'root':str(tmp_path/sid)}}}
        add(sid+'/manifest',plan)
        add(sid+'/state',{'jobs':states,'manifest_sha256':m.digest(m.canonical(plan))})
    def reader(requests):return {key:copy.deepcopy(records.get(key,{'error':'missing','path':path})) for key,host,path in requests}
    return registry,records,reader


def test_all_sources_collected_and_unfinished_excluded(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'sh':[('better','complete',.08)],'abc':[('live','running',.01)]})
    report=m.assemble(r,m.collect(r,reader))
    assert report['coverage_complete']
    assert len(report['sources'])==3
    assert report['raw_metric_winner_among_verified_sources']['id']=='sh/better'
    assert report['online_provisional'][0]['curve'][0]['lpips']==.005
    assert {v['id'] for v in report['completed_ranking']}=={'main/control','sh/better'}
    assert report['reviewed_geometry_candidate'] is None
    assert report['automatic_promotion'] is False


def test_missing_source_is_not_silently_cached(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'remote':[('best','complete',.01)]})
    del records['remote/state']
    report=m.assemble(r,m.collect(r,reader))
    assert not report['coverage_complete']
    assert report['raw_metric_winner_among_verified_sources']['id']=='main/control'
    assert not report['sources'][1]['available']


def test_protocol_and_completion_source_mismatch_excluded(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'other':[('wrong','complete',.01)],'stale':[('wrong','complete',.02)]})
    r['sources'][1]['protocol']='historical_unmasked'
    records['stale/wrong/complete']['value']['final_step']=201
    report=m.assemble(r,m.collect(r,reader))
    assert len(report['completed_ranking'])==1
    assert not report['coverage_complete']
    assert len(report['errors'])==2


def test_guard_and_explicit_bound_geometry_review(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'sh':[('better','complete',.08)]})
    ref=records['sh/better/complete']
    review={'candidate_id':'sh/better','complete_sha256':ref['sha256'],'metric_step':100,'geometry_pass':True,
            'visual_review_complete':True,'reviewer':'human','reviewed_at':'2026-11-06','evidence':['render_and_ray_receipt.json']}
    records['review/0']={'value':review};r['geometry_reviews']=[{'path':'review.json'}]
    report=m.assemble(r,m.collect(r,reader))
    assert report['reviewed_geometry_candidate']['id']=='sh/better'
    review['complete_sha256']='old_checkpoint'
    assert m.assemble(r,m.collect(r,reader))['reviewed_geometry_candidate'] is None


def test_seed_confirmation_and_actor_regression_not_promoted(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'x':[('raw','complete',.08),('repeat','complete',.001)]})
    for key in ['x/state','x/raw/complete']:
        result=records[key]['value']['jobs']['raw']['metrics'] if key.endswith('/state') else records[key]['value']
        result['selection']['by_actor']['black']['lpips']=.2
    for key in ['x/state','x/repeat/complete']:
        result=records[key]['value']['jobs']['repeat']['metrics'] if key.endswith('/state') else records[key]['value']
        result['config']['seed']=43
    report=m.assemble(r,m.collect(r,reader))
    assert report['raw_metric_winner_among_verified_sources']['id']=='x/raw'
    assert report['best_actor_guard_passing']['id']=='main/control'
    assert len(report['repeat_confirmation'])==1


def test_actual_cpu_file_read_sha_and_missing(tmp_path):
    p=tmp_path/'quote newline Ж.json';p.write_text(json.dumps({'hello':'value\n"Ж'}))
    records=m.fetch({}, {'ok':str(p),'missing':str(tmp_path/'absent')})
    assert records['ok']['value']=={'hello':'value\n"Ж'}
    assert records['ok']['sha256']==m.digest(p.read_bytes())
    assert 'error' in records['missing']


def test_selector_excluded_and_tie_policy(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1),('selector','complete',.01)]})
    records['main/manifest']['value']['jobs'][1]['command']=['python','task.py','--task','{"select_only":true}']
    records['main/state']['value']['manifest_sha256']=m.digest(m.canonical(records['main/manifest']['value']))
    report=m.assemble(r,m.collect(r,reader))
    assert len(report['completed_ranking'])==1
    rows=[{'id':'a','metric_step':10,'scores':{'lpips':.1,'psnr':30}}, {'id':'b','metric_step':20,'scores':{'lpips':.1009,'psnr':31}}, {'id':'c','metric_step':5,'scores':{'lpips':.102,'psnr':50}}]
    assert m.choose(rows)['id']=='b'


def test_remote_controller_local_worker_inherits_remote_address(tmp_path):
    r,records,reader=fixture(tmp_path,{'main':[('control','complete',.1)],'remote':[('live','running',.05)]})
    r['sources'][1]['host']={'address':'ubuntu@silver-sable'}
    calls=[]
    def capture(requests):
        calls.extend(requests);return reader(requests)
    m.collect(r,capture)
    assert all(host['address']=='ubuntu@silver-sable' for key,host,path in calls if key.startswith('remote/live/'))
