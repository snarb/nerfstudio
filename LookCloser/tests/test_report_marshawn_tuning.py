import importlib.util,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path
import pytest
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/report_marshawn_tuning.py'
s=importlib.util.spec_from_file_location('tuning_report',SCRIPT);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def result(lpips=.2,psnr=25,seed=42,black=.2,grey=.2):
    selection=dict(step=106316,metrics=dict(psnr=psnr,ssim=.8,lpips=lpips),by_actor=dict(black=dict(lpips=black),grey=dict(lpips=grey)))
    return dict(config=dict(seed=seed),selection=selection,technical_smoke=False)


def state_fixture():
    p={'hosts':{},'jobs':[dict(name=n,command=[]) for n in ['full_baseline_r0','good','bad','repeat_winner_seed43','structural_winner']]}
    p['jobs'][-1]['command']=['python','--task',json.dumps({'select_only':True})]
    values=[result(),result(.18,black=.19,grey=.18),result(.15,black=.3,grey=.1),result(.01,seed=43),result(.001)]
    state={'jobs':{j['name']:dict(status='complete',host='local',metrics=r) for j,r in zip(p['jobs'],values)},'updated':time.time()}
    return p,state


def test_guard_withholds_raw_best_and_excludes_repeat_selector():
    p,s=state_fixture();r,c=m.assemble(p,s)
    assert r['raw_best']['name']=='bad'
    assert r['best_guard_passing']['name']=='good'
    assert r['primary'] is None
    assert r['status']=='automatic_promotion_withheld_actor_regression'
    assert set(c)=={'bad','good','full_baseline_r0'}
    assert len(r['repeat_confirmation'])==1
    assert not r['visual_review_complete']


def test_tie_psnr_and_smoke_excluded():
    p,s=state_fixture();s['jobs']['bad']['metrics']['technical_smoke']=True
    s['jobs']['good']['metrics']=result(.1995,psnr=24)
    r,_=m.assemble(p,s)
    assert r['raw_best']['name']=='full_baseline_r0'
    assert r['primary']['name']=='full_baseline_r0'


def test_no_control_no_promotion_and_pending_unknown_oom():
    p,s=state_fixture();s['jobs']['full_baseline_r0']={'status':'running'}
    r,_=m.assemble(p,s)
    assert r['primary'] is None and r['baseline'] is None
    assert 'unknown' in m.markdown(r)
    assert 'validation_loss' not in m.markdown(r)


def package_fixture(tmp_path):
    root=tmp_path/'campaign';root.mkdir();(root/'data').mkdir();m.atomic(root/'data/box.json',{'aabb':[[0,0,0],[1,1,1]]})
    p={'hosts':{'local':{'python':sys.executable,'root':str(root)}},'jobs':[{'name':'full_baseline_r0','command':[]}]}
    out=root/'jobs/full_baseline_r0/run';out.mkdir(parents=True)
    r=result();r['config']['output']=str(out)
    folder=out/'renders/step-000106316';folder.mkdir(parents=True)
    for name in ['DC.png','HC.png','KA.png','DC_alpha.png','HC_alpha.png','KA_alpha.png']:(folder/name).write_bytes(b'synthetic-fixture-only')
    m.atomic(folder/'metrics.json',r['selection']);m.atomic(out/'complete.json',r)
    checkpoint=out/'checkpoints/step-000106316.ckpt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'checkpoint-fixture')
    m.atomic(out/'checkpoint_inventory.json',{'106316':dict(path=str(checkpoint.relative_to(out)),retained=True,sha256=m.sha(checkpoint))})
    m.atomic(out/'result.json',dict(selected_step=106316,checkpoint=str(checkpoint),render_directory=str(folder)))
    state={'jobs':{'full_baseline_r0':dict(status='complete',host='local',metrics=r)}}
    report,candidates=m.assemble(p,state)
    return root,p,report,candidates,checkpoint


def test_actual_copy_hash_zip_and_no_weights(tmp_path):
    root,p,r,c,checkpoint=package_fixture(tmp_path)
    receipt=m.package(root,p,r,c)
    assert receipt['visual_review_complete'] is False
    with zipfile.ZipFile(receipt['zip']) as z:
        assert z.testzip() is None
        assert 'full_baseline_r0/HC.png' in z.namelist()
        assert 'box.json' in z.namelist()
        assert not any(n.endswith('.ckpt') for n in z.namelist())
    assert m.package(root,p,r,c)['zip_sha256']==receipt['zip_sha256']


def test_changed_checkpoint_fails_closed(tmp_path):
    root,p,r,c,checkpoint=package_fixture(tmp_path);checkpoint.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checkpoint missing/changed'):m.package(root,p,r,c)
    assert not (root/'downloads/latest.json').exists()


def test_detached_watcher_can_restart(tmp_path):
    root=tmp_path/'campaign';root.mkdir();controller=root/'controller';controller.mkdir()
    m.atomic(controller/'manifest.json',dict(hosts={},jobs=[dict(name='pending',command=[])]))
    m.atomic(controller/'state.json',dict(jobs={'pending':dict(status='pending')},updated=time.time()))
    cmd=[sys.executable,str(SCRIPT),'launch','--campaign-root',str(root),'--interval','1']
    def wait(predicate):
        end=time.monotonic()+8
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.05)
        raise AssertionError('timeout')
    try:
        subprocess.run(cmd,check=True,capture_output=True)
        wait(lambda:(root/'report.json').exists())
        previous=m.read(root/'reporter/status.json')['process'];os.kill(previous['pid'],signal.SIGKILL)
        wait(lambda:not m.alive(previous))
        subprocess.run(cmd,check=True,capture_output=True)
        wait(lambda:m.read(root/'reporter/status.json')['process']['pid']!=previous['pid'])
        assert m.alive(m.read(root/'reporter/status.json')['process'])
    finally:
        record=m.read(root/'reporter/status.json',{}).get('process')
        if m.alive(record):os.kill(record['pid'],signal.SIGTERM)
