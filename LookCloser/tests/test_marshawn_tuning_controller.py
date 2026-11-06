"""Real CPU subprocess tests: GPUs are simulated, no training is launched."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/marshawn_tuning_controller.py'
spec=importlib.util.spec_from_file_location('tuning_controller',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def wait_for(predicate, seconds=15):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting for subprocess')


def plan(tmp_path):
    return {'schema_version':1,'poll_seconds':.05,'hosts':{'local':{'address':None,'python':sys.executable,
        'controller':str(SCRIPT),'root':str(tmp_path/'host'),'gpus':[0]}},'jobs':[{'name':'first','command':[sys.executable,'-c','pass']}]}


def test_manifest_cycle_and_freeze(tmp_path):
    p=plan(tmp_path);m.validate(p)
    p['jobs'][0]['dependencies']=['first']
    with pytest.raises(ValueError,match='Cyclic'):m.validate(p)
    p['jobs'][0]['dependencies']=[];source=tmp_path/'input.json';m.atomic(source,p)
    m.prepare(tmp_path/'campaign',source)
    p['jobs'][0]['command'][-1]='print(1)';m.atomic(source,p)
    with pytest.raises(ValueError,match='immutable'):m.prepare(tmp_path/'campaign',source)


def test_condition_and_missing_dependency():
    states={'control':{'metrics':{'lpips':.2}}}
    assert m.condition_pass({'job':'control','path':'metrics.lpips','op':'lt','value':.3},states)
    assert not m.condition_pass({'job':'control','path':'metrics.lpips','op':'lt','value':.1},states)
    with pytest.raises(KeyError):m.condition_pass({'job':'control','path':'metrics.missing','op':'lt','value':.1},states)


def test_transport_failure_never_redispatches(tmp_path,monkeypatch):
    p=plan(tmp_path);state={'jobs':{'first':{'status':'running','host':'local','gpu':0,'spec_sha256':'x'}}}
    calls=[]
    def fail(host,args,**kw):calls.append(args);raise subprocess.TimeoutExpired('ssh',1)
    monkeypatch.setattr(m,'host_call',fail)
    assert not m.tick(p,state,tmp_path)
    assert state['jobs']['first']['status']=='unknown'
    assert all(c[0]=='_probe' for c in calls)


def test_dead_worker_receipt_never_restarts(tmp_path):
    root=tmp_path/'host';counter=tmp_path/'counter'
    j={'name':'j','gpu':None,'command':[sys.executable,'-c',f'open({str(counter)!r},"a").write("x")'],'poll_seconds':.01}
    directory=root/'jobs'/'j';directory.mkdir(parents=True)
    m.atomic(directory/'receipt.json',{'status':'starting','supervisor':None})
    assert m.worker(root,j)==0
    assert not counter.exists()


def test_real_dispatcher_crash_resume_no_duplicate(tmp_path):
    # Fake driver only for CPU supervision test. No actual GPU is touched.
    bin_dir=tmp_path/'bin';bin_dir.mkdir();smi=bin_dir/'nvidia-smi'
    smi.write_text('#!/bin/sh\ncase "$1" in --query-gpu=*) echo "0, GPU-test, CPU-fixture, 1000, 0";; esac\n');smi.chmod(0o755)
    env=os.environ.copy();env['PATH']=str(bin_dir)+os.pathsep+env['PATH']
    p=plan(tmp_path);p['hosts']['local']['root']=str(tmp_path/'host');(tmp_path/'host').mkdir()
    counter=tmp_path/'counter'
    p['jobs']=[{'name':'first','command':[sys.executable,'-c',f'import time;open({str(counter)!r},"a").write("first\\n");time.sleep(2)']},
        {'name':'second','dependencies':['first'],'command':[sys.executable,'-c',f'open({str(counter)!r},"a").write("second\\n")']}]
    source=tmp_path/'plan.json';m.atomic(source,p);root=tmp_path/'campaign';m.prepare(root,source)
    def launch():return subprocess.run([sys.executable,str(SCRIPT),'resume','--root',str(root)],env=env,capture_output=True,text=True,check=True)
    launch()
    supervisor=None
    try:
        wait_for(lambda:counter.exists())
        controller=m.read(root/'controller.json')['process'];os.kill(controller['pid'],signal.SIGKILL)
        wait_for(lambda:not m.alive(controller))
        launch()
        wait_for(lambda:m.read(root/'state.json')['jobs']['second']['status']=='complete',20)
        assert counter.read_text().splitlines()==['first','second']
        assert m.read(root/'state.json')['jobs']['first']['status']=='complete'
        wait_for(lambda:not m.alive(m.read(root/'controller.json')['process']))
    finally:
        current=m.read(root/'controller.json',{}).get('process')
        if m.alive(current):os.kill(current['pid'],signal.SIGTERM)
        for receipt in (tmp_path/'host'/'jobs').glob('*/receipt.json'):
            item=m.read(receipt)
            for key in ['worker','supervisor']:
                record=item.get(key)
                if m.alive(record):os.kill(record['pid'],signal.SIGTERM)


def test_skipped_barrier_and_dependency_metrics(tmp_path, monkeypatch):
    p=plan(tmp_path)
    p['jobs']=[{'name':'optional','command':['true']},
               {'name':'control','command':['true']},
               {'name':'next','dependencies':['optional','control'],'command':['true']}]
    state={'jobs':{'optional':{'status':'skipped'},'control':{'status':'complete','metrics':{'lpips':.2}},'next':{'status':'pending'}}}
    launched=[]
    def remote(host,args,**kw):
        if args[0]=='_probe':
            data={'jobs':{},'gpus':{'0':{'uuid':'fake','processes':[]}},'gpu_error':None,
                  'controller_sha256':m.hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}
            return subprocess.CompletedProcess([],0,json.dumps(data),'')
        launched.append(json.loads(m.base64.b64decode(args[-1])))
        return subprocess.CompletedProcess([],0,'','')
    monkeypatch.setattr(m,'host_call',remote)
    m.tick(p,state,tmp_path)
    assert len(launched)==1
    assert json.loads(launched[0]['env']['TUNING_DEPENDENCIES_JSON'])=={'optional':None,'control':{'lpips':.2}}


def test_missing_receipt_after_ambiguous_launch_keeps_host_affinity(tmp_path,monkeypatch):
    p=plan(tmp_path);p['hosts']['remote']=dict(p['hosts']['local'],root=str(tmp_path/'remote'))
    state={'jobs':{'first':{'status':'unknown','host':'remote','gpu':0,'spec_sha256':'old'}}}
    started=[]
    def call(host,args,**kwargs):
        if args[0]=='_probe':return subprocess.CompletedProcess([],0,json.dumps({'jobs':{},'gpus':{'0':{'processes':[]}},'gpu_error':None,'controller_sha256':m.hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}),'')
        started.append(host['root']);return subprocess.CompletedProcess([],0,'','')
    monkeypatch.setattr(m,'host_call',call)
    m.tick(p,state,tmp_path)
    assert started==[str(tmp_path/'remote')]
    assert state['jobs']['first']['host']=='remote'
