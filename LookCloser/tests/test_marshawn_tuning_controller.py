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
    smi.write_text('#!/bin/sh\ncase "$1" in --query-gpu=*) echo "191, GPU-test, CPU-fixture, 1000, 0";; esac\n');smi.chmod(0o755)
    env=os.environ.copy();env['PATH']=str(bin_dir)+os.pathsep+env['PATH']
    p=plan(tmp_path);p['hosts']['local']['root']=str(tmp_path/'host');(tmp_path/'host').mkdir()
    p['hosts']['local']['gpus']=[191]  # Never contend with a live training GPU lock.
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
        launched.append(json.loads(kw['input_text']))
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


def test_large_spec_real_stdin_detach_and_dependency_file(tmp_path):
    # Larger than both Linux's per-argument and per-environment-string limit.
    text = ('quotes\" newline\n unicode Ж $() `ticks` ' * 12000)
    dependency = {'control': {'payload': text}}
    destination = tmp_path / 'result.json'
    task_script = SCRIPT.with_name('marshawn_tuning_task.py')
    code = ('import sys,json,os;sys.path.insert(0,'+repr(str(task_script.parent))+');'
            'from marshawn_tuning_task import read_dependencies;'
            'assert "TUNING_DEPENDENCIES_JSON" not in os.environ;'
            'open('+repr(str(destination))+',"w").write(json.dumps(read_dependencies()))')
    job = {'name':'large','gpu':None,'command':[sys.executable,'-c',code],
           'env':{'TUNING_DEPENDENCIES_JSON':json.dumps(dependency)},'poll_seconds':.01}
    root=tmp_path/'host';host={'python':sys.executable,'controller':str(SCRIPT),'address':None}
    args=['_start','--root',str(root),'--spec-stdin','--spec-sha256',m.digest(job)]
    first=m.host_call(host,args,input_text=json.dumps(job));assert first.returncode==0,first.stderr
    wait_for(lambda: m.read(root/'jobs/large/receipt.json',{}).get('status') in m.TERMINAL)
    assert m.read(root/'jobs/large/receipt.json')['status']=='complete'
    assert json.loads(destination.read_text())==dependency
    second=m.host_call(host,args,input_text=json.dumps(job));assert second.returncode==0
    assert len(list((root/'jobs/large').glob('launch-spec-*.json')))==1


def test_stdin_digest_rejection_before_claim(tmp_path):
    host={'python':sys.executable,'controller':str(SCRIPT),'address':None}
    result=m.host_call(host,['_start','--root',str(tmp_path),'--spec-stdin','--spec-sha256','wrong'],input_text='{"name":"never"}')
    assert result.returncode!=0 and 'digest mismatch' in result.stderr
    assert not (tmp_path/'jobs').exists()


def test_ssh_payload_never_enters_argv(monkeypatch):
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda command,**kwargs:calls.append((command,kwargs)))
    payload='x'*400000
    m.host_call({'python':'/python path','controller':'/script path','address':'host'},['_start','--root','a quoted path'],input_text=payload)
    command,kwargs=calls[0]
    assert max(map(len,command))<1024
    assert kwargs['input']==payload
    assert "'/python path'" in command[-1]


def test_dependency_file_tamper_rejected(tmp_path,monkeypatch):
    sys.path.insert(0,str(SCRIPT.parent))
    from marshawn_tuning_task import read_dependencies
    path=tmp_path/'dependencies.json';path.write_text('{}')
    monkeypatch.setenv('TUNING_DEPENDENCIES_FILE',str(path));monkeypatch.setenv('TUNING_DEPENDENCIES_SHA256','wrong')
    with pytest.raises(ValueError,match='digest mismatch'):read_dependencies()
