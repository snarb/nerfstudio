import copy
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/guard_marshawn_tau_resources.py'
s=importlib.util.spec_from_file_location('tau_guard',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
POLICY=dict(process_cap_MiB=16384,deadline_seconds_per_invocation=900,term_grace_seconds=25,max_consecutive_errors=3,max_error_seconds=30)
E=dict(pid=123,start='456',boot='abc')
A=dict(E,state='S',argv=['python','/exact/worker.py','--spec','/exact/spec.json'])

def test_limits_include_memory_deadline_and_wrong_uuid():
    assert g.check_limits(16384,['GPU-A'],'GPU-A',900,POLICY)==[]
    assert set(g.check_limits(16385,['GPU-B'],'GPU-A',901,POLICY))=={'memory_limit','physical_GPU_mismatch','deadline'}

def test_term_only_once_then_kill_after25_and_sticky_history(tmp_path):
    state={};sent=[];send=lambda pid,sig:sent.append((pid,sig));lookup=lambda pid:dict(A)
    g.fail(state,'deadline',worker_key='123');p=tmp_path/'state.json';g.atomic(p,state);state=g.read(p)
    assert g.signal_due(state,'123',E,'/exact/worker.py','/exact/spec.json',100,POLICY,lookup,send)=='TERM'
    assert g.signal_due(state,'123',E,'/exact/worker.py','/exact/spec.json',124.999,POLICY,lookup,send)=='waiting'
    assert g.signal_due(state,'123',E,'/exact/worker.py','/exact/spec.json',125,POLICY,lookup,send)=='KILL'
    assert g.signal_due(state,'123',E,'/exact/worker.py','/exact/spec.json',130,POLICY,lookup,send)=='waiting'
    assert sent==[(123,signal.SIGTERM),(123,signal.SIGKILL)]
    g.clear_errors(state);final=g.final_record(state)
    assert not final['passed'] and final['failed'] and len(final['failures'])==1
    assert [x['kind'] for x in final['events']]==['violation','SIGTERM','SIGKILL']

@pytest.mark.parametrize('change',[{'pid':124},{'start':'457'},{'boot':'def'},{'state':'Z'},{'argv':['python','/other.py','--spec','/exact/spec.json']},{'argv':['python','/exact/worker.py','--spec','/other.json']}])
def test_reused_or_unrelated_process_never_signalled(change):
    a=dict(A,**change);sent=[]
    assert g.signal_due({},'123',E,'/exact/worker.py','/exact/spec.json',100,POLICY,lambda pid:a,lambda *x:sent.append(x))=='absent_or_identity_mismatch'
    assert not sent

def test_escalation_rechecks_identity_after_initial_lookup():
    values=iter([dict(A),dict(A,start='new')]);sent=[];state={'terminations':{'123':{'term_at':0}}}
    assert g.signal_due(state,'123',E,'/exact/worker.py','/exact/spec.json',26,POLICY,lambda pid:next(values),lambda *x:sent.append(x))=='absent_or_identity_mismatch'
    assert not sent

def test_full_bound_command_change_is_not_signalled_even_with_same_paths():
    expected=dict(E,argv=A['argv']);actual=dict(A,argv=A['argv']+['--new-option']);sent=[]
    assert g.signal_due({},'123',expected,'/exact/worker.py','/exact/spec.json',100,POLICY,lambda pid:actual,lambda *x:sent.append(x))=='absent_or_identity_mismatch'
    assert not sent

def test_errors_are_bounded_and_success_never_clears_failure():
    state={}
    g.bounded_error(state,'SSH1',0,POLICY);g.bounded_error(state,'SSH2',5,POLICY);assert not state.get('failed')
    g.bounded_error(state,'SSH3',10,POLICY);assert state['failed']
    g.clear_errors(state);assert not g.final_record(state)['passed']
    other={};g.bounded_error(other,'slow',0,POLICY);g.bounded_error(other,'slow',31,POLICY);assert other['failed']

def test_skipped_unknown_or_failed_arm_never_counts_as_phase_success():
    assert g.phase_status(['complete']*3)==(True,False)
    assert g.phase_status(['complete','running'])==(False,False)
    assert g.phase_status(['complete','skipped'])==(True,True)
    assert g.phase_status(['complete','unknown'])==(False,True)
    assert g.phase_status([])==(False,False)

def test_real_ignoring_term_child_is_killed_without_signalling_other_process(tmp_path):
    worker=tmp_path/'worker.py';ready=tmp_path/'ready';spec=tmp_path/'spec.json'
    worker.write_text("import signal,time,pathlib\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npathlib.Path("+repr(str(ready))+").write_text('ready')\ntime.sleep(30)\n")
    child=subprocess.Popen([sys.executable,str(worker),'--spec',str(spec)])
    other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists();expected=g.identity(child.pid);state={};g.fail(state,'test_resource_limit')
        assert g.signal_due(state,'own',expected,str(worker),str(spec),100,POLICY)=='TERM'
        time.sleep(.02);assert child.poll() is None
        assert g.signal_due(state,'own',expected,str(worker),str(spec),126,POLICY)=='KILL'
        child.wait(timeout=3);assert child.returncode==-signal.SIGKILL and other.poll() is None
        assert not g.final_record(state)['passed']
    finally:
        for p in [child,other]:
            if p.poll() is None:p.kill()
            p.wait(timeout=3)

def test_local_dispatch_and_remote_heartbeat_scope_are_explicit():
    text=P.read_text()
    assert "policy['local_python'],policy['controller_path'],'launch'" in text
    assert "'local_supervisor_heartbeat_expired'" in text
    assert "heartbeat.get('dispatcher_stopped',False)" in text
    assert "raise ValueError('Launch locally from C; silver self-SSH is unsupported')" in text
    assert 'os.killpg' not in text

def test_direct_local_mode_on_remote_path_rejected_before_ssh(tmp_path):
    p=dict(local_root=str(tmp_path/'local'),remote_root=str(tmp_path/'remote'),local_python=sys.executable)
    g.require_local_origin(p,tmp_path/'local/external_guard_v2/policy.json')
    with pytest.raises(ValueError,match='Launch locally'):
        g.require_local_origin(p,tmp_path/'remote/external_guard_v2/policy.json')
