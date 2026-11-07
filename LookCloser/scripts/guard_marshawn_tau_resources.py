"""External v2 CPU guardian. Local dispatcher; independent remote worker watchdog.

No scientific snapshot, worker specification, model or optimizer is modified.
Failure is sticky, signals require exact PID/start/boot/argv identity, and a lost
local supervisor expires the remote heartbeat rather than leaving work unguarded.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        st=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(pid=int(pid),start=st[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    state=st[0],argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None
def same(expected,actual):
    return bool(actual and actual.get('state')!='Z' and all(str(expected[k])==str(actual[k]) for k in ('pid','start','boot')))
def authorized_worker(record,actual,worker_path,spec_path):
    return (same(record,actual) and worker_path in actual['argv'] and spec_path in actual['argv']
            and ('argv' not in record or record['argv']==actual['argv']))
def event(state,kind,**fields):
    state.setdefault('events',[]).append(dict(time=time.time(),kind=kind,**fields))
def fail(state,reason,**fields):
    state['failed']=True
    token=json.dumps(dict(reason=reason,**fields),sort_keys=True)
    if token not in state.setdefault('failure_keys',[]):
        state['failure_keys'].append(token);event(state,'violation',reason=reason,**fields)
def check_limits(memory,physical_uuids,expected_uuid,age,policy):
    reasons=[]
    if memory>policy['process_cap_MiB']:reasons.append('memory_limit')
    if any(u!=expected_uuid for u in physical_uuids):reasons.append('physical_GPU_mismatch')
    if age>policy['deadline_seconds_per_invocation']:reasons.append('deadline')
    return reasons
def bounded_error(state,error,now,policy):
    state['consecutive_errors']=state.get('consecutive_errors',0)+1
    state.setdefault('first_error_time',now)
    event(state,'supervision_error',error=str(error),consecutive=state['consecutive_errors'])
    if state['consecutive_errors']>=policy['max_consecutive_errors'] or now-state['first_error_time']>=policy['max_error_seconds']:
        fail(state,'bounded_supervision_error')
def clear_errors(state):
    state['consecutive_errors']=0;state.pop('first_error_time',None)
def phase_status(statuses):
    terminal=bool(statuses) and all(s in ('complete','failed','blocked','skipped') for s in statuses)
    failed=any(s in ('failed','blocked','unknown','skipped') for s in statuses)
    return terminal,failed
def signal_due(state,key,expected,worker_path,spec_path,now,policy,lookup=identity,send=os.kill,persist=None):
    """Signal the exact worker PID only; PID reuse or argv mismatch is never killed."""
    actual=lookup(expected['pid'])
    if not authorized_worker(expected,actual,worker_path,spec_path):return 'absent_or_identity_mismatch'
    record=state.setdefault('terminations',{}).setdefault(key,{})
    if 'term_at' not in record:
        # Persist intent in caller state before next iteration; every action also enters audit events.
        record['term_at']=now;event(state,'SIGTERM',worker=expected,intent_recorded=True)
        if persist:persist(state)
        send(expected['pid'],signal.SIGTERM)
        return 'TERM'
    if 'kill_at' not in record and now-record['term_at']>=policy['term_grace_seconds']:
        # Re-read immediately before escalation.
        if not authorized_worker(expected,lookup(expected['pid']),worker_path,spec_path):return 'absent_or_identity_mismatch'
        record['kill_at']=now;event(state,'SIGKILL',worker=expected,intent_recorded=True)
        if persist:persist(state)
        send(expected['pid'],signal.SIGKILL)
        return 'KILL'
    return 'waiting'
def final_record(state):
    return dict(instance=state.get('instance'),phase=state.get('phase'),passed=not state.get('failed',False),failed=state.get('failed',False),
                events=state.get('events',[]),terminations=state.get('terminations',{}),
                failures=state.get('failure_keys',[]),finished=time.time())


def verify_policy(path):
    policy=read(path)
    if policy['version']!=2 or policy['term_grace_seconds']!=25 or policy['heartbeat_timeout_seconds']>90:
        raise ValueError('Unexpected guardian safety policy')
    if sha(__file__)!=policy['guardian_sha256']:raise ValueError('Guardian source changed')
    return policy

def require_local_origin(policy,path):
    expected=Path(policy['local_root'])/'external_guard_v2/policy.json'
    if Path(path).resolve()!=expected.resolve() or str(Path(path).resolve()).startswith(policy['remote_root']):
        raise ValueError('Launch locally from C; silver self-SSH is unsupported')
    if not Path(policy['local_python']).exists():raise ValueError('Local C/python is unavailable')


def remote_guard(policy_path,phase,instance):
    policy=verify_policy(policy_path);root=Path(policy['remote_root']);out=root/'external_guard_v2'/phase
    if sha(root/f'{phase}_plan.json')!=policy['prepared_bindings'][f'{phase}_plan.json']:
        raise ValueError('Remote phase plan changed')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(out/'state.json') if (out/'state.json').exists() else {'instance':instance,'phase':phase,'started':time.time(),'failed':False,'events':[],'known':{}}
    if state['instance']!=instance:raise ValueError('Never overwrite a prior guard generation')
    atomic(out/'launch.json',dict(instance=instance,identity=identity(os.getpid()),policy_sha256=sha(policy_path)))
    plan=read(root/f'{phase}_plan.json');jobs={j['name']:j for j in plan['jobs']};worker=policy['worker_path']
    stop=[]
    signal.signal(signal.SIGTERM,lambda *args:stop.append('guardian_SIGTERM'))
    signal.signal(signal.SIGINT,lambda *args:stop.append('guardian_SIGINT'))
    while True:
        now=time.time();live=[];heartbeat={}
        try:
            heartbeat=read(out/'heartbeat.json')
            if heartbeat.get('instance')!=instance:raise ValueError('Heartbeat generation mismatch')
            if now-heartbeat['time']>policy['heartbeat_timeout_seconds']:fail(state,'local_supervisor_heartbeat_expired')
            if heartbeat.get('abort'):fail(state,'local_supervisor_requested_abort')
            if stop:fail(state,stop[0])
            apps=[]
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
            for line in raw.splitlines():
                fields=[v.strip() for v in line.split(',')]
                if len(fields)!=3:raise ValueError('Malformed nvidia process row')
                apps.append((fields[0],int(fields[1]),int(fields[2])))
            for name,job in jobs.items():
                receipt=Path(plan['hosts']['silver']['root'])/'jobs'/name/'receipt.json'
                if not receipt.exists():continue
                r=read(receipt);w=r.get('worker')
                if not w:continue
                spec=job['command'][job['command'].index('--spec')+1]
                actual=identity(w['pid'])
                if not authorized_worker(w,actual,worker,spec):continue
                key=f"{w['pid']}:{w['start']}:{w['boot']}"
                state['known'].setdefault(key,dict(worker=dict(w,argv=actual['argv']),spec=spec,name=name))
                if not authorized_worker(state['known'][key]['worker'],actual,worker,spec):
                    fail(state,'own_worker_command_changed',worker_key=key)
                own=[a for a in apps if a[1]==w['pid']]
                memory=sum(a[2] for a in own);age=now-float(r['started'])
                for reason in check_limits(memory,[a[0] for a in own],r['gpu_uuid'],age,policy):fail(state,reason,worker_key=key)
                live.append(dict(name=name,worker=w,key=key,memory_MiB=memory,age=age))
            clear_errors(state)
        except Exception as exc:
            bounded_error(state,repr(exc),now,policy)
        if state['failed']:
            # Discover own workers even if nvidia-smi or heartbeat parsing failed.
            # Resource-probe failure must not prevent identity-bound cleanup.
            for name,job in jobs.items():
                receipt=Path(plan['hosts']['silver']['root'])/'jobs'/name/'receipt.json'
                try:
                    if not receipt.exists():continue
                    r=read(receipt);w=r.get('worker')
                    if not w:continue
                    spec=job['command'][job['command'].index('--spec')+1]
                    actual=identity(w['pid'])
                    if authorized_worker(w,actual,worker,spec):
                        key=f"{w['pid']}:{w['start']}:{w['boot']}"
                        state['known'].setdefault(key,dict(worker=dict(w,argv=actual['argv']),spec=spec,name=name))
                except Exception as exc:event(state,'cleanup_receipt_error',name=name,error=repr(exc))
            # Persist the violation before signals; success can never erase it.
            atomic(out/'state.json',state)
            for key,known in state['known'].items():
                try:signal_due(state,key,known['worker'],worker,known['spec'],now,policy,persist=lambda s:atomic(out/'state.json',s))
                except (ProcessLookupError,PermissionError) as exc:event(state,'signal_error',worker=known['worker'],error=repr(exc))
        state['last_check']=now;state['live']=live;atomic(out/'state.json',state)
        survivors=[v for v in state['known'].values() if authorized_worker(v['worker'],identity(v['worker']['pid']),worker,v['spec'])]
        # A disconnected local supervisor may have left its dispatcher alive.
        # Stay latched in FAILED_REAPER until its stop is acknowledged, killing
        # only newly appearing workers of these exact jobs. Never resume science.
        terminal_request=heartbeat.get('finished',False) or (state['failed'] and heartbeat.get('dispatcher_stopped',False))
        if terminal_request and not survivors:
            atomic(out/'done.json',final_record(state));return
        time.sleep(policy['poll_seconds'])


def ssh_script(policy,script):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',policy['address'],'python3','-'],
                          input=script,text=True,capture_output=True,timeout=20,check=True).stdout
def remote_exchange(policy,phase,instance,abort=False,finished=False,launch=False,dispatcher_stopped=False):
    # Structured JSON is embedded as a Python string, never as shell command text.
    request=dict(root=policy['remote_root'],phase=phase,instance=instance,abort=abort,finished=finished,launch=launch,dispatcher_stopped=dispatcher_stopped)
    code="""import json,pathlib,time,os,subprocess
r=json.loads(REQUEST);root=pathlib.Path(r['root']);out=root/'external_guard_v2'/r['phase'];out.mkdir(parents=True,exist_ok=True)
for name in ['launch','state','done']:
 old=out/(name+'.json')
 if old.exists() and json.loads(old.read_text()).get('instance')!=r['instance']:raise RuntimeError('Never overwrite previous guard-instance evidence')
p=out/'heartbeat.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(instance=r['instance'],time=time.time(),abort=r['abort'],finished=r['finished'],dispatcher_stopped=r['dispatcher_stopped'])));os.replace(t,p)
if r['launch']:
 log=open(out/'guardian.log','ab');subprocess.Popen(['python3',str(root/'external_guard_v2/guardian.py'),'remote','--policy',str(root/'external_guard_v2/policy.json'),'--phase',r['phase'],'--instance',r['instance']],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={}
for name in ['launch','state','done']:
 p=out/(name+'.json')
 if p.exists():result[name]=json.loads(p.read_text())
launch=result.get('launch',{}).get('identity');result['guardian_alive']=False
if launch:
 try:
  st=pathlib.Path('/proc/'+str(launch['pid'])+'/stat').read_text().split(') ',1)[1].split();boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  result['guardian_alive']=st[0]!='Z' and st[19]==str(launch['start']) and boot==launch['boot']
 except FileNotFoundError:pass
print(json.dumps(result))
""".replace('REQUEST',repr(json.dumps(request)))
    return json.loads(ssh_script(policy,code))
def stop_own_controller(state,policy,phase,now):
    expected=state.get('controller')
    if not expected:return
    actual=identity(expected['pid']);controller=policy['controller_path'];root=str(Path(policy['local_root'])/(phase+'_controller'))
    # Same guarded escalation primitive; controller argv must include its own exact state root.
    signal_due(state,'local_controller',expected,controller,root,now,policy)


def local_guard(policy_path,phase):
    import uuid
    policy=verify_policy(policy_path);require_local_origin(policy,policy_path);root=Path(policy['local_root']);out=root/'external_guard_v2'/phase
    if sha(policy['controller_path'])!=policy['controller_sha256']:raise ValueError('Frozen dispatcher source changed')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'local.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    release=read(root/'external_guard_v2/guard_release.json')
    if release.get('reviewer')!='root' or release.get('external_guard_authorized') is not True or release.get('policy_sha256')!=sha(policy_path):
        raise ValueError('Exact v2 external guardian review required')
    for rel,h in policy['prepared_bindings'].items():
        if sha(root/rel)!=h:raise ValueError('Prepared external/scientific artifact changed: '+rel)
    for arm in ['control','old7369','seven10481']:
        spec=root/'specs'/f'{phase}_{arm}'/'spec.json';release=read(spec.parent/'root_release.json')
        if release.get('reviewer')!='root' or release.get('technical_authorized') is not True or release.get('spec_sha256')!=sha(spec):
            raise ValueError('Exact root technical release required')
    if phase=='technical' and not read(root/'zero_summary.json').get('passed'):raise ValueError('Zero proof required first')
    prior_controller=root/(phase+'_controller')/'controller.json'
    if prior_controller.exists():
        prior=read(prior_controller).get('process')
        if prior and same(prior,identity(prior['pid'])):raise ValueError('Existing dispatcher must not be adopted or interrupted')
    prior_state=root/(phase+'_controller')/'state.json'
    if prior_state.exists() and any(j['status']!='pending' for j in read(prior_state)['jobs'].values()):
        raise ValueError('Prior dispatched phase requires separate explicit recovery, not fresh guardian launch')
    instance=str(uuid.uuid4());state=dict(instance=instance,phase=phase,started=time.time(),failed=False,events=[])
    atomic(out/'local_launch.json',dict(identity=identity(os.getpid()),instance=instance,policy_sha256=sha(policy_path)))
    stop=[];signal.signal(signal.SIGTERM,lambda *args:stop.append('local_guard_SIGTERM'));signal.signal(signal.SIGINT,lambda *args:stop.append('local_guard_SIGINT'))
    remote={};started_dispatcher=False;abort_started=None;finished=False;abort_restart_used=False;launch_attempted=False
    try:
        remote_exchange(policy,phase,instance,launch=True)
        deadline=time.time()+30
        while time.time()<deadline:
            remote=remote_exchange(policy,phase,instance)
            if remote.get('launch',{}).get('instance')==instance and remote.get('guardian_alive'):
                if remote.get('state',{}).get('failed') or remote.get('done'):raise RuntimeError('Remote guard already failed/terminal')
                break
            time.sleep(1)
        else:raise RuntimeError('Remote guardian startup not acknowledged')
        controller_record=root/(phase+'_controller')/'controller.json'
        if controller_record.exists():
            old=read(controller_record).get('process')
            if old and same(old,identity(old['pid'])):raise RuntimeError('Refuse to adopt a pre-existing dispatcher')
        launch_attempted=True
        launch_result=subprocess.run([policy['local_python'],policy['controller_path'],'launch','--root',str(root/(phase+'_controller')),'--manifest',str(root/(phase+'_plan.json'))],check=True,capture_output=True,text=True,timeout=20)
        started_dispatcher=True
        # Frozen controller PID identity, not a process-group kill.
        pid=json.loads(launch_result.stdout)['controller_pid']
        actual=identity(pid)
        if not actual or policy['controller_path'] not in actual['argv']:raise RuntimeError('Cannot bind own controller identity')
        state['controller']=actual
    except Exception as exc:
        fail(state,'startup_error',error=repr(exc))
        if launch_attempted and not state.get('controller'):
            # A launch RPC can time out after creating the dispatcher. Recover only
            # the previously absent exact state-root identity, never another queue.
            record=root/(phase+'_controller')/'controller.json'
            if record.exists():
                candidate=read(record).get('process');actual=identity(candidate['pid']) if candidate else None
                if candidate and authorized_worker(candidate,actual,policy['controller_path'],str(root/(phase+'_controller'))):
                    state['controller']=actual;started_dispatcher=True
    while True:
        now=time.time()
        if stop:fail(state,stop[0])
        try:
            if started_dispatcher and not finished:
                cs=read(root/(phase+'_controller')/'state.json');statuses=[j['status'] for j in cs['jobs'].values()]
                finished,jobs_failed=phase_status(statuses)
                if jobs_failed:fail(state,'controller_job_failure',statuses=statuses)
                if not finished and not same(state['controller'],identity(state['controller']['pid'])):fail(state,'controller_died_before_terminal')
            controller_stopped=not started_dispatcher or bool(state.get('controller') and not same(state['controller'],identity(state['controller']['pid'])))
            remote=remote_exchange(policy,phase,instance,abort=state['failed'],finished=finished,dispatcher_stopped=controller_stopped)
            for key in ('launch','state','done'):
                if key in remote and remote[key].get('instance')!=instance:raise RuntimeError('Remote supervisor generation mismatch')
            if not remote.get('guardian_alive') and not remote.get('done'):
                fail(state,'remote_guard_died')
                if not abort_restart_used:
                    abort_restart_used=True
                    # Re-enter the same persisted state in abort-only mode; never restart science.
                    remote_exchange(policy,phase,instance,abort=True,launch=True,dispatcher_stopped=controller_stopped)
            if remote.get('state',{}).get('failed') or remote.get('done',{}).get('failed'):fail(state,'remote_guard_violation')
            clear_errors(state)
        except Exception as exc:bounded_error(state,repr(exc),now,policy)
        if state['failed']:
            abort_started=abort_started or now
            # Stop only our dispatcher: existing workers are terminated by remote guardian,
            # even after link loss because heartbeat expiry is independent of SSH.
            try:stop_own_controller(state,policy,phase,now)
            except Exception as exc:event(state,'controller_signal_error',error=repr(exc))
        state.update(last_check=now,remote=remote,started_dispatcher=started_dispatcher,finished_controller=finished)
        atomic(out/'local_state.json',state)
        if remote.get('done'):
            result=final_record(state);result['remote']=remote['done'];result['passed']=result['passed'] and remote['done']['passed'] and finished
            atomic(out/'local_done.json',result);return
        if abort_started and now-abort_started>policy['heartbeat_timeout_seconds']+policy['term_grace_seconds']+30:
            result=final_record(state);result.update(passed=False,remote_completion_unconfirmed=True,remote_watchdog_retains_deadline=True)
            atomic(out/'local_done.json',result);return
        time.sleep(policy['poll_seconds'])


def launch(policy_path,phase):
    policy=verify_policy(policy_path);require_local_origin(policy,policy_path)
    out=Path(policy['local_root'])/'external_guard_v2'/phase;out.mkdir(parents=True,exist_ok=True)
    previous=out/'local_launch.json'
    if (out/'local_done.json').exists():raise ValueError('Terminal guard evidence is immutable; prepare a new reviewed generation to retry')
    if previous.exists() and same(read(previous)['identity'],identity(read(previous)['identity']['pid'])):
        raise ValueError('Local guardian already running')
    log=open(out/'local_guardian.log','ab')
    p=subprocess.Popen([policy['local_python'],str(Path(__file__).resolve()),'local','--policy',str(Path(policy_path).resolve()),'--phase',phase],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=p.pid,phase=phase,local_only=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['launch','local','remote']);p.add_argument('--policy',type=Path,required=True);p.add_argument('--phase',choices=['zero','technical'],required=True);p.add_argument('--instance')
    a=p.parse_args()
    if a.command=='remote':remote_guard(a.policy,a.phase,a.instance)
    elif a.command=='local':local_guard(a.policy,a.phase)
    else:launch(a.policy,a.phase)
