"""Durable multi-host cold-frame queue with bounded same-frame crash retries.

Reuses the existing fleet controller's global GPU locks and SSH-UNKNOWN rules.
Each retry has a new immutable attempt ID and the SAME frame directory/host.
A separate watchdog restarts a dead dispatcher, adopting surviving supervisors.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
import re
import marshawn_tuning_controller as fleet


def create(root,manifest):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);plan=fleet.read(manifest)
    if plan.get('schema')!=1 or not plan.get('hosts'):raise ValueError('Expected schema1 video plan with hosts')
    if not re.fullmatch('[0-9a-f]{64}',plan.get('pipeline_sha256','')):raise ValueError('Sealed pipeline SHA256 required')
    frames=plan.get('frames',list(range(1660,2020,5)))
    if len(set(frames))!=len(frames) or any(type(f)!=int or f<1660 for f in frames):raise ValueError('Unique global frame IDs required')
    if not 0<float(plan.get('poll_seconds',30))<=60:raise ValueError('Poll interval must be at most60seconds')
    frozen=root/'video_plan.json'
    if frozen.exists() and fleet.read(frozen)!=plan:raise ValueError('Video plan immutable')
    if not frozen.exists():fleet.atomic(frozen,plan)
    state=fleet.read(root/'state.json')
    if state is None:
        slots=[key for key,h in plan['hosts'].items() for _ in h['gpus']]
        logical={str(f):dict(host=slots[i%len(slots)],attempt=0,name='frame%06d_a00'%f) for i,f in enumerate(frames)}
        state=dict(schema=1,manifest_sha256=fleet.digest(plan),jobs={},frames=logical,created=time.time())
        for item in logical.values():state['jobs'][item['name']]={'status':'pending'}
        fleet.atomic(root/'state.json',state)
    if state['manifest_sha256']!=fleet.digest(plan):raise ValueError('State/manifest differs')
    return plan,state


def jobs(plan,state):
    result=[]
    for frame,item in state['frames'].items():
        host=plan['hosts'][item['host']]
        root=Path(host['root']);code=root/'code/LookCloser/scripts'
        result.append(dict(name=item['name'],host=item['host'],
            command=[host['python'],str(code/'marshawn_video_worker.py'),'--campaign',str(root),
                     '--frame',frame,'--pipeline',str(root/'pipeline.json'),'--pipeline-sha256',plan['pipeline_sha256']],
            cwd=str(root),env={'PYTHONPATH':str(root/'code')+':'+str(code)},
            progress_files=[str(root/'frames'/('%06d'%int(frame))/'pipeline_progress.json'),
                            str(root/'frames'/('%06d'%int(frame))/'train/progress.json')],
            metrics_file=str(root/'frames'/('%06d'%int(frame))/'frame_completion.json')))
    return result


def eligible(plan,state,job):
    pattern=plan.get('prepared_dispatch_marker')
    if not pattern:return True
    frame=next(int(f) for f,item in state['frames'].items() if item['name']==job['name'])
    marker=Path(pattern.replace('{frame}','%06d'%frame))
    receipt=fleet.read(marker,{})
    return receipt.get('frame')==frame and receipt.get('host')==job['host']


def retry_failed(plan,state,now=None):
    now=time.time() if now is None else now;changed=[]
    for frame,item in state['frames'].items():
        attempt=state['jobs'][item['name']]
        if attempt['status']!='failed':continue
        # Even a FAILED cached receipt is insufficient without a fresh remote
        # probe proving neither supervisor nor its child remains alive.
        snapshot=state.get('host_snapshots',{}).get(item['host'],{})
        remote=snapshot.get('jobs',{}).get(item['name'])
        if not remote or 'error' in snapshot or remote.get('supervisor_alive') or remote.get('worker_alive'):continue
        if item['attempt']+1>=int(plan.get('max_attempts',12)):continue
        delay=min(1800,float(plan.get('retry_base_seconds',60))*2**item['attempt'])
        if now<float(attempt.get('finished',attempt.get('dispatched',now)))+delay:continue
        item['attempt']+=1;item['name']='frame%06d_a%02d'%(int(frame),item['attempt'])
        state['jobs'][item['name']]={'status':'pending'};changed.append(int(frame))
    return changed


def dispatch(root):
    root=Path(root).resolve();held=fleet.lock(root/'controller.lock')
    if held is None:return 0
    plan,state=create(root,root/'video_plan.json')
    fleet.atomic(root/'controller.json',dict(process=fleet.identity(),started=time.time()))
    while True:
        current=[job for job in jobs(plan,state) if eligible(plan,state,job)]
        # Include only current attempts in the plan; old immutable receipt states
        # remain in the ledger and cannot be relaunched.
        fleet.tick(dict(schema_version=1,hosts=plan['hosts'],jobs=current,poll_seconds=plan.get('poll_seconds',30)),state,root)
        changed=retry_failed(plan,state)
        if changed:fleet.append(root/'retries.jsonl',dict(time=time.time(),frames=changed))
        fleet.atomic(root/'state.json',state)
        selected=[state['jobs'][item['name']]['status'] for item in state['frames'].values()]
        fleet.atomic(root/'summary.json',dict(time=time.time(),frames=len(selected),
            counts={k:selected.count(k) for k in set(selected)},
            next_supervision_deadline=time.time()+1800,
            all_complete=all(s=='complete' for s in selected)))
        if all(s=='complete' for s in selected):
            command=plan.get('after_training_command')
            if command and not (root/'postprocess_complete.json').exists():
                with (root/'postprocess.log').open('ab') as stream:
                    code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
                if code:raise RuntimeError('Postprocess failed; watchdog will resume it')
                fleet.atomic(root/'postprocess_complete.json',dict(time=time.time(),command=command))
            fleet.atomic(root/'campaign_complete.json',dict(time=time.time(),frames=len(selected)))
            return 0
        # Exhausted failures remain visible and continue supervision; never mark
        # a partial campaign complete or silently skip a global frame.
        time.sleep(float(plan.get('poll_seconds',30)))


def watchdog(root):
    root=Path(root).resolve();held=fleet.lock(root/'watchdog.lock')
    if held is None:return 0
    fleet.atomic(root/'watchdog.json',dict(process=fleet.identity(),started=time.time()))
    while not (root/'campaign_complete.json').exists():
        controller=fleet.read(root/'controller.json',{})
        if not fleet.alive(controller.get('process')):
            with (root/'controller.log').open('ab') as stream:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_dispatch','--root',str(root)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            fleet.append(root/'watchdog_checks.jsonl',dict(time=time.time(),action='dispatcher_started',pid=process.pid))
        else:fleet.append(root/'watchdog_checks.jsonl',dict(time=time.time(),action='dispatcher_alive',process=controller['process']))
        time.sleep(30)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','launch','status','_dispatch','_watchdog'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
    if a.action=='_dispatch':return dispatch(a.root)
    if a.action=='_watchdog':return watchdog(a.root)
    if a.action=='status':
        print(json.dumps(dict(summary=fleet.read(a.root/'summary.json'),
            controller_alive=fleet.alive(fleet.read(a.root/'controller.json',{}).get('process')),
            watchdog_alive=fleet.alive(fleet.read(a.root/'watchdog.json',{}).get('process'))),indent=2));return 0
    plan,state=create(a.root,a.manifest or a.root/'video_plan.json')
    if a.action=='prepare':print(json.dumps({'frames':len(state['frames']),'hosts':list(plan['hosts'])}));return 0
    from marshawn_video_train import verify_code
    verify_code(a.root)
    if fleet.alive(fleet.read(a.root/'watchdog.json',{}).get('process')):print('Watchdog already alive');return 0
    with (a.root/'watchdog.log').open('ab') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_watchdog','--root',str(a.root.resolve())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'watchdog_pid':process.pid,'frames':len(state['frames'])}));return 0


if __name__=='__main__':sys.exit(main())
