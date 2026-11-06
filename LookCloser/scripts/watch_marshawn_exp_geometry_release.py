"""Root-authorized conditional P/PB queue. No model promotion or implicit release."""
import argparse,fcntl,hashlib,json,math,os,shlex,subprocess,sys,time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def once(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Existing immutable receipt differs: '+str(path))
    else:atomic(path,value)


def eligible(parent,policy):
    if parent.get('config')!=policy['expected_parent_config']:raise ValueError('Parent configuration drift')
    if parent.get('technical_smoke')is not False or parent.get('final_step',0)<106316 or parent.get('plateau',{}).get('stopped')is not True:return False
    selected=parent['selection'];metrics=selected['metrics']
    for key in ('psnr','ssim','lpips'):
        value=metrics[key]
        if type(value)not in(int,float)or not math.isfinite(value):raise ValueError('Invalid parent metric')
    if not 0<=metrics['lpips']<=policy['maximum_actor_lpips']:return False
    for actor,reference in policy['reference_actor_lpips'].items():
        value=selected['by_actor'][actor]['lpips']
        if type(value)not in(int,float)or not math.isfinite(value)or value<0:raise ValueError('Invalid per-actor metric')
        if value>reference*policy['maximum_actor_ratio']:return False
    # Explicitly bind the six-record v1 protocol; a scalar bearing the same name
    # is not enough to admit a different evaluation domain.
    domain='fixed GT actor crop; identical RGB0.5 exterior; AlexNet LPIPS without resize'
    records=selected.get('records',[])
    if selected.get('domain')!=domain or selected.get('aggregation')!='equal actors per camera, then equal cameras':raise ValueError('Parent v1 protocol differs')
    if len(records)!=6 or {(r.get('view'),r.get('actor'))for r in records}!={(v,a)for v in ('DC','HC','KA')for a in ('black','grey')}or any(r.get('domain')!=domain for r in records):raise ValueError('Incomplete v1 six-pair evidence')
    return True


def precedence_ready(state,policy):
    jobs=state.get('jobs',{})
    terminal={'complete','failed','skipped','blocked'}
    for name in policy['precedence_jobs']:
        item=jobs.get(name,{})
        if item.get('status')in terminal:continue
        if item.get('status')not in {'running','starting','dispatching'}or item.get('host')is None or item.get('gpu')is None:return False
    return True


def validate_policy(root,policy):
    if policy.get('conditional_experiment_launch_authorized')is not True or policy.get('model_promotion_authorized')is not False:raise ValueError('Explicit experiment-only root authorization required')
    if not policy.get('reviewer')or not policy.get('reviewed_at'):raise ValueError('Named root review required')
    if policy['maximum_actor_lpips']!=.06 or policy['maximum_actor_ratio']!=1.03:raise ValueError('Reviewed parent guards changed')
    if set(policy['reference_actor_lpips'])!={'black','grey'}:raise ValueError('Both actor references required')
    for name,digest in policy['reviewed_artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve())or sha(p)!=digest:raise ValueError('Reviewed artifact drift: '+name)
    if sha(Path(__file__))!=policy['reviewed_artifacts'][Path(__file__).name]:raise ValueError('Watcher implementation is not reviewed')


def release_values(root,policy,parent_raw,policy_hash):
    result={}
    for arm in ('P','PB'):
        value=json.loads((root/f'release_{arm}.pending.json').read_text())
        value.update(approved_for_training=True,reviewer=policy['reviewer'],reviewed_at=policy['reviewed_at'],
            pending=[],completed_parent_receipt_sha256=hashlib.sha256(parent_raw).hexdigest(),conditional_policy_sha256=policy_hash,
            model_promotion_authorized=False,geometry_quality_approved=False)
        result[arm]=value
    return result


def remote_launch(root,payload):
    policy=payload['policy'];policy_raw=payload['policy_raw'].encode()
    if json.loads(policy_raw)!=policy or hashlib.sha256(policy_raw).hexdigest()!=payload['policy_sha256']:raise ValueError('Policy transport changed')
    validate_policy(root,policy)
    raw=Path(policy['parent_complete_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=payload['parent_sha256']or not eligible(json.loads(raw),policy):raise ValueError('Completed parent changed or ineligible')
    once(root/'conditional_launch_policy.approved.json',policy)
    for arm,value in release_values(root,policy,raw,payload['policy_sha256']).items():once(root/f'release_{arm}.json',value)
    sys.path.insert(0,str(root/'code/LookCloser/scripts'))
    from run_marshawn_exp_geometry_release import verify
    for arm in ('P','PB'):verify(root/f'spec_{arm}.json')
    controller=root/'code/LookCloser/scripts/marshawn_tuning_controller.py'
    subprocess.run([sys.executable,str(controller),'launch','--root',str(root/'controller'),'--manifest',str(root/'controller_plan.json')],check=True)
    # The audit is a separate read-only process; its own durable lock prevents
    # duplicate work if a launch acknowledgement is lost.
    if policy.get('audit_103_authorized')is True:
        with(root/'audit_queue.log').open('ab')as log:
            subprocess.Popen([sys.executable,str(root/'watch_marshawn_exp_geometry_audits.py'),'--campaign',str(root.parent)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(launched=True,model_promoted=False,parent_sha256=payload['parent_sha256'],policy_sha256=payload['policy_sha256'])


def watch(path):
    root=path.resolve().parent;raw_policy=path.read_text();policy=json.loads(raw_policy);digest=sha(path)
    lock=(root/'conditional_release.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if sha(path)!=digest:raise ValueError('Policy changed after launch')
        validate_policy(root,policy);check=dict(time=time.time(),policy_sha256=digest,parent_available=False)
        try:
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',policy['parent_host'],shlex.join(['cat',policy['parent_complete_path']])],capture_output=True,timeout=25)
            if r.returncode:check['read_error']=r.stderr.decode(errors='replace')[-300:]
            else:
                check['parent_available']=True;parent=json.loads(r.stdout);check['eligible']=eligible(parent,policy)
                if not check['eligible']:
                    atomic(root/'conditional_release_blocked.json',check)
                    with(root/'conditional_release_checks.jsonl').open('a')as log:log.write(json.dumps(check)+'\n')
                    return
                queue=Path(policy['precedence_controller_state'])
                check['precedence_ready']=precedence_ready(json.loads(queue.read_text())if queue.exists()else{},policy)
                if check['precedence_ready']:
                    payload=dict(policy=policy,policy_raw=raw_policy,policy_sha256=digest,parent_sha256=hashlib.sha256(r.stdout).hexdigest())
                    (root/'observed_parent_complete.json').write_bytes(r.stdout)
                    command=[policy['remote_python'],str(Path(policy['remote_root'])/Path(__file__).name),'--remote-launch','--root',policy['remote_root']]
                    launch=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',policy['parent_host'],shlex.join(command)],input=json.dumps(payload).encode(),capture_output=True,timeout=120)
                    with(root/'conditional_launch.log').open('ab')as log:log.write(launch.stdout+launch.stderr)
                    if launch.returncode:raise RuntimeError('Remote release/launch failed; see log')
                    for arm,value in release_values(root,policy,r.stdout,digest).items():once(root/f'release_{arm}.json',value)
                    reporter=root/'reporter'
                    subprocess.run([sys.executable,str(reporter/'report_marshawn_cross_campaign.py'),'launch','--root',str(reporter)],check=True)
                    check.update(experiment_queue_launched=True,model_promoted=False);atomic(root/'conditional_release_complete.json',check)
        except (OSError,subprocess.SubprocessError)as error:check['transport_error']=repr(error)
        with(root/'conditional_release_checks.jsonl').open('a')as log:log.write(json.dumps(check)+'\n')
        if check.get('experiment_queue_launched'):return
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path);p.add_argument('--remote-launch',action='store_true');p.add_argument('--root',type=Path);a=p.parse_args()
    if a.remote_launch:print(json.dumps(remote_launch(a.root,json.load(sys.stdin))))
    else:
        try:watch(a.policy)
        except Exception as error:
            atomic(a.policy.resolve().parent/'conditional_release_error.json',dict(time=time.time(),error=repr(error),training_failure_claim=False))
            raise
