"""Ten-GPU HARD140000 revision; source inputs remain in the archived campaign.

The ledger and plan are separate from the original campaign. Frame assignments
are claimed once, after source/gate qualification, and shared with transient
source-preparation schedulers. This controller never moves a live model.
"""
from pathlib import Path
import argparse, copy, hashlib, importlib, json, os, shlex, subprocess, sys, time, uuid


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def modules(V):
    sys.path[:0] = [str(V / 'code'), str(V / 'code/LookCloser/scripts')]
    fleet = importlib.import_module('marshawn_tuning_controller')
    campaign = importlib.import_module('marshawn_video_campaign')
    stage = importlib.import_module('marshawn_video_stage')
    return fleet, campaign, stage


def configuration(F, C, V):
    fleet, _, _ = modules(V); baseline = read(V / 'video_plan.json')
    recipe=read(C/'recipe.json')
    if recipe.get('hard_max_iterations')!=140000:raise ValueError('HARD140000 recipe required')
    hosts = copy.deepcopy(baseline['hosts'])
    hosts['silver'].update(root='/cache/ubuntu/lookcloser/campaigns/'+C.name,controller='/cache/ubuntu/lookcloser/campaigns/'+C.name+'/code/LookCloser/scripts/marshawn_tuning_controller.py')
    hosts['olive'] = dict(python='/home/pavlo/.conda/envs/nerfstudio/bin/python', root=str(C),
                          controller=str(C / 'code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=[0])
    hosts['dev3'] = dict(address='ubuntu@dev3', python='/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',
                         root=str(C), controller=str(C / 'code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=[0])
    for key, host in hosts.items():
        host['operational_controller'] = str(Path(host['root']) / 'operations/fleet10/marshawn_fleet10_140k.py')
    result = dict(baseline, schema=1, fleet_version=10, scientific_campaign=str(C), source_campaign=str(V), hard_max_iterations=140000,
                  baseline_plan_sha256=sha(V / 'video_plan.json'), hosts=hosts,
                  assignments=str(V / 'operations/fleet10/assignments'), demands=str(V / 'operations/fleet10/demands'),
                  prepared_dispatch_marker=str(C/'frames/{frame}/dispatch_ready.json'), host_preference=['olive', 'dev3', 'silver'], controller_source_sha256=sha(__file__),
                  source_code_seal_sha256=sha(C / 'code_seal.json'), pipeline_sha256=sha(C/'pipeline.json'),
                  postprocess_command_file=str(F / 'postprocess_command.json'))
    result.pop('after_training_command', None)
    if sum(len(h['gpus']) for h in hosts.values()) != 10:
        raise ValueError('Expected eight Silver plus Olive and DEV3 GPUs')
    return result


def guard(F):
    plan = read(F / 'video_plan.json'); V = Path(plan['source_campaign']); C=Path(plan['scientific_campaign'])
    if sha(__file__) != plan['controller_source_sha256'] or sha(C / 'code_seal.json') != plan['source_code_seal_sha256']:
        raise ValueError('Fleet or immutable scientific code source changed')
    if sha(V / 'video_plan.json') != plan['baseline_plan_sha256']:
        raise ValueError('Original scientific campaign plan changed')
    return plan, V


def prepare(F, C, V):
    F.mkdir(parents=True, exist_ok=True); plan = configuration(F, C, V); fleet, _, _ = modules(V)
    previous = read(F / 'video_plan.json')
    if previous is not None and previous != plan:
        raise ValueError('Versioned fleet plan is immutable')
    if previous is None:
        fleet.atomic(F / 'video_plan.json', plan)
    return plan




def jobs(plan, state, campaign):
    filtered = dict(state, frames={f: row for f, row in state['frames'].items() if row.get('host')})
    result = campaign.jobs(plan, filtered)
    for job in result:
        row = next(item for item in filtered['frames'].values() if item['name'] == job['name'])
        if row.get('gpu') is not None:
            job['gpu'] = row['gpu']
    return result


def preflight(F):
    plan,V=guard(F);fleet,campaign,_=modules(V);C=Path(plan['scientific_campaign'])
    if read(C/'recipe.json').get('hard_max_iterations')!=140000:raise ValueError('Uncapped recipe')
    proof={}
    for key,host in plan['hosts'].items():
        for gpu in host['gpus']:
            state=dict(frames={'1660':dict(host=key,gpu=gpu,name='frame001660_a00')},jobs={'frame001660_a00':dict(status='pending')})
            job=jobs(plan,state,campaign)[0]
            if str(Path(host['root'])/'code/LookCloser/scripts/marshawn_video_worker.py')not in job['command']:raise ValueError('Wrong capped worker')
            if plan['pipeline_sha256']not in job['command']:raise ValueError('Wrong pipeline')
            proof[key+':'+str(gpu)]=dict(command=job['command'],gpu=gpu,cold_new_revision=True)
    result=dict(fleet_plan_sha256=sha(F/'video_plan.json'),capacity=10,hard_max_iterations=140000,
                all72_new_revision=str(C),old_overcap_weights_forbidden=True,slot_commands=proof,no_GPU_work_started=True,time=time.time())
    fleet.atomic(F/'preflight.json',result);return result


def probe_slots(host_root):
    fleet, _, _ = modules(host_root); result = fleet.probe(host_root)
    for key, device in result['gpus'].items():
        lease = fleet.lock(Path('/tmp') / ('lookcloser_tuning_gpu_%s_%s.lock' % (os.getuid(), key)))
        device['global_lease_free'] = lease is not None
        if lease is not None:
            lease.close()
    return result


def slot_snapshots(plan, F):
    result = {}
    for key, host in plan['hosts'].items():
        command = [host['python'], host['operational_controller'], '_slots', '--host-root', host['root'],
                   '--expected-source-sha', plan['controller_source_sha256']]
        if host.get('address'):
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host['address'], shlex.join(command)]
        try:
            reply = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            result[key] = json.loads(reply.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result[key] = dict(error=str(error))
    return result


def select_slot(plan, state, assignments, snapshots):
    occupied = set(); counts = {h: 0 for h in plan['hosts']}
    for row in state['frames'].values():
        job = state['jobs'][row['name']]
        if row.get('host') and job['status'] != 'complete':
            occupied.add((row['host'], row.get('gpu'))); counts[row['host']] += 1
    for row in assignments:
        frame = state['frames'].get(str(row['frame']), {})
        if not frame.get('host'):
            occupied.add((row['host'], row['gpu'])); counts[row['host']] += 1
    rank = {key: i for i, key in enumerate(plan['host_preference'])}
    candidates = []
    for key, host in plan['hosts'].items():
        snapshot = snapshots.get(key, {})
        if snapshot.get('error') or snapshot.get('gpu_error'):
            continue
        for gpu in host['gpus']:
            device = snapshot.get('gpus', {}).get(str(gpu), {})
            if (key, gpu) in occupied or not device or device.get('processes') or not device.get('global_lease_free') or device.get('used_mib', 999999) >= 1024:
                continue
            candidates.append((counts[key] / len(host['gpus']), rank[key], gpu, key))
    if not candidates:
        return None
    _, _, gpu, key = min(candidates)
    return key, gpu


def qualify_frame_permissions(plan, V, frame, host_key):
    """Grant DEV3 UID1000 only its assigned frame and immutable attempt slots."""
    if host_key != 'dev3':
        return None
    frame_root = V / 'frames' / ('%06d' % frame); frame_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['setfacl', '-R', '-P', '-m', 'u:1000:rwX', str(frame_root)], check=True, capture_output=True)
    for directory in [frame_root] + [p for p in frame_root.rglob('*') if p.is_dir() and not p.is_symlink()]:
        subprocess.run(['setfacl', '-m', 'd:u:1000:rwx,d:u:1001:rwx', str(directory)], check=True, capture_output=True)
    # Shared FSX supervisor roots stay disjoint by immutable frame/attempt ID.
    for attempt in range(plan['max_attempts']):
        directory = V / 'jobs' / ('frame%06d_a%02d' % (frame, attempt)); directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(['setfacl', '-m', 'u:1000:rwx,d:u:1000:rwx,d:u:1001:rwx', str(directory)], check=True, capture_output=True)
    host = plan['hosts']['dev3']; proof = frame_root / '.fleet10_dev3_write_qualification.json'
    code = ('import pathlib,json,os; p=pathlib.Path(%r); '
            'p.write_text(json.dumps(dict(uid=os.getuid(),frame=%d,scope="assigned frame only"))+"\\n"); '
            'print(p.read_text())') % (str(proof), frame)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host['address'],
                             shlex.join([host['python'], '-c', code])], check=True, capture_output=True, text=True, timeout=30)
    record = json.loads(result.stdout)
    if record['uid'] != 1000 or read(proof) != record:
        raise ValueError('DEV3 shared-FSX write qualification failed')
    return dict(receipt=str(proof), sha256=sha(proof), uid=1000)


def reserve(F, frame, source_proof=None, force_host=None, force_gpu=None, resume_authorization=None):
    plan, V = guard(F); fleet, _, stage = modules(V)
    C=Path(plan['scientific_campaign']); data = V / 'frames' / ('%06d' % frame) / 'data'
    safe = V / 'operations/advanced_inputs' / ('%06d' % frame) / 'safe_gate_before_dispatch.json'
    marker = data.parent / 'native_semantic_ready.json'
    if source_proof:
        safe=Path(source_proof)
        bootstrap_document=read(safe,{})
        if bootstrap_document.get('kind')=='SAME_FRAME_INPUT_REUSE_FOR_HARD140K_REVISION_V1':
            data=Path(bootstrap_document['data_root'])
    control = V / 'operations/fleet10'; control.mkdir(parents=True, exist_ok=True)
    held = fleet.lock(control / 'assignment.lock')
    if held is None:
        return dict(status='awaiting_assignment_lock')
    if not safe.exists() or (not marker.exists() and not source_proof):
        raise ValueError('Source-qualified semantic and current safe-gate receipts required')
    proof = read(safe); manifest = stage.inventory(data, frame)
    marker_sha=sha(marker) if marker.exists() else None
    bootstrap=bool(source_proof and frame in {1660,1665,1670} and proof.get('kind')=='SAME_FRAME_INPUT_REUSE_FOR_HARD140K_REVISION_V1' and proof.get('authorized')is True)
    if proof['frame'] != frame or proof['final_ready_sha256'] != sha(data / 'ready.json') or (not bootstrap and proof['source_marker_sha256'] != marker_sha):
        raise ValueError('Final safe source input binding changed')
    if bootstrap and (proof['input_manifest_sha256']!=stage.digest(manifest) or proof['admission_manifest_sha256']!=sha(data/'admission/manifest.json') or proof['target_code_seal_sha256']!=plan['source_code_seal_sha256']):
        raise ValueError('Bootstrap exact same-frame data/grid binding changed')
    state = read(F / 'state.json')
    if state is None:
        raise ValueError('Fleet must be adopted before assigning new frames')
    assignment_path = Path(plan['assignments']) / ('%06d.json' % frame)
    expected = dict(frame=frame, input_manifest_sha256=stage.digest(manifest), data_ready_sha256=sha(data / 'ready.json'),
                    native_semantic_ready_sha256=marker_sha, safe_gate_before_dispatch_sha256=sha(safe),
                    source_proof_kind=proof.get('kind','CURRENT_NATIVE_SAFE_GATE'),
                    fleet_plan_sha256=sha(F / 'video_plan.json'))
    existing = read(assignment_path)
    if existing:
        if any(existing.get(k) != value for k, value in expected.items()):
            raise ValueError('Assigned frame source or plan changed')
        return dict(status='reserved', assignment_path=str(assignment_path), assignment=existing)
    resume=False
    if resume_authorization:
        r=read(resume_authorization)
        resume=bool(frame==1670 and r.get('authorized')is True and r.get('frame')==1670 and r.get('same_frame_resume')is True and r.get('hard_max_iterations')==140000)
        if not resume or force_host!='silver' or force_gpu!=1:raise ValueError('1670 cap-resume requires explicit source-bound SG1 authorization')
    if (C/'frames'/('%06d'%frame)/'dispatch_ready.json').exists() or ((C/'frames'/('%06d'%frame)/'spec.json').exists() and not resume):
        raise ValueError('Cannot assign an already published/bound frame')
    demand_path = Path(plan['demands']) / ('%06d.json' % frame)
    fleet.atomic(demand_path, dict(expected, status='awaiting_idle_training_slot', ready_for_training=True,
                 preferred_host=force_host,preferred_gpu=force_gpu,time=time.time()))
    snapshots = slot_snapshots(plan, F); fleet.atomic(F / 'slot_snapshots.json', snapshots)
    if force_host is not None:
        if force_host not in plan['hosts'] or force_gpu not in plan['hosts'][force_host]['gpus']:raise ValueError('Invalid explicit training slot')
        snapshots={k:s for k,s in snapshots.items() if k==force_host}
        snapshots[force_host]['gpus']={k:v for k,v in snapshots[force_host].get('gpus',{}).items() if str(k)==str(force_gpu)}
    assignments = [read(p) for p in Path(plan['assignments']).glob('*.json')]
    selected = select_slot(plan, state, assignments, snapshots)
    if selected is None:
        return dict(status='awaiting_idle_training_slot', frame=frame)
    host_key, gpu = selected; host = plan['hosts'][host_key]
    permission = qualify_frame_permissions(plan, C, frame, host_key)
    assignment = dict(expected, kind='MARSAWN_FLEET10_FRAME_ASSIGNMENT_V1', host=host_key, gpu=gpu,
                      host_root=host['root'], python=host['python'], address=host.get('address'),
                      assignment_nonce=uuid.uuid4().hex, status='reserved', time=time.time(),
                      frame_permission_qualification=permission,hard_max_iterations=140000,
                      same_frame_resume_authorization_sha256=sha(resume_authorization) if resume else None)
    fleet.atomic(assignment_path, assignment); fleet.atomic(demand_path, dict(expected, status='assigned', host=host_key, gpu=gpu))
    return dict(status='reserved', assignment_path=str(assignment_path), assignment=assignment)


def adopt(F):
    plan,V=guard(F);fleet,_,_=modules(V)
    for name in ['controller.json','watchdog.json']:
        if fleet.alive(read(V/name,{}).get('process')):raise ValueError('Retire original uncapped CPU orchestrators first')
    existing=read(F/'state.json')
    if existing:return existing
    logical={str(frame):dict(host=None,gpu=None,attempt=0,name='frame%06d_a00'%frame)for frame in plan['frames']}
    state=dict(schema=1,fleet_version=10,manifest_sha256=fleet.digest(plan),frames=logical,jobs={r['name']:dict(status='pending')for r in logical.values()},created=time.time(),hard_max_iterations=140000,no_old_worker_adoption=True)
    fleet.atomic(F/'state.json',state)
    fleet.atomic(V/'operations/fleet10_active.json',dict(kind='FLEET10_ACTIVE_LEDGER_V1',controller_root=str(F),plan=str(F/'video_plan.json'),plan_sha256=sha(F/'video_plan.json'),state=str(F/'state.json'),assignments=plan['assignments'],demands=plan['demands'],capacity=10,hard_max_iterations=140000,scientific_campaign=plan['scientific_campaign']))
    return state


def dispatch(F):
    plan, V = guard(F); fleet, campaign, _ = modules(V); held = fleet.lock(F / 'controller.lock')
    if held is None:
        return
    release=read(Path(plan['scientific_campaign'])/'release.json')
    if release.get('production_enabled')is not True or release.get('code_seal_sha256')!=plan['source_code_seal_sha256']:
        raise ValueError('Capped scientific release is not enabled')
    state = read(F / 'state.json'); fleet.atomic(F / 'controller.json', dict(process=fleet.identity(), started=time.time()))
    while True:
        guard(F)
        for path in Path(plan['assignments']).glob('*.json'):
            assignment = read(path); row = state['frames'][str(assignment['frame'])]
            if row.get('host') and (row['host'] != assignment['host'] or row.get('gpu') != assignment['gpu']):
                raise ValueError('Frame host/GPU ownership changed')
            if not row.get('host'):
                row.update(host=assignment['host'], gpu=assignment['gpu'], assignment_sha256=sha(path))
        current = [job for job in jobs(plan, state, campaign) if campaign.eligible(plan, state, job)]
        fleet.tick(dict(schema_version=1, hosts=plan['hosts'], jobs=current, poll_seconds=plan['poll_seconds']), state, F)
        changed = campaign.retry_failed(plan, state)
        if changed:
            fleet.append(F / 'retries.jsonl', dict(time=time.time(), frames=changed))
        fleet.atomic(F / 'state.json', state)
        selected = [state['jobs'][row['name']]['status'] for row in state['frames'].values()]
        fleet.atomic(F / 'summary.json', dict(time=time.time(), frames=len(selected), capacity=10,
                     counts={s: selected.count(s) for s in set(selected)}, next_supervision_deadline=time.time() + 1800,
                     per_host={h: sum(row.get('host') == h and state['jobs'][row['name']]['status'] == 'running' for row in state['frames'].values()) for h in plan['hosts']}))
        if all(s == 'complete' for s in selected):
            command = read(plan['postprocess_command_file'])
            if command is not None and not (F / 'postprocess_complete.json').exists():
                with (F / 'postprocess.log').open('ab') as stream:
                    subprocess.run(command['command'], check=True, stdout=stream, stderr=subprocess.STDOUT)
                fleet.atomic(F / 'postprocess_complete.json', dict(time=time.time(), command=command))
            if (F / 'postprocess_complete.json').exists():
                fleet.atomic(F / 'campaign_complete.json', dict(time=time.time(), frames=len(selected))); return
        time.sleep(plan['poll_seconds'])


def watchdog(F):
    plan, V = guard(F); fleet, _, _ = modules(V); held = fleet.lock(F / 'watchdog.lock')
    if held is None:
        return
    fleet.atomic(F / 'watchdog.json', dict(process=fleet.identity(), started=time.time()))
    while not (F / 'campaign_complete.json').exists():
        guard(F)
        if not fleet.alive(read(F / 'controller.json', {}).get('process')):
            with (F / 'controller.log').open('ab') as stream:
                process = subprocess.Popen([sys.executable, __file__, '_dispatch', '--root', str(F)], stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            fleet.append(F / 'watchdog_checks.jsonl', dict(time=time.time(), started=fleet.identity(process.pid)))
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['prepare', 'preflight', 'reserve', 'adopt', 'launch', '_dispatch', '_watchdog', '_slots'])
    p.add_argument('--root', type=Path); p.add_argument('--campaign', type=Path); p.add_argument('--source-campaign', type=Path); p.add_argument('--frame', type=int)
    p.add_argument('--host-root', type=Path); p.add_argument('--expected-source-sha');p.add_argument('--source-proof',type=Path)
    p.add_argument('--force-host');p.add_argument('--force-gpu',type=int);p.add_argument('--resume-authorization',type=Path);a = p.parse_args()
    if a.expected_source_sha and sha(__file__) != a.expected_source_sha:
        raise ValueError('Fleet slot-probe source differs')
    if a.action == '_slots':
        print(json.dumps(probe_slots(a.host_root))); raise SystemExit(0)
    F = a.root.resolve()
    if a.action == 'prepare':
        print(json.dumps(prepare(F, a.campaign.resolve(), a.source_campaign.resolve())))
    elif a.action == 'preflight':
        print(json.dumps(preflight(F)))
    elif a.action == 'reserve':
        print(json.dumps(reserve(F, a.frame,a.source_proof,a.force_host,a.force_gpu,a.resume_authorization)))
    elif a.action == 'adopt':
        adopt(F); print(json.dumps(dict(adopted=True)))
    elif a.action == '_dispatch':
        dispatch(F)
    elif a.action == '_watchdog':
        watchdog(F)
    else:
        plan, V = guard(F); fleet, _, _ = modules(V)
        release=read(Path(plan['scientific_campaign'])/'release.json')
        if release.get('production_enabled')is not True or release.get('code_seal_sha256')!=plan['source_code_seal_sha256']:
            raise ValueError('Capped scientific release is not enabled')
        if read(F / 'state.json') is None:
            raise ValueError('Explicit identity-checked adoption required before launch')
        with (F / 'watchdog.log').open('ab') as stream:
            child = subprocess.Popen([sys.executable, __file__, '_watchdog', '--root', str(F)], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(watchdog_pid=child.pid)))
