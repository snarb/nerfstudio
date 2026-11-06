#!/usr/bin/env python3
"""Durable multi-host DAG dispatcher. No quality, iteration or wall-time cap.

Workers own stopping/plateau. Each host must already have this exact script and
inputs. An SSH transport failure is UNKNOWN, never proof a job stopped. Worker
receipts and advisory locks allow a dispatcher to die/restart without launching
a second optimizer. Commands are argv arrays; shell interpolation is never used.
"""
from __future__ import annotations
import argparse, base64, csv, fcntl, hashlib, json, math, os, re, shlex
import shutil, socket, subprocess, sys, time
from pathlib import Path

TERMINAL = {'complete', 'failed', 'skipped', 'blocked'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    try:
        stat = Path('/proc/%d/stat' % pid).read_text().split(') ')[1].split()
        return {'pid': pid, 'start': stat[19], 'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'state': stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    if not record: return False
    current = identity(record['pid'])
    return bool(current and current['state'] != 'Z' and all(current[k] == record[k] for k in ['pid', 'start', 'boot']))


def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stream = Path(path).open('a+')
    try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: stream.close(); return None
    return stream


def validate(plan):
    if plan.get('schema_version', 1) != 1 or not plan.get('hosts') or not plan.get('jobs'):
        raise ValueError('Expected schema v1 hosts and nonempty jobs')
    names = set()
    for host in plan['hosts'].values():
        for key in ['python', 'controller', 'root', 'gpus']:
            if key not in host: raise ValueError('Host missing ' + key)
        if not isinstance(host['gpus'], list) or not host['gpus']: raise ValueError('Host needs GPU slots')
    for job in plan['jobs']:
        name = job.get('name', '')
        if not re.fullmatch('[A-Za-z0-9_-]+', name) or name in names: raise ValueError('Unique safe job names required')
        names.add(name)
        if not isinstance(job.get('command'), list) or not job['command'] or not all(isinstance(s,str) for s in job['command']):
            raise ValueError('command must be nonempty argv strings')
        if job.get('host') and job['host'] not in plan['hosts']: raise ValueError('Unknown host')
        if not isinstance(job.get('env', {}), dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in job.get('env',{}).items()):
            raise ValueError('Environment keys and values must be strings')
        condition = job.get('condition')
        if condition and (condition.get('job') not in job.get('dependencies',[]) or condition.get('op') not in ('lt','le','gt','ge','eq') or not condition.get('path') or 'value' not in condition):
            raise ValueError('Condition must reference a declared dependency and supported comparison')
    graph = {j['name']: j.get('dependencies', []) for j in plan['jobs']}
    done = set()
    while len(done) != len(names):
        ready = {n for n, deps in graph.items() if n not in done and set(deps) <= done}
        if not ready: raise ValueError('Cyclic graph or missing dependency')
        done.update(ready)
    if not 0 < float(plan.get('poll_seconds', 30)) <= 60: raise ValueError('poll_seconds must be (0,60]')
    return plan


def gpu_snapshot():
    def query(kind, fields):
        p = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=15)
        if p.returncode: raise RuntimeError(p.stderr.strip())
        return list(csv.reader(p.stdout.splitlines(), skipinitialspace=True))
    devices = {int(r[0]): {'uuid': r[1], 'name': r[2], 'total_mib': int(r[3]), 'used_mib': int(r[4]), 'processes': []}
               for r in query('gpu', 'index,uuid,name,memory.total,memory.used')}
    by_uuid = {d['uuid']: d for d in devices.values()}
    for r in query('compute-apps', 'gpu_uuid,pid,process_name'):
        if r and r[0] in by_uuid: by_uuid[r[0]]['processes'].append({'pid':int(r[1]), 'name':r[2]})
    return devices


def probe(root):
    root = Path(root); jobs = {}
    for p in (root / 'jobs').glob('*/receipt.json'):
        item = read(p); item['supervisor_alive'] = alive(item.get('supervisor'))
        item['worker_alive'] = alive(item.get('worker'))
        jobs[p.parent.name] = item
    try: gpu, error = gpu_snapshot(), None
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc: gpu, error = {}, str(exc)
    return {'time':time.time(), 'host':socket.gethostname(), 'jobs':jobs, 'gpus':gpu, 'gpu_error':error,
            'free_bytes':shutil.disk_usage(next(p for p in [root,*root.parents] if p.exists())).free,
            'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def host_call(host, argv, *, timeout=25, input_text=None):
    command = [host['python'], host['controller']] + argv
    if host.get('address'):
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host['address'], shlex.join(command)]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, input=input_text)


def materialize(job, host, gpu):
    job = dict(job); job['gpu'] = gpu
    values = {'host_root':host['root'], 'job_root':str(Path(host['root'])/'jobs'/job['name'])}
    def subst(text):
        for key, value in values.items(): text = text.replace('{'+key+'}', value)
        return text
    job['command'] = [subst(s) for s in job['command']]
    if job.get('cwd'): job['cwd'] = subst(job['cwd'])
    job['env'] = {k:subst(v) for k,v in job.get('env',{}).items()}
    job['progress_files'] = [subst(s) for s in job.get('progress_files',[])]
    if job.get('metrics_file'): job['metrics_file'] = subst(job['metrics_file'])
    return job


def worker(root, spec):
    """One durable supervisor. Claim precedes subprocess, and is never erased."""
    root = Path(root); directory = root/'jobs'/spec['name']; directory.mkdir(parents=True,exist_ok=True)
    held = lock(directory/'worker.lock')
    if held is None: return 0
    receipt_path = directory/'receipt.json'
    if receipt_path.exists(): return 0  # including uncertain prior launches: never duplicate
    gpu = spec.get('gpu')
    gpu_lock = lock(Path('/tmp')/('lookcloser_tuning_gpu_%s_%s.lock' % (os.getuid(), gpu))) if gpu is not None else None
    if gpu is not None and gpu_lock is None: return 75
    device = None
    if gpu is not None:
        try:
            device = gpu_snapshot().get(int(gpu))
            if device is None or device['processes']: return 75
        except (OSError, RuntimeError, subprocess.SubprocessError): return 75
    data = {'name':spec['name'], 'status':'starting', 'spec_sha256':digest(spec), 'supervisor':identity(), 'worker':None,
            'gpu':gpu, 'gpu_uuid':device['uuid'] if device else None, 'started':time.time()}
    atomic(directory/'spec.json',spec); atomic(receipt_path,data)
    env = os.environ.copy(); env.update(spec.get('env',{}))
    # Linux limits each argv/environment string to 128KiB. Dependency histories
    # grow across DAG barriers, so they must never enter execve's environment.
    if 'TUNING_DEPENDENCIES_JSON' in env:
        dependencies = json.loads(env.pop('TUNING_DEPENDENCIES_JSON'))
        dependency_file = directory / 'dependencies.json'
        atomic(dependency_file, dependencies)
        env['TUNING_DEPENDENCIES_FILE'] = str(dependency_file.resolve())
        env['TUNING_DEPENDENCIES_SHA256'] = hashlib.sha256(dependency_file.read_bytes()).hexdigest()
    if device: env.update(CUDA_VISIBLE_DEVICES=device['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    try:
        with (directory/'worker.log').open('ab') as output:
            process = subprocess.Popen(spec['command'],cwd=spec.get('cwd'),env=env,stdin=subprocess.DEVNULL,
                                       stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            data.update(status='running', worker=identity(process.pid)); atomic(receipt_path,data)
            while True:
                code = process.poll(); checkpoint = {'time':time.time(),'worker_alive':code is None,'worker_pid':process.pid}
                for path in spec.get('progress_files',[]):
                    try: checkpoint.setdefault('progress',{})[path] = read(path)
                    except (ValueError,OSError): checkpoint.setdefault('progress_errors',[]).append(path)
                try: checkpoint['gpus'] = gpu_snapshot()
                except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as exc: checkpoint['gpu_error'] = str(exc)
                with (directory/'worker.log').open('rb') as stream:
                    stream.seek(0,2); stream.seek(max(0,stream.tell()-65536)); tail = stream.read().decode(errors='replace').lower()
                checkpoint['oom_evidence'] = 'out of memory' in tail or 'cudnn_status_alloc_failed' in tail
                checkpoint['free_bytes'] = shutil.disk_usage(directory).free
                append(directory/'checks.jsonl',checkpoint)
                data.update(heartbeat=checkpoint['time'], latest_check=checkpoint); atomic(receipt_path,data)
                if code is not None: break
                time.sleep(float(spec.get('poll_seconds',30)))
        data.update(status='complete' if code == 0 else 'failed',returncode=code,finished=time.time())
        if spec.get('metrics_file'):
            try: data['metrics'] = read(spec['metrics_file'])
            except (ValueError,OSError) as exc: data['metrics_error'] = str(exc)
        atomic(receipt_path,data)
        return code
    except Exception as exc:
        # If a child might be alive, keep UNKNOWN rather than auto-retrying it.
        data.update(status='unknown' if alive(data.get('worker')) else 'failed',error=repr(exc),finished=time.time())
        atomic(receipt_path,data); raise


def start_worker(root, spec):
    root = Path(root); directory = root/'jobs'/spec['name']; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'receipt.json').exists(): return
    payload_path = directory / ('launch-spec-' + digest(spec) + '.json')
    if payload_path.exists() and read(payload_path) != spec:
        raise ValueError('Immutable launch specification changed')
    if not payload_path.exists(): atomic(payload_path, spec)
    with (directory/'supervisor.log').open('ab') as stream:
        subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker','--root',str(root),'--spec-file',str(payload_path.resolve()),'--spec-sha256',digest(spec)],
                         stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def condition_pass(condition, states):
    if not condition: return True
    source = states[condition['job']]
    value = source
    for key in condition['path'].split('.'): value = value[key]
    expected = condition['value']; op = condition['op']
    return {'lt':lambda:value<expected,'le':lambda:value<=expected,'gt':lambda:value>expected,
            'ge':lambda:value>=expected,'eq':lambda:value==expected}[op]()


def tick(plan, state, root):
    snapshots = {}
    for key, host in plan['hosts'].items():
        try:
            result = host_call(host, ['_probe','--root',host['root']])
            if result.returncode: raise RuntimeError(result.stderr[-1000:])
            snapshots[key] = json.loads(result.stdout)
            if snapshots[key].get('controller_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError('Controller source differs on host '+key)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            snapshots[key] = {'error':str(exc)}
    jobs = state['jobs']
    for name,item in jobs.items():
        if item['status'] in TERMINAL or not item.get('host'): continue
        snap = snapshots[item['host']]
        if 'error' in snap: item.update(status='unknown',error=snap['error']); continue
        remote = snap['jobs'].get(name)
        if remote:
            if remote.get('spec_sha256') != item['spec_sha256']: raise RuntimeError('Remote job spec changed: '+name)
            status = remote['status']
            if status not in TERMINAL and not remote['supervisor_alive']: status = 'orphaned' if remote['worker_alive'] else 'failed'
            item.update(remote); item['status']=status
        elif item['status'] in ('dispatching','unknown'):
            # An absent receipt does not rule out a delayed detached launcher.
            # Retry only the same host/job/GPU, whose durable claim is idempotent.
            item.update(status='pending')
    reserved = {(j.get('host'),j.get('gpu')) for j in jobs.values() if j['status'] not in TERMINAL|{'pending'}}
    for job in plan['jobs']:
        item = jobs[job['name']]
        if item['status'] != 'pending': continue
        deps = [jobs[n]['status'] for n in job.get('dependencies',[])]
        if any(s in ('failed','blocked') for s in deps): item['status']='blocked'; continue
        if any(s not in ('complete','skipped') for s in deps): continue
        try:
            if not condition_pass(job.get('condition'),jobs): item['status']='skipped'; continue
        except (KeyError,TypeError,ValueError) as exc:
            item.update(status='blocked',error='Invalid dependency result: '+str(exc)); continue
        for key,host in plan['hosts'].items():
            if job.get('host') and job['host'] != key: continue
            if item.get('host') and item['host'] != key: continue
            snap=snapshots[key]
            if 'error' in snap or snap.get('gpu_error'): continue
            for gpu in host['gpus']:
                if job.get('gpu') is not None and job['gpu'] != gpu: continue
                if item.get('gpu') is not None and item['gpu'] != gpu: continue
                device=snap['gpus'].get(str(gpu),snap['gpus'].get(gpu))
                if (key,gpu) in reserved or not device or device['processes']: continue
                spec=materialize(job,host,gpu); spec['poll_seconds']=plan.get('poll_seconds',30)
                spec['env']['TUNING_DEPENDENCIES_JSON']=json.dumps({name:jobs[name].get('metrics') for name in job.get('dependencies',[])},sort_keys=True)
                item.update(status='dispatching',host=key,gpu=gpu,spec_sha256=digest(spec),dispatched=time.time())
                atomic(Path(root)/'state.json',state)  # intent BEFORE SSH side effect
                try:
                    r=host_call(host,['_start','--root',host['root'],'--spec-stdin','--spec-sha256',digest(spec)], input_text=json.dumps(spec))
                    if r.returncode: item.update(status='unknown',error=r.stderr[-1000:])
                except (OSError,subprocess.SubprocessError) as exc: item.update(status='unknown',error=str(exc))
                reserved.add((key,gpu)); break
            if item['status'] != 'pending': break
    state['updated']=time.time(); state['host_snapshots']=snapshots
    atomic(Path(root)/'state.json',state)
    append(Path(root)/'checks.jsonl',{'time':state['updated'],'jobs':{n:j['status'] for n,j in jobs.items()},'hosts':snapshots})
    return all(item['status'] in TERMINAL for item in jobs.values())


def prepare(root, manifest):
    plan=validate(read(manifest)); root=Path(root); root.mkdir(parents=True,exist_ok=True)
    frozen=root/'manifest.json'
    if frozen.exists() and digest(read(frozen)) != digest(plan): raise ValueError('Campaign manifest is immutable')
    if not frozen.exists(): atomic(frozen,plan)
    state=read(root/'state.json')
    if state is None:
        state={'manifest_sha256':digest(plan),'created':time.time(),'jobs':{j['name']:{'status':'pending'} for j in plan['jobs']}}
        atomic(root/'state.json',state)
    if state['manifest_sha256'] != digest(plan): raise ValueError('State/manifest mismatch')
    return plan,state


def dispatch(root):
    root=Path(root); held=lock(root/'controller.lock')
    if held is None: return 0
    plan,state=prepare(root,root/'manifest.json')
    atomic(root/'controller.json',{'process':identity(),'started':time.time()})
    while not tick(plan,state,root): time.sleep(float(plan.get('poll_seconds',30)))
    return 0 if all(j['status'] in ('complete','skipped') for j in state['jobs'].values()) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','dry-run','launch','resume','status','report','_dispatch','_probe','_start','_worker'])
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--manifest',type=Path); parser.add_argument('--spec64')
    parser.add_argument('--spec-file',type=Path); parser.add_argument('--spec-stdin',action='store_true'); parser.add_argument('--spec-sha256')
    args=parser.parse_args()
    if args.action=='_probe': print(json.dumps(probe(args.root))); return 0
    if args.action in ('_start','_worker'):
        if sum(bool(x) for x in (args.spec64,args.spec_file,args.spec_stdin)) != 1:
            raise ValueError('Exactly one specification input is required')
        spec=json.load(sys.stdin) if args.spec_stdin else read(args.spec_file) if args.spec_file else json.loads(base64.b64decode(args.spec64))
        if (args.spec_stdin or args.spec_file) and not args.spec_sha256:
            raise ValueError('File/stdin specifications require a digest')
        if args.spec_sha256 and digest(spec) != args.spec_sha256:
            raise ValueError('Specification digest mismatch')
        return worker(args.root,spec) if args.action=='_worker' else start_worker(args.root,spec)
    if args.action=='_dispatch': return dispatch(args.root)
    if args.action=='dry-run':
        plan=validate(read(args.manifest or args.root/'manifest.json'))
        print(json.dumps({'jobs':len(plan['jobs']),'hosts':plan['hosts'],'dependencies':{j['name']:j.get('dependencies',[]) for j in plan['jobs']}},indent=2)); return 0
    if args.action in ('prepare','launch','resume'):
        plan,state=prepare(args.root,args.manifest or args.root/'manifest.json')
        if args.action=='prepare': print(json.dumps({'prepared':str(args.root),'jobs':len(state['jobs'])})); return 0
        current=read(args.root/'controller.json',{})
        if alive(current.get('process')): print('Controller already alive'); return 0
        with (args.root/'controller.log').open('ab') as stream:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_dispatch','--root',str(args.root.resolve())],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'controller_pid':p.pid,'root':str(args.root)})); return 0
    state=read(args.root/'state.json',{})
    controller=read(args.root/'controller.json',{})
    print(json.dumps({'controller_alive':alive(controller.get('process')),'state':state},indent=2)); return 0

if __name__=='__main__':
    sys.exit(main())
