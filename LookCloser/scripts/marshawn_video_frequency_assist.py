#!/usr/bin/env python3
"""Operational DEV3 six-camera frequency assist and verified silver publication.

Lives OUTSIDE the production code seal. Fitting mathematics and map validation
come only from the sealed campaign frequency worker and canonical preprocessor.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,shlex,socket,subprocess,sys,time,traceback

CAMERAS=['AA','AB','AC','AD','BA','BB']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with temp.open('w') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def append(path,value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value)+'\n')


def identity(pid=None):
    pid=pid or os.getpid();parts=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
    return dict(pid=pid,start=parts[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host=socket.gethostname())


def alive(expected):
    try:return expected and identity(expected['pid'])==expected and Path('/proc/%d/stat'%expected['pid']).read_text().split(') ',1)[1].split()[0]!='Z'
    except (OSError,KeyError):return False


def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a+')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:stream.close();return None
    return stream


def guard(config):
    if config.get('authorized') is not True:raise ValueError('Operational assist has not been released by campaign owner')
    if config['cameras']!=CAMERAS:raise ValueError('Only the six assigned physical cameras may fit on DEV3')
    for path,digest in config['files_sha256'].items():
        if sha(path)!=digest:raise ValueError('Released operational/fitter source changed: '+path)
    seal=Path(config['root'])/'code_seal.json'
    if sha(seal)!=config['code_seal_sha256']:raise ValueError('Production source seal changed')


def worker_module(config):
    path=Path(config['root'])/'code/LookCloser/scripts/marshawn_video_frequency.py'
    spec=importlib.util.spec_from_file_location('_sealed_video_frequency_assist',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def paths(config,frame):
    root=Path(config['root']);data=root/'frames'/f'{frame:06d}'/'data'
    operational=Path(config['operations'])/'frames'/f'{frame:06d}';operational.mkdir(parents=True,exist_ok=True)
    if os.getuid()==config['olive_uid']:
        subprocess.run(['setfacl','-m','u:1000:rwx,d:u:1000:rwx',str(operational)],check=True,capture_output=True)
    return data,operational


def specs(config,frame):
    data,operational=paths(config,frame);ready=read(data/'ready.json');rows={r['camera']:r for r in ready['records']}
    frequencies=data/'lookcloser_frequencies';frequencies.mkdir(exist_ok=True)
    if os.getuid()==config['olive_uid']:
        subprocess.run(['setfacl','-m','u:1000:rwx,d:u:1000:rwx',str(frequencies)],check=True,capture_output=True)
    camera_rows={r['physical_camera']:r for r in read(data/'transforms.json')['frames']};result=[]
    for camera in CAMERAS:
        row=rows[camera];image=data/row['image'];stem=Path(camera_rows[camera]['file_path']).stem
        spec=dict(frame=frame,camera=camera,image=str(image),image_sha256=row['rgb_sha256'],
            output=str(data/'lookcloser_frequencies'/(stem+'.pt')),source_code=str(Path(config['root'])/'code'),
            preprocess_source_sha256=config['preprocess_source_sha256'])
        destination=operational/('spec_'+camera+'.json')
        if destination.exists() and read(destination)!=spec:raise ValueError('Frame-local frequency spec changed')
        if not destination.exists():atomic(destination,spec)
        result.append((destination,spec))
    return result


def verify_maps(config,frame):
    module=worker_module(config);complete=[]
    for _,spec in specs(config,frame):
        module.validate_spec(spec)
        if not module.validate_existing(spec):return False
        metadata=read(Path(spec['output']).with_suffix('.json'))
        if frame!=1660 and any(metadata.get(k)!=spec[k] for k in ['frame','camera','preprocess_source_sha256']):
            raise ValueError('Frequency map lacks exact frame/camera/canonical-fitter provenance')
        complete.append(dict(camera=spec['camera'],image_sha256=spec['image_sha256'],
            map_sha256=sha(spec['output']),metadata_sha256=sha(Path(spec['output']).with_suffix('.json'))))
    _,operational=paths(config,frame)
    atomic(operational/'six_maps_verified.json',dict(frame=frame,cameras=CAMERAS,records=complete,
        canonical_preprocess_sha256=config['preprocess_source_sha256'],verified_at=time.time(),no_learned_state_transferred=True))
    return True


def probe(config,frame):
    _,operational=paths(config,frame);path=operational/'remote_progress.json'
    old=read(path) if path.exists() else {}
    controller=alive(old.get('controller'));worker=alive(old.get('worker'))
    memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],
        capture_output=True,text=True,timeout=15)
    oom={}
    for camera in CAMERAS:
        log=operational/(camera+'.log')
        if log.exists():
            with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().lower()
            oom[camera]=b'out of memory' in tail or b'cuda error: memory' in tail
    return dict(status='running' if controller or worker else 'proved_stopped',controller_alive=bool(controller),
        worker_alive=bool(worker),observed_host=socket.gethostname(),time=time.time(),progress=old,
        GPU_memory_evidence=memory.stdout.strip(),GPU_probe_returncode=memory.returncode,OOM_evidence=oom)


def remote_frame(config,frame):
    guard(config);_,operational=paths(config,frame);held=lock(operational/'remote_assist.lock')
    if held is None:return 75
    gpu=lock('/tmp/lookcloser_tuning_gpu_%d_%d.lock'%(os.getuid(),config['dev3_gpu_index']))
    if gpu is None:
        atomic(operational/'remote_progress.json',dict(status='GPU_lease_busy',controller=identity(),time=time.time()));return 75
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    actual={int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.splitlines()}
    if actual.get(config['dev3_gpu_index'])!=config['dev3_gpu_uuid']:raise ValueError('DEV3 physical UUID/index changed')
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
    if any(config['dev3_gpu_uuid'] in line for line in compute.splitlines()):
        atomic(operational/'remote_progress.json',dict(status='GPU_occupied',controller=identity(),time=time.time(),evidence=compute.strip()));return 75
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=config['dev3_gpu_uuid'],OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    code=Path(config['root'])/'code';env['PYTHONPATH']=str(code)+':'+str(code/'LookCloser/scripts')
    progress=dict(frame=frame,controller=identity(),status='checking_same_frame_maps',time=time.time(),GPU_uuid=config['dev3_gpu_uuid'])
    atomic(operational/'remote_progress.json',progress)
    for spec_path,spec in specs(config,frame):
        # A new process per image gives each fit a fresh allocator/model/RNG.
        command=[sys.executable,str(code/'LookCloser/scripts/marshawn_video_frequency.py'),'--spec',str(spec_path)]
        with (operational/(spec['camera']+'.log')).open('ab') as stream:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                pass_fds=(gpu.fileno(),held.fileno()))
            progress.update(status='fitting_or_verifying',camera=spec['camera'],worker=identity(child.pid),time=time.time())
            atomic(operational/'remote_progress.json',progress)
            while child.poll() is None:
                progress['checked_at']=time.time();atomic(operational/'remote_progress.json',progress);time.sleep(10)
            if child.returncode:
                progress.update(status='worker_failed',returncode=child.returncode,time=time.time())
                atomic(operational/'remote_progress.json',progress);return child.returncode
    if not verify_maps(config,frame):raise ValueError('DEV3 returned without all six source-bound maps')
    progress.update(status='six_maps_complete',worker=None,time=time.time());atomic(operational/'remote_progress.json',progress)
    return 0


def ssh(config,argv):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',config['dev3_target'],shlex.join(argv)],
        capture_output=True,text=True,timeout=45)


def daemon(config,config_path):
    guard(config);root=Path(config['root']);operations=Path(config['operations']);held=lock(operations/'controller.lock')
    if held is None:return 0
    atomic(operations/'controller_identity.json',identity());last_hour=0
    while True:
        now=time.time();frames=[r['frame'] for r in read(root/'source_inventory.json')['frames']];pending=[];done=[];state={}
        for frame in frames:
            froot=root/'frames'/f'{frame:06d}'
            if (froot/'dispatch_ready.json').exists():done.append(frame);continue
            data,operational=paths(config,frame)
            marker=froot/'central_prepared.json'
            if not marker.exists():pending.append(frame);continue
            central=read(marker)
            if central.get('train')!=66 or central['prepared_ready_sha256']!=sha(data/'prepared_ready.json'):
                raise ValueError('Central prepared marker is stale/incomplete')
            try:
                if verify_maps(config,frame):
                    code=root/'code/LookCloser/scripts';command=[sys.executable,str(code/'marshawn_video_stage.py'),'push',
                        '--source-campaign',str(root),'--frame',str(frame),'--target',config['silver_target'],
                        '--target-root',config['silver_root'],'--target-python',config['silver_python'],'--host-key',config['silver_host_key']]
                    with (operational/'silver_stage.log').open('ab') as stream:
                        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            atomic(operations/'status.json',dict(status='staging_verified_six_maps',frame=frame,worker=identity(child.pid),checked_at=time.time()));time.sleep(10)
                        if child.returncode:raise RuntimeError('Silver publication failed: '+str(child.returncode))
                    if not (froot/'dispatch_ready.json').exists():raise RuntimeError('Stage returned without dispatch marker')
                    done.append(frame);continue
                command=[config['dev3_python'],str(Path(__file__).resolve()),'_probe','--config',str(config_path),'--frame',str(frame)]
                observed=ssh(config,command)
                if observed.returncode:raise RuntimeError('SSH UNKNOWN; no duplicate launch authorized: '+observed.stderr[-300:])
                remote= json.loads(observed.stdout)
                state=dict(status='adopting_DEV3_worker' if remote['status']=='running' else 'launching_DEV3_worker',frame=frame,remote=remote)
                if remote['status']=='proved_stopped':
                    launch=ssh(config,[config['dev3_python'],str(Path(__file__).resolve()),'_start_remote','--config',str(config_path),'--frame',str(frame)])
                    if launch.returncode:raise RuntimeError('SSH launch outcome UNKNOWN; re-probe before retry: '+launch.stderr[-300:])
                    state['launch']=json.loads(launch.stdout)
                # Chronological assist queue; preparation proceeds independently.
                break
            except Exception as error:
                state=dict(status='retry_after_proof',frame=frame,error=str(error),traceback=traceback.format_exc());break
        state.update(time=now,controller=identity(),dispatched=len(done),total=len(frames),awaiting_preparation=pending)
        atomic(operations/'status.json',state)
        if now-last_hour>=3600:
            append(operations/'hourly_checks.jsonl',state);last_hour=now
        if len(done)==len(frames):atomic(operations/'complete.json',state);return 0
        time.sleep(30)


def watchdog(config,config_path):
    guard(config);operations=Path(config['operations']);held=lock(operations/'watchdog.lock')
    if held is None:return 0
    atomic(operations/'watchdog_identity.json',identity())
    while not (operations/'complete.json').exists():
        saved=operations/'controller_identity.json';current=read(saved) if saved.exists() else None
        if not alive(current):
            with (operations/'controller.log').open('ab') as stream:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_daemon','--config',str(config_path)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            append(operations/'watchdog_checks.jsonl',dict(action='controller_started',worker=identity(child.pid),time=time.time()))
        else:append(operations/'watchdog_checks.jsonl',dict(action='controller_alive',worker=current,time=time.time()))
        time.sleep(30)
    return 0


def configure(a):
    root=a.root.resolve();ops=root/'operations/frequency_assist';ops.mkdir(parents=True,exist_ok=True)
    code=root/'code';worker=code/'LookCloser/scripts/marshawn_video_frequency.py';fitter=code/'nerfstudio/scripts/lookcloser_preprocess.py'
    config=dict(schema=1,authorized=False,root=str(root),operations=str(ops),cameras=CAMERAS,
        olive_uid=os.getuid(),
        dev3_target='ubuntu@dev3',dev3_python='/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',dev3_gpu_index=0,
        dev3_gpu_uuid='GPU-ab838312-8799-795c-7df2-b085cb3db092',silver_target='ubuntu@silver-sable',
        silver_root='/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107',
        silver_python='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python',silver_host_key=a.silver_host_key,
        preprocess_source_sha256=sha(fitter),code_seal_sha256=sha(root/'code_seal.json'),
        files_sha256={str(Path(__file__).resolve()):sha(__file__),str(worker):sha(worker),str(fitter):sha(fitter)})
    shared=ops/'frames';shared.mkdir(exist_ok=True)
    subprocess.run(['setfacl','-m','u:1000:rwx,d:u:1000:rwx',str(shared)],check=True,capture_output=True)
    atomic(ops/'config.json',config);print(ops/'config.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['configure','launch','check','_daemon','_watchdog','_probe','_remote','_start_remote'])
    p.add_argument('--root',type=Path);p.add_argument('--config',type=Path);p.add_argument('--frame',type=int);p.add_argument('--silver-host-key',default='silver')
    a=p.parse_args()
    if a.action=='configure':configure(a);return 0
    config=read(a.config)
    if a.action=='_probe':print(json.dumps(probe(config,a.frame)));return 0
    guard(config)
    if a.action=='check':print(json.dumps(dict(valid=True,authorized=True,CUDA_started=False)));return 0
    if a.action=='_remote':return remote_frame(config,a.frame)
    if a.action=='_daemon':return daemon(config,a.config)
    if a.action=='_watchdog':return watchdog(config,a.config)
    if a.action=='_start_remote':
        _,ops=paths(config,a.frame)
        with (ops/'remote_controller.log').open('ab') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_remote','--config',str(a.config),'--frame',str(a.frame)],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(started=identity(child.pid))));return 0
    if a.action=='launch':
        with (Path(config['operations'])/'watchdog.log').open('ab') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_watchdog','--config',str(a.config)],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(Path(config['operations'])/'launch.json',dict(process=identity(child.pid),command=sys.argv,time=time.time()));print(child.pid);return 0


if __name__=='__main__':sys.exit(main())
