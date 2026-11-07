"""Exclusive future RAW source producer and source-ready DEV3 six-map queue.

Operational services only; frozen decoder/fitter mathematics remain untouched.
No model, prepared, admission, frequency-ready or dispatch marker is published.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shlex,shutil,socket,subprocess,sys,time,traceback

SIX=['AA','AB','AC','AD','BA','BB']


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid())
    with tmp.open('w') as stream:json.dump(v,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(p)


def append(p,v):
    with Path(p).open('a') as stream:stream.write(json.dumps(v)+'\n');stream.flush()


def identity(pid=None):
    pid=os.getpid() if pid is None else int(pid)
    try:
        part=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()
        return dict(pid=pid,start=part[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host=socket.gethostname(),state=part[0])
    except (OSError,ValueError):return None


def alive(saved):
    if not saved:return False
    now=identity(saved['pid'])
    return bool(now and now['state']!='Z' and all(now[k]==saved[k] for k in ('pid','start','boot','host')))


def lock(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);stream=p.open('a+')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:stream.close();return None
    return stream


def guard(c):
    for name,digest in c['guard_files'].items():
        if sha(name)!=digest:raise ValueError('Immutable source/config guard changed: '+name)


def source_guard_files(V):
    code=V/'code';scripts=code/'LookCloser/scripts';B=Path('/fsx/tmp/marshawn_001660_hd')
    names=[V/'code_seal.json',scripts/'marshawn_video_prep_sources.py',scripts/'marshawn_actor_coordinates.py',
        scripts/'marshawn_radial_photometry.py',scripts/'prepare_marshawn_color_variants.py',
        code/'nerfstudio/data/utils/colmap_parsing_utils.py',
        B/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json',
        B/'report/shared_radial1660_train_PRECHECK/fit_k4.json',
        Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py')]
    R=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
    names.extend([R/'images.txt',R/'cameras.txt'])
    return {str(p):sha(p) for p in names}


def configure(V):
    V=Path(V).resolve();ops=V/'operations/future_sources';ops.mkdir(parents=True,exist_ok=True)
    frames=list(range(1675,2020,5));guards=source_guard_files(V)
    tf=read('/fsx/tmp/marshawn_001660_hd/color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json')
    cameras=sorted(r['physical_camera'] for r in tf['frames'])
    if len(cameras)!=66 or len(set(cameras))!=66:raise ValueError('ALL66 calibrated sources required')
    c=dict(schema=1,root=str(V),frames=frames,cameras=cameras,workers=4,reserve_bytes=256*2**30,frame_budget_bytes=8*2**30,
        decode_python='/home/pavlo/.conda/envs/nerfstudio/bin/python',dev3_python='/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',
        dev3_target='ubuntu@dev3',dev3_gpu_index=0,dev3_gpu_uuid='GPU-ab838312-8799-795c-7df2-b085cb3db092',
        guard_files=guards,service_sha256=sha(__file__),reserved_frequency_cameras=SIX)
    path=ops/'config.json'
    if path.exists() and read(path)!=c:raise ValueError('Future service configuration is immutable')
    atomic(path,c)
    ownership=dict(schema=1,kind='EXCLUSIVE_FROZEN_RGB_NATIVE_DECODE_V1',owner='future_sources_v1',frame_ids=frames,
        cameras=cameras,root=str(V),config=str(path),config_sha256=sha(path),service_sha256=sha(__file__),
        source_ready_pattern=str(V/'sources/{frame}/source_ready.json'),
        consumer_contract='V3 waits source_ready and verifies all66 records; never invokes decoder for owned frame IDs',
        no_model_prepared_or_dispatch_state=True)
    atomic(ops/'ownership.json',ownership);return path


def verify_sources(c,frame):
    """Every RAW/RGB/native byte and camera binding checked before source_ready."""
    V=Path(c['root']);records=[];tf_sha=c['guard_files']['/fsx/tmp/marshawn_001660_hd/color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json']
    photo_sha=c['guard_files']['/fsx/tmp/marshawn_001660_hd/report/shared_radial1660_train_PRECHECK/fit_k4.json']
    script_sha=c['guard_files'][str(V/'code/LookCloser/scripts/marshawn_video_prep_sources.py')]
    rawroot=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local') if frame<=1900 else Path('/fsx/oregon/gta_marshawn_2_1_tiff16_f1901-2020_20261105/by_local')
    for camera in c['cameras']:
        p=V/'sources'/str(frame)/camera/'receipt.json';r=read(p)
        expected=rawroot/camera/('%s.%06d.tif'%(camera,frame-1252))
        if r['frame']!=frame or r['camera']!=camera or Path(r['raw']).resolve()!=expected.resolve():raise ValueError('Wrong RAW frame/camera binding')
        if r['transforms_sha256']!=tf_sha or r['photometry_sha256']!=photo_sha or r['script_sha256']!=script_sha:raise ValueError('Frozen photometry/calibration/decoder changed')
        for key,field in [('raw','raw_sha256'),('image','image_sha256'),('raw_native_image','raw_native_image_sha256')]:
            if sha(r[key])!=r[field]:raise ValueError('Source image/RAW bytes changed: '+camera+' '+key)
        if r['raw_native_raster']!=[8192,4320]:raise ValueError('Native raster differs')
        records.append(dict(camera=camera,receipt=str(p),receipt_sha256=sha(p),raw=r['raw'],raw_sha256=r['raw_sha256'],
            image=r['image'],image_sha256=r['image_sha256'],raw_native_image=r['raw_native_image'],raw_native_image_sha256=r['raw_native_image_sha256']))
    return records


def decode(c,config):
    V=Path(c['root']);ops=V/'operations/future_sources';held=lock(ops/'decode.lock')
    if held is None:return
    atomic(ops/'decode_identity.json',identity());guard(c)
    for frame in c['frames']:
        ready=V/'sources'/str(frame)/'source_ready.json'
        if ready.exists():
            r=read(ready)
            if r['config_sha256']!=sha(config) or r['frame']!=frame or r['cameras']!=c['cameras']:raise ValueError('Source-ready ownership changed')
            continue
        guard(c)
        while shutil.disk_usage(V).free<c['reserve_bytes']+c['frame_budget_bytes']:
            atomic(ops/'decode_progress.json',dict(frame=frame,status='waiting_disk_reserve',free_bytes=shutil.disk_usage(V).free,
                required_bytes=c['reserve_bytes']+c['frame_budget_bytes'],time=time.time(),controller=identity()));time.sleep(30)
        # V3 honors ownership.json; additionally reject an already-running foreign
        # decoder targeting this same source tree/frame before invoking unlocked code.
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:argv=proc.read_bytes().decode().split('\0')
            except (OSError,UnicodeError):continue
            if str(V/'code/LookCloser/scripts/marshawn_video_prep_sources.py') in argv and str(V/'sources') in argv and str(frame) in argv:
                raise RuntimeError('Foreign source decoder already owns frame '+str(frame))
        command=[c['decode_python'],str(V/'code/LookCloser/scripts/marshawn_video_prep_sources.py'),'--frames',str(frame),
            '--cameras']+c['cameras']+['--output',str(V/'sources'),'--raw-native','--workers','4']
        env=dict(os.environ,PYTHONPATH=str(V/'code')+':'+str(V/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (ops/('%06d.log'%frame)).open('ab') as stream:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(held.fileno(),))
            atomic(ops/'decode_child.json',dict(frame=frame,child=identity(child.pid),started=time.time()))
            while child.poll() is None:
                free=shutil.disk_usage(V).free
                atomic(ops/'decode_progress.json',dict(frame=frame,status='decoding',controller=identity(),child=identity(child.pid),free_bytes=free,time=time.time()))
                if free<c['reserve_bytes']:
                    child.terminate();child.wait();raise RuntimeError('Disk reserve reached; partial frame retained for same-frame retry')
                time.sleep(10)
        if child.returncode:raise RuntimeError('Frozen decoder failed '+str(child.returncode))
        guard(c);records=verify_sources(c,frame)
        atomic(ready,dict(schema=1,kind='ALL66_FROZEN_RGB_NATIVE_SOURCE_READY_V1',frame=frame,cameras=c['cameras'],records=records,
            config_sha256=sha(config),guard_files=c['guard_files'],decoder_workers=4,source_fps=120,output_fps=24,
            source_verified_all66=True,no_NeRF_state_loaded=True,no_model_prepared_or_dispatch_state=True,verified_at=time.time()))
        atomic(ops/'decode_progress.json',dict(frame=frame,status='source_ready',ready_sha256=sha(ready),time=time.time()))
    atomic(ops/'decode_complete.json',dict(frames=c['frames'],time=time.time()))


def acl(path):
    subprocess.run(['setfacl','-m','u:1000:rwx,d:u:1000:rwx',str(path)],check=True,capture_output=True)


def dev3_specs(c,frame):
    V=Path(c['root']);source=read(V/'sources'/str(frame)/'source_ready.json')
    if source['frame']!=frame or source['config_sha256']!=sha(V/'operations/future_sources/config.json') or source['source_verified_all66'] is not True:
        raise ValueError('Unqualified source-ready marker')
    rows={r['camera']:r for r in source['records']};data=V/'frames'/('%06d'%frame)/'data';ops=V/'operations/future_frequency'/('%06d'%frame)
    for p in [data,data/'images',data/'lookcloser_frequencies',ops]:p.mkdir(parents=True,exist_ok=True);acl(p)
    specs=[]
    for camera in SIX:
        r=rows[camera]
        if sha(r['receipt'])!=r['receipt_sha256'] or sha(r['image'])!=r['image_sha256']:raise ValueError('Ready source changed')
        image=data/'images'/('train_'+camera+'.png')
        if image.exists():
            if sha(image)!=r['image_sha256']:raise ValueError('Future data RGB differs; never overwrite')
        else:image.symlink_to(Path(r['image']))
        spec=dict(frame=frame,camera=camera,image=str(image),image_sha256=r['image_sha256'],
            output=str(data/'lookcloser_frequencies'/('train_'+camera+'.pt')),source_code=str(V/'code'),
            preprocess_source_sha256=sha(V/'code/nerfstudio/scripts/lookcloser_preprocess.py'))
        path=ops/(camera+'.json')
        if path.exists() and read(path)!=spec:raise ValueError('Future six-map spec changed')
        atomic(path,spec);specs.append(path)
    return specs


def dev3_remote(c,frame):
    V=Path(c['root']);ops=V/'operations/future_frequency'/('%06d'%frame);held=lock(ops/'worker.lock')
    if held is None:return 75
    gpu=lock('/tmp/lookcloser_tuning_gpu_%d_0.lock'%os.getuid())
    if gpu is None:return 75
    actual=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual!=c['dev3_gpu_uuid']:raise ValueError('DEV3 physical UUID changed')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    if actual in busy.splitlines():return 75
    guard(c);env=dict(os.environ,CUDA_VISIBLE_DEVICES=actual,CUDA_DEVICE_ORDER='PCI_BUS_ID',LOOKCLOSER_GPU_LEASE_FD=str(gpu.fileno()),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    for camera in SIX:
        path=ops/(camera+'.json');atomic(ops/'worker_identity.json',dict(controller=identity(),frame=frame,camera=camera,time=time.time()))
        with (ops/(camera+'.log')).open('ab') as stream:
            child=subprocess.Popen([sys.executable,str(V/'code/LookCloser/scripts/marshawn_video_frequency.py'),'--spec',str(path)],
                env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(gpu.fileno(),held.fileno()))
            atomic(ops/'worker_identity.json',dict(controller=identity(),child=identity(child.pid),frame=frame,camera=camera,time=time.time()))
            while child.poll() is None:
                atomic(ops/'worker_progress.json',dict(controller=identity(),child=identity(child.pid),frame=frame,camera=camera,time=time.time()));time.sleep(10)
            if child.returncode:return child.returncode
    # Recheck all map bytes/recipe through the immutable fitter validator on CPU.
    import importlib.util
    p=V/'code/LookCloser/scripts/marshawn_video_frequency.py';s=importlib.util.spec_from_file_location('_future_fitter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    records=[]
    for camera in SIX:
        spec=read(ops/(camera+'.json'));m.validate_spec(spec)
        if not m.validate_existing(spec):raise ValueError('Six-map postvalidation failed')
        records.append(dict(camera=camera,map_sha256=sha(spec['output']),metadata_sha256=sha(Path(spec['output']).with_suffix('.json')),image_sha256=spec['image_sha256']))
    atomic(ops/'six_maps_verified.json',dict(frame=frame,cameras=SIX,records=records,source_ready_sha256=sha(V/'sources'/str(frame)/'source_ready.json'),
        canonical_preprocess_sha256=sha(V/'code/nerfstudio/scripts/lookcloser_preprocess.py'),prepared_publication_allowed=False,time=time.time()))
    return 0


def dev3(c,config):
    V=Path(c['root']);ops=V/'operations/future_frequency';ops.mkdir(exist_ok=True);held=lock(ops/'controller.lock')
    if held is None:return
    atomic(ops/'controller_identity.json',identity())
    for frame in c['frames']:
        while not (V/'sources'/str(frame)/'source_ready.json').exists():
            atomic(ops/'progress.json',dict(frame=frame,status='waiting_source_ready',time=time.time()));time.sleep(10)
        guard(c);specs=dev3_specs(c,frame);fops=specs[0].parent
        while not (fops/'six_maps_verified.json').exists():
            # Authoritative remote probe checks both parent and inherited-lock
            # child identities before any retry; SSH failure remains UNKNOWN.
            code="""import json,sys,subprocess; from pathlib import Path
sys.path.insert(0,sys.argv[1]);import marshawn_future_services as f
p=Path(sys.argv[2]);r=f.read(p) if p.exists() else {};oom={}
for camera in f.SIX:
 log=p.parent/(camera+'.log')
 if log.exists():
  with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().lower()
  oom[camera]=b'out of memory' in tail or b'cuda error: memory' in tail
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
print(json.dumps({'controller_alive':f.alive(r.get('controller')),'child_alive':f.alive(r.get('child')),'progress':r,'GPU_memory':gpu.stdout.strip(),'GPU_probe_returncode':gpu.returncode,'OOM_evidence':oom}))"""
            command=[c['dev3_python'],'-c',code,str(Path(__file__).parent),str(fops/'worker_identity.json')]
            probe=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',c['dev3_target'],shlex.join(command)],capture_output=True,text=True,timeout=30)
            if probe.returncode:
                atomic(ops/'progress.json',dict(frame=frame,status='SSH_UNKNOWN_no_retry',error=probe.stderr[-500:],time=time.time()));time.sleep(15);continue
            observed=json.loads(probe.stdout)
            if not observed['controller_alive'] and not observed['child_alive']:
                argv=[c['dev3_python'],str(Path(__file__).resolve()),'remote','--config',str(config),'--frame',str(frame)]
                launcher="""import subprocess,sys,json; from pathlib import Path
with Path(sys.argv[1]).open('ab') as o:p=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=o,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid}))"""
                cmd=[c['dev3_python'],'-c',launcher,str(fops/'supervisor.log')]+argv
                launched=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',c['dev3_target'],shlex.join(cmd)],capture_output=True,text=True,timeout=30)
                append(ops/'launches.jsonl',dict(frame=frame,returncode=launched.returncode,stdout=launched.stdout[-1000:],stderr=launched.stderr[-1000:],time=time.time()))
            status=dict(frame=frame,status='six_maps_active_or_waiting_lease',remote=observed,time=time.time())
            atomic(ops/'progress.json',status);append(ops/'checks.jsonl',status);time.sleep(15)
    atomic(ops/'complete.json',dict(frames=c['frames'],maps=len(c['frames'])*6,time=time.time()))


def watchdog(c,config,mode):
    V=Path(c['root']);ops=V/('operations/future_sources' if mode=='decode' else 'operations/future_frequency');ops.mkdir(exist_ok=True)
    held=lock(ops/'watchdog.lock')
    if held is None:return
    atomic(ops/'watchdog_identity.json',identity())
    complete=ops/('decode_complete.json' if mode=='decode' else 'complete.json')
    identity_path=ops/('decode_identity.json' if mode=='decode' else 'controller_identity.json')
    while not complete.exists():
        guard(c);current=read(identity_path) if identity_path.exists() else None
        child=read(ops/'decode_child.json').get('child') if mode=='decode' and (ops/'decode_child.json').exists() else None
        if not alive(current) and not alive(child):
            with (ops/'controller.log').open('ab') as stream:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),mode,'--config',str(config)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            append(ops/'watchdog_checks.jsonl',dict(action='controller_started',process=identity(p.pid),time=time.time()))
        else:append(ops/'watchdog_checks.jsonl',dict(action='controller_or_inherited_child_alive',controller=current,child=child,time=time.time()))
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['configure','decode','dev3','remote','watch_decode','watch_dev3'])
    p.add_argument('--root',type=Path);p.add_argument('--config',type=Path);p.add_argument('--frame',type=int);a=p.parse_args()
    if a.action=='configure':print(configure(a.root));raise SystemExit()
    c=read(a.config)
    if sha(__file__)!=c['service_sha256']:raise ValueError('Operational service code changed')
    guard(c)
    if a.action=='decode':decode(c,a.config)
    elif a.action=='dev3':dev3(c,a.config)
    elif a.action=='remote':raise SystemExit(dev3_remote(c,a.frame))
    else:watchdog(c,a.config,'decode' if a.action=='watch_decode' else 'dev3')
