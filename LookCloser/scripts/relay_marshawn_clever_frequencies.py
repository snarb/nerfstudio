#!/usr/bin/env python3
"""Lease one central camera at a time, fit it on clever, validate and publish its map.

Clever has a private filesystem: advisory locks are held HERE for the complete
remote fit/validation/publish interval. Only the active camera is reserved.
Never overwrites a complete map pair. Run with validated local nerfstudio Python.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import time

CAMERAS = ('MD','NA','NB','NC','ND','NE','OA','OD')
FITTER_SHA = '6ef90cbbbd6f277886c3aaa2112d4b151f08c325c4ae5fb7b0b63da3d3b75bd7'
STAGE = '/home/ubuntu/lookcloser_transfer/marshawn_001660_stage'
RUNTIME = '/home/ubuntu/lookcloser_transfer/marshawn_001660_runtime/new_py311/bin/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,starttime=text[19],state=text[0])


def validate_pair(tensor_path,metadata_path,frame,source,expected_script=FITTER_SHA):
    import torch
    metadata=read(metadata_path)
    required=dict(ssim_threshold=.95,train_steps_per_level=1000,eval_patch_batch_size=1024,
        min_res=16,max_res=8192,n_levels=16,patch_size=8,stride=8,value_type='scalar_resolution',
        source_image_sha256=sha(source),fit_script_sha256=expected_script,image=Path(frame['file_path']).name)
    if any(metadata.get(key)!=value for key,value in required.items()):
        bad={key:dict(expected=value,actual=metadata.get(key)) for key,value in required.items() if metadata.get(key)!=value}
        raise ValueError(f'Frequency provenance mismatch: {bad}')
    if metadata.get('image_shape') != [frame['h'],frame['w']] or metadata.get('crop_coords_xywh') is not None:
        raise ValueError('frequency raster/crop contract differs from the prepared image')
    values=torch.load(tensor_path,map_location='cpu',weights_only=True)
    shape=((frame['h']-8)//8+1,(frame['w']-8)//8+1)
    if not isinstance(values,torch.Tensor) or tuple(values.shape)!=shape or not values.is_floating_point():
        raise ValueError('frequency tensor shape/dtype mismatch')
    if not bool(torch.isfinite(values).all()) or not bool(((values>=16-1e-4)&(values<=8192+.01)).all()):
        raise ValueError('frequency tensor contains nonfinite/out-of-schedule values')
    return dict(tensor_sha256=sha(tensor_path),metadata_sha256=sha(metadata_path),source_image_sha256=required['source_image_sha256'],
                fit_script_sha256=expected_script,shape=list(shape),host=metadata.get('host'))


def publish_pair(incoming,frequency_dir,stem,receipt):
    """Caller must hold the central queue lock until both renames and receipt finish."""
    frequency_dir=Path(frequency_dir);incoming=Path(incoming)
    target=frequency_dir/(stem+'.pt');metadata=target.with_suffix('.json')
    if target.exists() and metadata.exists():raise FileExistsError('complete map pair already exists; never overwrite')
    # Preserve crash leftovers outside the frequency inventory before committing.
    for path in (target,metadata):
        if path.exists():shutil.move(str(path),str(incoming/(path.name+'.previous_partial')))
    pending=[]
    for suffix in ('.json','.pt'):
        src=incoming/(stem+suffix);tmp=frequency_dir/(stem+f'.relay_{os.getpid()}.pending'+suffix)
        shutil.copyfile(src,tmp)
        with tmp.open('rb') as stream:os.fsync(stream.fileno())
        if sha(tmp)!=sha(src):raise IOError('copy digest differs before atomic publication')
        pending.append((tmp,frequency_dir/(stem+suffix)))
    for tmp,destination in pending:os.replace(tmp,destination)
    write(incoming/'publication.json',dict(published=time.time(),**receipt))


REMOTE_COMMON = '''import json,os,signal,subprocess,time,hashlib
from pathlib import Path
request=json.loads(input())
stage=Path(request['stage'])
def identity(pid):
 try:
  parts=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'starttime':parts[19],'state':parts[0]}
 except FileNotFoundError:return None
def alive(worker):
 current=identity(worker['pid'])
 return current is not None and current['starttime']==worker['starttime'] and current['state']!='Z'
def gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 return {'raw':raw,'free_mib':int(raw.split(',')[1].strip())}
def apps():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
'''

REMOTE_START = REMOTE_COMMON + '''
benchmark=json.loads((stage/'frequency_chunk_benchmark/results.json').read_text())
fitter=stage/'code/fit_marshawn_frequency_queue.py'
assert benchmark['exact_parity_pass'] and benchmark['fitter_sha256']==request['fitter_sha256']
assert hashlib.sha256(fitter.read_bytes()).hexdigest()==request['fitter_sha256']
assert benchmark['python_executable']==request['runtime']
active=stage/'frequency_relay_active.json'
if active.exists() and alive(json.loads(active.read_text())):raise RuntimeError('an owned relay worker is already alive; refuse duplicate')
resource=gpu()
if resource['free_mib']<19*1024:raise RuntimeError('GPU free below19GiB launch gate')
source=stage/'data'/request['frame']['file_path']
assert hashlib.sha256(source.read_bytes()).hexdigest()==request['source_sha256']
job=stage/'frequency_relay_jobs'/request['job_id'];job.mkdir(parents=True,exist_ok=False)
data=job/'data';data.mkdir();(data/'images').symlink_to(stage/'data/images',target_is_directory=True)
metadata=request['metadata'];metadata['train_filenames']=[request['frame']['file_path']]
(data/'transforms.json').write_text(json.dumps(metadata,indent=2)+'\\n')
(job/'request.json').write_text(json.dumps(request,indent=2)+'\\n')
environment=dict(os.environ);environment['PYTHONPATH']=str(stage/'code')+':'+str(stage/'code/LookCloser/scripts')
with (job/'fitter_stdout.log').open('w') as log:
 process=subprocess.Popen([request['runtime'],str(fitter),'--data',str(data)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
worker=identity(process.pid);worker.update(job=str(job),job_id=request['job_id'],camera=request['camera'],started=time.time(),gpu_before=resource,apps_before=apps(),runtime_torch=benchmark['torch'],runtime_cuda=benchmark['cuda'],fitter_sha256=request['fitter_sha256'])
(job/'worker.json').write_text(json.dumps(worker,indent=2)+'\\n');active.write_text(json.dumps(worker,indent=2)+'\\n')
print(json.dumps(worker))
'''

REMOTE_PREFLIGHT = REMOTE_COMMON + '''
benchmark=json.loads((stage/'frequency_chunk_benchmark/results.json').read_text())
assert benchmark['exact_parity_pass'] and benchmark['fitter_sha256']==request['fitter_sha256']
assert hashlib.sha256((stage/'code/fit_marshawn_frequency_queue.py').read_bytes()).hexdigest()==request['fitter_sha256']
assert benchmark['python_executable']==request['runtime']
resource=gpu()
if resource['free_mib']<19*1024:raise RuntimeError('GPU free below19GiB pre-lease gate')
print(json.dumps({'gpu':resource,'apps':apps(),'runtime_torch':benchmark['torch'],'runtime_cuda':benchmark['cuda'],'parity_verified':True}))
'''

REMOTE_STATUS = REMOTE_COMMON + '''
worker=request['worker'];job=Path(worker['job']);log=job/'fitter_stdout.log'
tail=log.read_text(errors='replace')[-16000:] if log.exists() else ''
errors=[line[-500:] for line in tail.splitlines() if any(token in line.lower() for token in ['out of memory','cuda error','illegal memory access'])]
print(json.dumps({'alive':alive(worker),'gpu':gpu(),'apps':apps(),'oom_evidence':errors[-5:],'log_tail':tail[-1000:]}))
'''

REMOTE_STOP = REMOTE_COMMON + '''
worker=request['worker'];active=stage/'frequency_relay_active.json'
assert active.exists()
record=json.loads(active.read_text())
assert all(record[key]==worker[key] for key in ['pid','starttime','job_id'])
if alive(worker):
 assert os.getpgid(worker['pid'])==worker['pid']
 os.killpg(worker['pid'],signal.SIGTERM)
 deadline=time.monotonic()+10
 while alive(worker) and time.monotonic()<deadline:time.sleep(.2)
 if alive(worker):os.killpg(worker['pid'],signal.SIGKILL)
print(json.dumps({'stopped':not alive(worker),'worker':worker}))
'''


def remote(host,code,payload):
    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(code)],
        input=json.dumps(payload)+'\n',capture_output=True,text=True,timeout=60)
    if process.returncode:raise RuntimeError(f'SSH operation failed ({process.returncode}): {process.stderr[-2500:]}')
    return json.loads(process.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/data'))
    parser.add_argument('--root',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/clever_frequency_relay'))
    parser.add_argument('--host',default='ubuntu@clever-shadow')
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--max-fit-seconds',type=float,default=600)
    parser.add_argument('--slow-fit-seconds',type=float,default=240)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();metadata=read(args.data/'transforms.json')
    frames={f['physical_camera']:f for f in metadata['frames'] if f['file_path'] in metadata['train_filenames']}
    if any(camera not in frames for camera in CAMERAS):raise ValueError('requested cameras are not all train views')
    if args.dry_run:
        print(json.dumps(dict(cameras=CAMERAS,available=[c for c in CAMERAS if not (args.data/'lookcloser_frequencies'/(Path(frames[c]['file_path']).stem+'.pt')).exists()],fitter_sha256=FITTER_SHA,stage=STAGE,runtime=RUNTIME)));return
    os.umask(0);args.root.mkdir(parents=True,exist_ok=True)
    controller_lock=open(args.root/'controller.lock','a+b')
    try:fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('a durable root relay controller already owns this campaign')
    lease=dict(**identity(os.getpid()),host=socket.gethostname(),created=time.time(),helper_sha256=sha(__file__),cameras=CAMERAS,
        fitter_sha256=FITTER_SHA,data=str(args.data),transforms_sha256=sha(args.data/'transforms.json'),status='starting',active_camera=None)
    write(args.root/'lease.json',lease)
    def event(kind,**data):
        record=dict(time=time.time(),event=kind,root_pid=os.getpid(),**data)
        with (args.root/'events.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'relay signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    worker=None;held=None;published=[]
    try:
        for camera in CAMERAS:
            frame=frames[camera];stem=Path(frame['file_path']).stem;directory=args.data/'lookcloser_frequencies'
            target=directory/(stem+'.pt');sidecar=target.with_suffix('.json')
            if not (target.exists() and sidecar.exists()):
                event('prelease_runtime_gpu_gate',camera=camera,result=remote(args.host,REMOTE_PREFLIGHT,dict(stage=STAGE,runtime=RUNTIME,fitter_sha256=FITTER_SHA)))
            held=open(directory/(stem+'.lock'),'a+b')
            try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                held.close();held=None;event('skip_busy',camera=camera);continue
            if target.exists() and sidecar.exists():
                validate_pair(target,sidecar,frame,args.data/frame['file_path'])
                event('skip_complete',camera=camera);held.close();held=None;continue
            job_id=f'{camera}_{os.getpid()}_{time.time_ns()}'
            lease.update(status='leased',active_camera=camera,active_job=job_id);write(args.root/'lease.json',lease)
            request=dict(stage=STAGE,runtime=RUNTIME,fitter_sha256=FITTER_SHA,camera=camera,job_id=job_id,
                frame=frame,metadata=metadata,source_sha256=sha(args.data/frame['file_path']))
            worker=remote(args.host,REMOTE_START,request)
            write(args.root/'remote_worker.json',worker);event('worker_started',camera=camera,worker=worker)
            began=time.monotonic()
            while True:
                status=remote(args.host,REMOTE_STATUS,dict(stage=STAGE,worker=worker))
                event('supervision',camera=camera,elapsed_seconds=time.monotonic()-began,**status)
                write(args.root/'state.json',dict(camera=camera,worker=worker,status=status,time=time.time(),published=published))
                if status['alive'] and (status['gpu']['free_mib']<16*1024 or status['oom_evidence'] or time.monotonic()-began>args.max_fit_seconds):
                    stopped=remote(args.host,REMOTE_STOP,dict(stage=STAGE,worker=worker));event('worker_stopped_resource_gate',camera=camera,stop=stopped)
                    raise RuntimeError('own worker stopped: GPU reserve, OOM evidence or fit deadline gate')
                if not status['alive']:break
                time.sleep(args.poll_seconds)
            incoming=args.root/'incoming'/job_id;incoming.mkdir(parents=True,exist_ok=False)
            command=['rsync','-az']
            for suffix in ('.pt','.json'):command+=['--include=/'+stem+suffix]
            command+=['--exclude=*',args.host+':'+worker['job']+'/data/lookcloser_frequencies/',str(incoming)+'/']
            subprocess.run(command,check=True,timeout=120)
            validation=validate_pair(incoming/(stem+'.pt'),incoming/(stem+'.json'),frame,args.data/frame['file_path'])
            elapsed=time.monotonic()-began
            receipt=dict(camera=camera,elapsed_seconds=elapsed,worker=worker,root_identity=identity(os.getpid()),**validation)
            publish_pair(incoming,directory,stem,receipt)
            published.append(camera);event('published',**receipt)
            held.close();held=None;worker=None
            lease.update(active_camera=None,status='between_jobs',published=published);write(args.root/'lease.json',lease)
            if elapsed>args.slow_fit_seconds:
                lease['status']='stopped_slow_host';write(args.root/'lease.json',lease)
                event('stopped_slow_host',elapsed_seconds=elapsed,threshold=args.slow_fit_seconds);return
        lease.update(status='done',active_camera=None,published=published);write(args.root/'lease.json',lease);event('done',published=published)
    except BaseException as error:
        event('failed',error=repr(error),worker=worker)
        lease.update(status='failed',error=repr(error));write(args.root/'lease.json',lease)
        if worker is not None:
            try:event('failure_cleanup',result=remote(args.host,REMOTE_STOP,dict(stage=STAGE,worker=worker)))
            except BaseException as stop_error:event('supervision_interrupted',error=repr(stop_error),last_worker=worker)
        raise
    finally:
        if held is not None:held.close()
        controller_lock.close()


if __name__=='__main__':main()
