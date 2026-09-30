"""Fit fresh per-frame frequency maps on local or dev3 GPU, with durable receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from prepare_luster_video import environment, write, SCRIPTS


def remote(host, code, *args):
    command='python3 -c '+shlex.quote(code)+' '+shlex.join([str(a) for a in args])
    return subprocess.check_output(['ssh','-o','BatchMode=yes',host,command],text=True)


def validate_adoption(existing, frame, worker):
    """Only explicitly adopt an unfinished claim whose local owner has exited."""
    if existing.get('frame') != frame or existing.get('worker') != worker or existing.get('complete'):
        raise ValueError('Cannot adopt a completed or differently owned frequency claim')
    pid = int(existing['pid'])
    if pid <= 0:
        raise ValueError('Invalid frequency queue PID')
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return
    if state != 'Z':
        raise ValueError('Frequency claim still has a live owner')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--remote',action='store_true')
    p.add_argument('--remote-root',default='/fsx/tmp/luster/lookcloser_video_000470_000529_20260930')
    p.add_argument('--host',default='ubuntu@dev3');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--adopt',help='A first frame whose frequency worker is already running')
    args=p.parse_args();name='dev3' if args.remote else 'local';queue=args.root/'frequency_queue'
    queue.mkdir(exist_ok=True);claims=queue/'claims';claims.mkdir(exist_ok=True)
    frames=json.loads((args.root/'manifest.json').read_text())['frames']
    def claim():
        with (queue/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for frame in ([args.adopt] if args.adopt else frames):
                data=args.root/'frames'/frame/'data'
                if not (data/'audit_preprocessing.json').exists() or (data/'frequency_complete.json').exists():continue
                claim_path=claims/f'{frame}.json';previous=None
                if claim_path.exists():
                    if frame != args.adopt:continue
                    previous=json.loads(claim_path.read_text())
                    validate_adoption(previous,frame,name)
                record=dict(frame=frame,worker=name,pid=os.getpid(),time=time.time())
                if previous is not None:record['adopted_claim']=previous
                write(claim_path,record)
                return frame
        return None
    while True:
        frame=claim()
        if frame is None:
            if args.adopt:args.adopt=None;continue
            if all((args.root/'frames'/f/'data/frequency_complete.json').exists() for f in frames):break
            write(queue/f'{name}.json',dict(phase='waiting_for_prepared_frame',pid=os.getpid(),time=time.time()))
            time.sleep(30);continue
        data=args.root/'frames'/frame/'data';logdir=args.root/'frames'/frame/'logs';logdir.mkdir(exist_ok=True)
        adopting=frame==args.adopt;args.adopt=None
        start=time.time();process=None
        if args.remote:
            target=Path(args.remote_root)/'frequency_work'/frame
            if not adopting:
                remote(args.host,"from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)",target)
                subprocess.run(['rsync','-a',str(data/'images'),str(data/'transforms.json'),f'{args.host}:{target}/'],check=True)
                code='''import os,sys,subprocess,json
from pathlib import Path
base=Path(sys.argv[1]);data=Path(sys.argv[2]);repo=base/'code'
env=os.environ.copy();env.update(PYTHONPATH=str(repo)+':'+str(repo/'LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_HOME='/usr/local/cuda')
env['PATH']='/home/ubuntu/anaconda3/envs/nerfstudio/bin:/usr/local/cuda/bin:'+env['PATH']
with (data/'frequency_stdout.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/anaconda3/envs/nerfstudio/bin/python',str(repo/'LookCloser/scripts/prepare_luster_frequencies.py'),str(data),'--workers',sys.argv[3]],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(data/'worker.json').write_text(json.dumps(dict(pid=p.pid,workers=int(sys.argv[3]))))
'''
                remote(args.host,code,args.remote_root,target,args.workers)
        elif not adopting:
            log=(logdir/'frequencies_queue.log').open('x')
            process=subprocess.Popen([sys.executable,str(SCRIPTS/'prepare_luster_frequencies.py'),str(data),'--workers',str(args.workers)],env=environment(),stdout=log,stderr=subprocess.STDOUT)
        with (queue/f'{name}_supervision.jsonl').open('a') as journal:
            while True:
                if args.remote:
                    code='''import json,os,sys,subprocess
from pathlib import Path
d=Path(sys.argv[1]);pid=json.loads((d/'worker.json').read_text())['pid']
try: alive=Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
except FileNotFoundError: alive=False
p=d/'frequency_progress.json';log=d/'frequency_stdout.log'
with log.open('rb') as f:f.seek(max(0,log.stat().st_size-6000));tail=f.read().decode(errors='replace')
print(json.dumps(dict(pid=pid,alive=alive,complete=(d/'frequency_complete.json').exists(),progress=json.loads(p.read_text()) if p.exists() else None,oom='out of memory' in tail.lower(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True),tail=tail[-1500:] if not alive else None)))
'''
                    status=json.loads(remote(args.host,code,target))
                else:
                    progress=data/'frequency_progress.json'
                    if process is not None:
                        alive=process.poll() is None
                    else:
                        controller=logdir/'frequencies_w12/status.json'
                        existing=json.loads(controller.read_text());alive=existing['exit'] is None
                    status=dict(alive=alive,complete=(data/'frequency_complete.json').exists(),
                                progress=json.loads(progress.read_text()) if progress.exists() else None,
                                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True))
                    logfile=logdir/'frequencies_queue.log' if process is not None else logdir/'frequencies_w12/stdout.log'
                    with logfile.open('rb') as stream:
                        stream.seek(max(0,logfile.stat().st_size-6000));tail=stream.read().decode(errors='replace')
                    status['oom']='out of memory' in tail.lower()
                status.update(frame=frame,queue_pid=os.getpid(),time=time.time(),seconds=time.time()-start)
                write(queue/f'{name}.json',status);journal.write(json.dumps(status)+'\n');journal.flush()
                if status['complete']:break
                if not status['alive'] or status.get('oom'):raise RuntimeError(f'Frequency worker failed: {status}')
                time.sleep(30)
        if args.remote:
            for item in ['lookcloser_frequencies','frequency_complete.json']:
                subprocess.run(['rsync','-a',f'{args.host}:{target}/{item}',str(data)+'/'],check=True)
        # Per-image receipts bind the RGB and map bytes; the full data audit
        # follows after the common temporal AABB is frozen.
        write(claims/f'{frame}.json',dict(frame=frame,worker=name,pid=os.getpid(),complete=True,time=time.time()))
        print(json.dumps(dict(frame=frame,worker=name,complete=True,seconds=time.time()-start)),flush=True)
    write(queue/f'{name}_complete.json',dict(time=time.time(),frames=frames))


if __name__=='__main__':main()
