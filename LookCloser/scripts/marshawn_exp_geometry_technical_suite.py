"""Exclusive-GPU zero-update probe and optional real Adam pause/resume smoke."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from marshawn_tuning_controller import gpu_snapshot,atomic


def main(a):
    root=a.root.resolve();scripts=Path(__file__).resolve().parent
    lock=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{a.gpu}.lock').open('a+')
    while True:
        locked=False
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
            gpu=gpu_snapshot()[a.gpu]
            if not gpu['processes'] and gpu['used_mib']<=128:break
        except BlockingIOError:
            pass
        if locked:fcntl.flock(lock,fcntl.LOCK_UN)
        if not a.wait_idle:raise RuntimeError('Requested GPU is occupied or reserved')
        atomic(root/f'technical_suite_{a.arm}.json',dict(arm=a.arm,gpu=a.gpu,phase='waiting_idle',pid=os.getpid(),time=time.time(),production=False))
        time.sleep(60)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    receipt=root/f'technical_suite_{a.arm}.json';commands=[]
    def run(argv,label):
        atomic(receipt,dict(arm=a.arm,gpu=a.gpu,phase=label,pid=os.getpid(),time=time.time(),commands=commands,production=False))
        with (root/f'{a.arm}_{label}.log').open('a')as stream:
            result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        commands.append(dict(label=label,argv=argv,returncode=result.returncode))
        if result.returncode and not (label=='pause' and result.returncode==143):
            raise RuntimeError(f'{label} failed: {result.returncode}')
    spec=root/f'technical_{a.arm}.json';manifest=Path(json.loads(spec.read_text())['prefix_candidate']['manifest'])
    probe=root/f'probe_{a.arm}'
    if not (probe/'probe.json').exists():
        run([sys.executable,str(scripts/'probe_marshawn_exp_geometry.py'),'--spec',str(spec),'--output',str(probe)],'probe')
    if not a.probe_only:
        out=Path(json.loads(spec.read_text())['output'])
        if not (out/'latest.json').exists():
            run([sys.executable,str(scripts/'train_marshawn_exp_geometry.py'),'--spec',str(spec),'--smoke-steps','32'],'pause')
        latest=json.loads((out/'latest.json').read_text())
        if latest['step']!=15:raise ValueError('Expected deliberate technical pause at15')
        before=root/f'{a.arm}_paused_step15.ckpt'
        if not before.exists():os.link(out/latest['checkpoint'],before)
        run([sys.executable,str(scripts/'train_marshawn_exp_geometry.py'),'--spec',str(spec),'--smoke-steps','32'],'resume')
        latest=json.loads((out/'latest.json').read_text());after=out/latest['checkpoint']
        run([sys.executable,str(scripts/'verify_marshawn_exp_geometry_smoke.py'),'--before',str(before),'--after',str(after),
            '--manifest',str(manifest),'--renders',str(out/'renders/step-000000031'),'--output',str(root/f'verification_{a.arm}.json')],'verify')
    atomic(receipt,dict(arm=a.arm,gpu=a.gpu,phase='probe_complete'if a.probe_only else'technical_complete',pid=os.getpid(),time=time.time(),commands=commands,production=False))
    print(json.dumps(dict(arm=a.arm,finished=True,probe_only=a.probe_only)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['P','PB'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--probe-only',action='store_true');p.add_argument('--wait-idle',action='store_true');main(p.parse_args())
