"""Root-released bounded technical-only cold coarse-step .003125 test on an idle locked GPU."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
import hashlib

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def rows(gpu):
    text=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return [dict(pid=int(line.split(',')[0]),mib=int(line.split(',')[1]))for line in text.splitlines()]

def main(root):
    policy=json.loads((root/'technical_policy.json').read_text());release=json.loads((root/'technical_release.json').read_text())
    if release.get('approved_for_technical')is not True or release.get('policy_sha256')!=sha(root/'technical_policy.json'):raise ValueError('Exact root technical release required')
    if policy['production_training']is not False or policy['max_seconds']>900 or policy['max_process_mib']>12288:raise ValueError('Unreviewed technical scope')
    for name,digest in policy['artifacts_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Reviewed artifact changed: '+name)
    for name,digest in json.loads((root/'source_manifest.json').read_text())['files_sha256'].items():
        if sha(root/'code'/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if (root/'technical_complete.json').exists():return
    if (root/'technical_failed.json').exists():raise ValueError('Preserve failed generation; separate review required')
    gpu=policy['gpu'];waiting=time.monotonic();lease=None
    while lease is None:
        handle=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock').open('a+')
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if rows(gpu):handle.close()
            else:lease=handle;break
        except BlockingIOError:handle.close()
        write(root/'technical_dispatch.json',dict(waiting_for_idle_global_lock=True,pid=os.getpid(),time=time.time(),gpu=gpu))
        if time.monotonic()-waiting>policy['max_wait_seconds']:raise TimeoutError('GPU remained busy; no technical worker launched')
        time.sleep(30)
    uuid=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if uuid!=policy['gpu_uuid']:raise ValueError('Reviewed GPU identity differs')
    proc=None;owned=set();checks=[];started=time.monotonic();scripts=root/'code/LookCloser/scripts'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root/'code')+':'+str(scripts),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def check(phase):
        current=rows(gpu);foreign=[r for r in current if r['pid']not in owned]
        if foreign:raise RuntimeError('Unplanned GPU process while exclusive lease held')
        memory=sum(r['mib']for r in current)
        if memory>policy['max_process_mib']:raise RuntimeError('Technical process memory cap exceeded')
        if time.monotonic()-started>policy['max_seconds']:raise TimeoutError('Technical deadline')
        record=dict(time=time.time(),phase=phase,gpu=gpu,gpu_uuid=uuid,processes=current,total_owned_mib=memory,exclusive_global_lock=True)
        checks.append(record)
        with(root/'technical_resource_checks.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    def run(name,args,expected=0):
        nonlocal proc
        check('before_'+name)
        with(root/(name+'.log')).open('w')as f:
            proc=subprocess.Popen([policy['python'],*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=root/'code/LookCloser',start_new_session=True);owned.add(proc.pid)
            while proc.poll()is None:check(name);time.sleep(5)
            code=proc.returncode;check('after_'+name);proc=None
        if code!=expected:raise RuntimeError(f'{name} exit{code}; expected{expected}')
    def interrupted(signum,frame):raise RuntimeError('Technical supervisor interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        check('initial')
        for label,start in [('warmup',0),('ARM',4096)]:
            spec=root/f'technical_{label}.json';run_dir=root/f'technical_{label}';command=[scripts/'train_marshawn_semantic_admission.py','--spec',spec,'--smoke-steps',str(start+32)]
            run(label+'_before',command,143)
            before=run_dir/'checkpoints'/f'step-{start+15:09d}.ckpt';saved=run_dir/'technical_before.ckpt';os.link(before,saved)
            run(label+'_after',command)
            run(label+'_verify',[scripts/'verify_marshawn_semantic_smoke.py','--before',saved,'--after',run_dir/'checkpoints'/f'step-{start+31:09d}.ckpt','--run',run_dir,'--output',root/f'verification_{label}.json'])
        check('complete')
        write(root/'technical_complete.json',dict(passed=True,policy_sha256=sha(root/'technical_policy.json'),release_sha256=sha(root/'technical_release.json'),elapsed_seconds=time.monotonic()-started,max_process_mib=max(r['total_owned_mib']for r in checks),GPU=uuid,exclusive_global_lock_held_across_both_pause_resume_paths=True,production_started=False,quality_claim=False,receipts_sha256={name:sha(root/name)for name in ['inherited_proofs/native_probe_probe.json','verification_warmup.json','verification_ARM.json']}))
    except BaseException as error:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if proc is not None and proc.poll()is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        write(root/'technical_failed.json',dict(error=repr(error),elapsed_seconds=time.monotonic()-started,other_producers_modified=False));raise
    finally:lease.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
