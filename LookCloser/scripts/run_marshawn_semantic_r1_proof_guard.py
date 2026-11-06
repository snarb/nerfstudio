"""Bounded root-reviewed R1 read-only proof; no optimizer and no producer mutation."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def identity(pid):
    text=Path(f'/proc/{pid}/stat').read_text();parts=text[text.rfind(')')+2:].split()
    return dict(pid=int(pid),start=parts[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=parts[0])
def gpu_rows(gpu):
    cmd=['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits']
    rows=[]
    for line in subprocess.check_output(cmd,text=True).splitlines():
        p,m=line.split(',');rows.append(dict(pid=int(p.strip()),mib=int(m.strip())))
    return rows


def main(root):
    root=Path(root);policy=json.loads((root/'technical_policy.json').read_text());release=json.loads((root/'technical_release.json').read_text())
    if release.get('approved_for_technical_overlap')is not True or release.get('policy_sha256')!=sha(root/'technical_policy.json'):raise ValueError('Technical overlap awaits exact root release')
    if policy['gpu']!=7 or policy['max_seconds']>300 or policy['max_process_mib']>10240 or policy['production_training']is not False:raise ValueError('Unreviewed resource scope')
    for name,digest in policy['artifacts_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Reviewed technical artifact changed: '+name)
    inventory=json.loads((root/'source_manifest.json').read_text())
    for name,digest in inventory['files_sha256'].items():
        if sha(root/'code'/name)!=digest:raise ValueError('Frozen candidate code changed: '+name)
    lock=open(f"/tmp/lookcloser_readonly_audit_{os.getuid()}_{policy['gpu']}.lock",'a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Deliberately do not acquire or impersonate the producer's global GPU lock.
    expected=policy['producer_identity'];progress=Path(policy['producer_progress']);log=Path(policy['producer_log']);began=time.monotonic();checks=[];owned=set();proc=None
    def check(phase):
        current=identity(expected['pid'])
        if any(current[k]!=expected[k]for k in ('pid','start','boot'))or current['state']=='Z':raise RuntimeError('Producer identity changed or died')
        state=json.loads(progress.read_text());rows=gpu_rows(policy['gpu'])
        if time.time()-state['time']>180:raise RuntimeError('Producer progress stale >180s')
        unexpected=[r for r in rows if r['pid']not in owned|{expected['pid']}]
        if unexpected:raise RuntimeError('Unplanned GPU overlap '+str(unexpected))
        memory=sum(r['mib']for r in rows if r['pid']in owned)
        if memory>policy['max_process_mib']:raise RuntimeError('Candidate process VRAM exceeds release')
        if time.monotonic()-began>policy['max_seconds']:raise TimeoutError('Technical overlap deadline')
        with log.open('rb')as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
        oom=any(v in tail for v in ('CUDA out of memory','torch.OutOfMemoryError'))
        if oom:raise RuntimeError('Producer OOM evidence')
        row=dict(time=time.time(),phase=phase,producer=current,producer_step=state['step'],producer_progress_age=time.time()-state['time'],gpu_rows=rows,candidate_mib=memory,producer_oom_evidence=oom)
        checks.append(row)
        with (root/'technical_resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        return row
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(policy['gpu']),PYTHONPATH=str(root/'code')+':'+str(root/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    python=policy['python'];scripts=root/'code/LookCloser/scripts'
    def run(name,args,expected_code=0):
        nonlocal proc
        check('before_'+name)
        with (root/f'{name}.log').open('w')as out:
            proc=subprocess.Popen([python,*map(str,args)],stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=root/'code/LookCloser')
            owned.add(proc.pid)
            while proc.poll()is None:
                check(name);time.sleep(5)
            result=proc.returncode;check('after_'+name);proc=None
            if result!=expected_code:raise RuntimeError(f'{name} exit {result}, expected {expected_code}')
    def interrupted(signum, frame):
        raise RuntimeError(f'Proof supervisor interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        first=check('initial')
        run('R1_native_proof',[root/'proof_helpers/probe_marshawn_semantic_r1.py','--spec',root/'spec.json','--archive',Path(policy['source_archive']),'--output',root/'R1_native_proof.json','--gpu',str(policy['gpu'])])
        while json.loads(progress.read_text())['step']<=first['producer_step']:
            check('waiting_next_producer_heartbeat');time.sleep(5)
        final=check('complete')
        if final['producer_step']<=first['producer_step']:raise RuntimeError('No producer progress during technical gate')
        result=dict(passed=True,policy_sha256=sha(root/'technical_policy.json'),technical_release_sha256=sha(root/'technical_release.json'),elapsed_seconds=time.monotonic()-began,producer_before=first,producer_after=final,max_candidate_process_mib=max(r['candidate_mib']for r in checks),production_training=False,shared_producer_global_lock_untouched=True,performance_comparison=False,receipts_sha256={n:sha(root/n)for n in ['R1_native_proof.json']})
        atomic(root/'technical_complete.json',result)
    except BaseException as error:
        # Repeated termination must not interrupt cleanup and orphan our child.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if proc is not None and proc.poll()is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        atomic(root/'technical_failed.json',dict(error=repr(error),elapsed_seconds=time.monotonic()-began,production_modified=False,checks=len(checks)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
