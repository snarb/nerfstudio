"""Sequential read-only103TRAIN/12DC replay of the completed matched64752 pair."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def main(c):
    root=c/'late_distortion_audit_helpers';code=c/'late_distortion_code';scripts=code/'LookCloser/scripts'
    sys.path.insert(0,str(scripts))
    from marshawn_tuning_controller import gpu_snapshot,identity,alive
    from marshawn_tuning_occupancy import sha
    manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('Audit helper changed')
    queries=[('clothing103',c/'independent_occupancy/queries/clothing_103_predeclared.json','e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'),('DC12',c/'independent_occupancy/DC_error12_15188/queries.json','08caff3fbdb24a811745761b3fa685c33fd8655be6342203cf851f4fcf1b6d00')]
    for _,q,h in queries:
        if sha(q)!=h:raise ValueError('Sealed diagnostic query changed')
    # Parent authorized <=10GiB sequential read-only overlap; producer untouched.
    gpu=5;branch=c/'exp_geometry_candidate_20261106';producer=branch/'run_P';progress=producer/'progress.json';log=branch/'controller/jobs/exp_expanded84_prefix003/worker.log'
    expected=identity(json.loads(progress.read_text())['pid']);assert expected and alive(expected)
    guard=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proc=None;started=time.monotonic();state=dict(pid=os.getpid(),phase='auditing',producer=expected,audits=[],optimizer_updates=0);write(root/'state.json',state)
    def check():
        row=json.loads(progress.read_text());devices=gpu_snapshot();current=identity(expected['pid'])
        if current is None or current['start']!=expected['start'] or not alive(expected):raise RuntimeError('Producer identity/aliveness changed')
        if time.time()-row['time']>180:raise RuntimeError('Producer heartbeat stale')
        processes=devices[gpu]['processes'];allowed={expected['pid']}|({proc.pid}if proc else set())
        if any(r['pid']not in allowed for r in processes):raise RuntimeError('Unplanned concurrent GPU process')
        raw=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        memory=sum(int(line.split(',')[1])for line in raw.splitlines()if proc is not None and int(line.split(',')[0])==proc.pid)
        if memory>10240:raise RuntimeError('Read-only audit process exceeds10GiB')
        if time.monotonic()-started>1200:raise TimeoutError('Pair audit deadline20minutes')
        with log.open('rb')as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
        if any(t in tail for t in ['CUDA out of memory','torch.OutOfMemoryError']):raise RuntimeError('Producer OOM evidence')
        result=dict(time=time.time(),progress=row,gpu=devices[gpu],candidate_mib=memory,producer_alive=True,producer_oom=False)
        with(root/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(result)+'\n')
        return result
    def interrupted(signum,frame):raise RuntimeError('Audit interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for arm in ['control001','treatment1']:
            folder=c/'independent_occupancy'/('late_distortion_'+arm+'_064752')
            for name,query,_ in queries:
                output=folder/(name+'_trace');healthfile=folder/(name+'_producer_health.json')
                if (output/'rays/diagnostic_summary.json').exists()and healthfile.exists()and json.loads(healthfile.read_text()).get('audit_environment_healthy'):continue
                before=check();health=dict(before_progress=before['progress'],before_gpu=before['gpu'],producer_identity=expected,authorized_overlap=True,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
                write(healthfile,health)
                command=[sys.executable,str(root/'audit_marshawn_tuning_checkpoint.py'),'trace','--data',str(c/'data'),'--output',str(output),'--queries',str(query),'--receipt',str(folder/'step-000064752_grid.json'),'--checkpoint',str(folder/'checkpoint.ckpt'),'--config',str(folder/'config.yml'),'--code-root',str(code),'--gpu',str(gpu),'--skip-wholegrid','--allow-readonly-overlap','--memory-cap-gib','10','--expected-sh','true','--expected-density','trunc_exp','--expected-projection','legacy']
                env=dict(os.environ,PYTHONPATH=str(code)+':'+str(scripts),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                with(folder/(name+'_trace.log')).open('w')as f:
                    proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=code/'LookCloser',start_new_session=True)
                    while proc.poll()is None:check();time.sleep(5)
                    result=proc.returncode
                after=check();health.update(after_progress=after['progress'],after_gpu=after['gpu'],producer_alive_after=True,returncode=result,audit_environment_healthy=result==0 and after['progress']['step']>=before['progress']['step']);write(healthfile,health);proc=None
                if not health['audit_environment_healthy']:raise RuntimeError('Audit needs attention; producer unchanged')
                state['audits'].append(dict(arm=arm,queries=name,output=str(output)));write(root/'state.json',state)
        state['phase']='traces_complete';write(root/'state.json',state)
    except BaseException as error:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if proc is not None and proc.poll()is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        state.update(phase='failed',error=repr(error),producer_not_modified=True);write(root/'state.json',state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args().campaign)
