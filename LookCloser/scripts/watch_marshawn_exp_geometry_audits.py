"""Durable60-second15188 gate watcher and sequential read-only clothing audits."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(args):
    c=args.campaign.resolve();a=c/'independent_occupancy';branch=c/'exp_geometry_candidate_20261106';code=branch/'code';scripts=code/'LookCloser/scripts'
    sys.path.insert(0,str(scripts));from marshawn_tuning_controller import gpu_snapshot,identity,alive
    lock=(branch/'clothing_103_watcher.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    state_path=branch/'clothing_103_watcher.json';state=dict(pid=os.getpid(),poll_seconds=60,phase='waiting',step=15188,preserved={},audits=[])
    if state_path.exists():state=json.loads(state_path.read_text());state['pid']=os.getpid()
    write(state_path,state)
    if state['phase']=='report_complete':return
    while len(state['preserved'])<2:
        for arm in ('P','PB'):
            if arm in state['preserved']:continue
            run=branch/('run_'+arm);cp=run/'checkpoints/step-000015188.ckpt';receipt=run/'occupancy/step-000015188_grid.json';raster=run/'renders/step-000015188/metrics.json'
            if not all(p.exists()for p in [cp,receipt,raster]):continue
            folder=a/('exp_geometry_'+arm+'_015188');folder.mkdir(exist_ok=True);target=folder/'checkpoint.ckpt'
            if not target.exists():os.link(cp,target)
            for src in [run/'worker/lookcloser/frozen/config.yml',receipt,receipt.with_suffix('.npz')]:shutil.copyfile(src,folder/src.name)
            shutil.copytree(run/'renders/step-000015188',folder/'renders',dirs_exist_ok=True)
            archive=dict(step=15188,source_checkpoint=str(cp),checkpoint_sha256=sha(target),preserved_hardlink=True,
                         original_spec_sha256=sha(branch/('spec_'+arm+'.json')),protected_at=time.time())
            write(folder/'archive_receipt.json',archive);state['preserved'][arm]=dict(folder=str(folder),**archive);write(state_path,state)
        if len(state['preserved'])<2:
            state['last_poll']=time.time();state['producer_steps']={arm:json.loads((branch/('run_'+arm)/'progress.json').read_text())['step']for arm in ('P','PB') if (branch/('run_'+arm)/'progress.json').exists()};write(state_path,state);time.sleep(60)
    state['phase']='auditing';write(state_path,state)
    query=branch/'clothing_103_predeclared.json';expected=json.loads(query.read_text());assert len(expected['pixels'])==103
    assert sha(query)=='e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'
    requests=[('exp_reference','P',a/'structural_exp_hplus1_015188',c/'structural_extension_code')]
    requests += [(arm,arm,Path(state['preserved'][arm]['folder']),code)for arm in ('P','PB')]
    for name,producer_arm,folder,frozen in requests:
        trace=folder/'clothing103_trace';health_path=folder/'clothing103_producer_health.json'
        if (trace/'rays/diagnostic_summary.json').exists():state['audits'].append(dict(name=name,reused_complete_trace=True));write(state_path,state);continue
        global_guard=None;process=None;before={};producer=None
        # Arms may start far apart. Borrow either still-running producer's GPU,
        # or exclusively claim an idle device if both have already completed.
        while True:
            devices=gpu_snapshot();gpu=None
            for candidate_arm in (producer_arm,'PB'if producer_arm=='P'else'P'):
                candidate=branch/('run_'+candidate_arm);progress=json.loads((candidate/'progress.json').read_text())
                candidate_process=identity(progress['pid']);job='exp_expanded84_prefix003'+('_band001'if candidate_arm=='PB'else'')
                receipt=json.loads((branch/'controller/jobs'/job/'receipt.json').read_text());candidate_gpu=int(receipt['gpu'])
                if candidate_process and alive(candidate_process)and any(x['pid']==progress['pid']for x in devices[candidate_gpu]['processes']):
                    gpu=candidate_gpu;before=progress;process=candidate_process;producer=candidate;break
            if gpu is None:
                for index,device in devices.items():
                    if device['processes']or device['used_mib']>128:continue
                    candidate_guard=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{index}.lock').open('a+')
                    try:fcntl.flock(candidate_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:candidate_guard.close();continue
                    if gpu_snapshot()[index]['processes']:candidate_guard.close();continue
                    global_guard=candidate_guard;gpu=index;break
            if gpu is not None:break
            state['phase']='waiting_idle_audit_gpu';write(state_path,state);time.sleep(60)
        guard=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
        health=dict(before_progress=before,producer_identity=process,before_gpu=devices[gpu],authorized_overlap=process is not None,exclusive_idle_gpu=process is None,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
        write(health_path,health);state['active_audit']=name;write(state_path,state)
        command=[sys.executable,str(branch/'audit_marshawn_tuning_checkpoint.py'),'trace','--data',str(c/'data'),'--output',str(trace),'--queries',str(query),
                 '--receipt',str(folder/'step-000015188_grid.json'),'--checkpoint',str(folder/'checkpoint.ckpt'),'--config',str(folder/'config.yml'),'--code-root',str(frozen),
                 '--gpu',str(gpu),'--skip-wholegrid','--allow-readonly-overlap','--memory-cap-gib','10','--expected-sh','true','--expected-density','trunc_exp','--expected-projection','legacy']
        env=os.environ.copy();env.update(PYTHONPATH=str(frozen)+':'+str(frozen/'LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (folder/'clothing103_trace.log').open('w')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=frozen/'LookCloser')
        after=json.loads((producer/'progress.json').read_text())if producer is not None else{}
        producer_alive=alive(process)if process is not None else None
        healthy=result.returncode==0 and (process is None or (producer_alive and after['step']>=before['step']))
        health.update(after_progress=after,producer_alive_after=producer_alive,audit_environment_healthy=bool(healthy),after_gpu=gpu_snapshot()[gpu],returncode=result.returncode)
        write(health_path,health);state['audits'].append(dict(name=name,returncode=result.returncode,before_step=before.get('step'),after_step=after.get('step'),producer_alive_after=producer_alive,exclusive_idle_gpu=process is None));write(state_path,state);guard.close()
        if global_guard:global_guard.close()
        if not healthy:
            state['phase']='audit_requires_attention';write(state_path,state);raise RuntimeError('Audit failure/producer health recorded separately; training not modified')
    state['phase']='traces_complete';state['finished']=time.time();write(state_path,state)
    comparison=a/'exp_geometry_matched15k'
    if not (comparison/'complete.json').exists():
        subprocess.run([sys.executable,str(branch/'compare_marshawn_exp_geometry_traces.py'),'--campaign',str(c),'--output',str(comparison)],check=True)
    subprocess.run([sys.executable,str(branch/'summarize_marshawn_depth_residuals.py'),'--comparison',str(comparison/'comparison.json'),'--output',str(comparison/'depth_residuals.json')],check=True)
    state['phase']='report_complete';state['report']=str(comparison);write(state_path,state);print(json.dumps(state),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args())
