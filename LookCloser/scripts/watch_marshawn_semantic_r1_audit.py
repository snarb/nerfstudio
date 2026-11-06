"""Preserve first15188 semantic checkpoint and audit103TRAIN plus12DC diagnostic rays."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path


def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def main(c,branch):
    scripts=branch/'code/LookCloser/scripts';sys.path.insert(0,str(scripts))
    from marshawn_tuning_controller import gpu_snapshot,identity,alive
    from marshawn_semantic_admission import sha
    helper_root=Path(__file__).resolve().parent
    for name,digest in json.loads((helper_root/'manifest.json').read_text())['files_sha256'].items():
        if sha(helper_root/name)!=digest:raise ValueError('Audit helper source changed')
    lock=(branch/'semantic_audit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=branch/'semantic_audit_state.json';state=json.loads(state_path.read_text())if state_path.exists()else dict(phase='waiting15188',audits=[],pid=os.getpid());state['pid']=os.getpid();write(state_path,state)
    if state['phase']=='complete':return
    run=branch/'run';folder=c/'independent_occupancy/semantic_actor_r1_015188';folder.mkdir(exist_ok=True)
    cp=run/'checkpoints/step-000015188.ckpt';receipt=run/'occupancy/step-000015188_grid.json'
    while not (folder/'archive_receipt.json').exists():
        if all(p.exists()for p in [cp,receipt,run/'renders/step-000015188/metrics.json']):
            if not(folder/'checkpoint.ckpt').exists():os.link(cp,folder/'checkpoint.ckpt')
            for src in [receipt,receipt.with_suffix('.npz'),run/'worker/lookcloser/frozen/config.yml']:shutil.copy2(src,folder/src.name)
            shutil.copytree(run/'renders/step-000015188',folder/'renders',dirs_exist_ok=True)
            digest=sha(folder/'checkpoint.ckpt')
            if digest!=json.loads(receipt.read_text())['checkpoint_sha256']:raise ValueError('Archived semantic checkpoint hash mismatch')
            write(folder/'archive_receipt.json',dict(checkpoint_sha256=digest,step=15188,source=str(cp),immutable_hardlink=True,semantic_spec_sha256=sha(branch/'spec.json')))
            break
        state['last_poll']=time.time();write(state_path,state);time.sleep(60)
    state['phase']='preserved';state['archive']=str(folder);write(state_path,state)
    helpers=branch/'audit_helpers';code=branch/'code';env=dict(os.environ,PYTHONPATH=str(helpers)+':'+str(code)+':'+str(scripts),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,query,digest in [('clothing103',branch/'source/TRAIN103_queries.json','e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'),('DC12',c/'independent_occupancy/DC_error12_15188/queries.json','08caff3fbdb24a811745761b3fa685c33fd8655be6342203cf851f4fcf1b6d00')]:
        if sha(query)!=digest:raise ValueError('Sealed query changed')
        output=folder/(name+'_trace');health_file=folder/(name+'_producer_health.json')
        if (output/'rays/diagnostic_summary.json').exists():continue
        global_guard=None
        while True:
            state['phase']='waiting_producer_or_idle_gpu';write(state_path,state)
            job=branch/'controller/jobs/exp_actor_semantic_W64pad1_r1/receipt.json'
            if not job.exists():time.sleep(60);continue
            job=json.loads(job.read_text());gpu=int(job['gpu']);before=json.loads((run/'progress.json').read_text());producer=identity(before['pid']);devices=gpu_snapshot()
            if not(producer and alive(producer)):
                producer=None;before={}
                for index,device in devices.items():
                    if device['processes']or device['used_mib']>128:continue
                    claim=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{index}.lock').open('a+')
                    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:claim.close();continue
                    if gpu_snapshot()[index]['processes']:claim.close();continue
                    global_guard=claim;gpu=index;break
                if global_guard is not None:break
                time.sleep(60);continue
            if any(r['pid']==before['pid']for r in devices[gpu]['processes']):break
            time.sleep(60)
        guard=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+')
        fcntl.flock(guard,fcntl.LOCK_EX)
        health=dict(before_progress=before,producer_identity=producer,before_gpu=devices[gpu],authorized_overlap=producer is not None,exclusive_idle_gpu=producer is None,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
        write(health_file,health);state['phase']='auditing_'+name;write(state_path,state)
        command=[sys.executable,str(helpers/'audit_marshawn_semantic_checkpoint.py'),'--data',str(c/'data'),'--output',str(output),'--queries',str(query),'--receipt',str(folder/'step-000015188_grid.json'),'--checkpoint',str(folder/'checkpoint.ckpt'),'--config',str(folder/'config.yml'),'--code-root',str(code),'--gpu',str(gpu),'--memory-cap-gib','10']
        with(folder/(name+'_trace.log')).open('w')as log:result=subprocess.run(command,env=env,cwd=code/'LookCloser',stdout=log,stderr=subprocess.STDOUT)
        after=json.loads((run/'progress.json').read_text())if producer is not None else{};producer_alive=alive(producer)if producer is not None else None;healthy=result.returncode==0 and (producer is None or (producer_alive and after['step']>=before['step']));health.update(after_progress=after,producer_alive_after=producer_alive,after_gpu=gpu_snapshot()[gpu],returncode=result.returncode,audit_environment_healthy=bool(healthy));write(health_file,health);guard.close()
        if global_guard is not None:global_guard.close()
        state['audits'].append(dict(name=name,returncode=result.returncode,healthy=healthy));write(state_path,state)
        if not healthy:raise RuntimeError('Independent audit needs attention; producer unchanged')
    state['phase']='traces_complete';write(state_path,state)
    subprocess.run([sys.executable,str(helpers/'compare_marshawn_semantic_r1_traces.py'),'--campaign',str(c),'--output',str(c/'independent_occupancy/semantic_r0_r1_matched15k')],env=env,check=True)
    subprocess.run([sys.executable,str(helpers/'summarize_marshawn_depth_residuals.py'),'--comparison',str(c/'independent_occupancy/semantic_r0_r1_matched15k/comparison.json'),'--output',str(c/'independent_occupancy/semantic_r0_r1_matched15k/depth_residuals.json')],env=env,check=True)
    subprocess.run([sys.executable,str(helpers/'compare_marshawn_semantic_r1_dc.py'),'--campaign',str(c),'--output',str(c/'independent_occupancy/semantic_r0_r1_DC12_matched15k')],env=env,check=True)
    state['phase']='complete';write(state_path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--branch',type=Path,required=True);a=p.parse_args();main(a.campaign,a.branch)
