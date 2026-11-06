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
    c=args.campaign.resolve();a=c/'independent_occupancy';branch=c/'sh_geometry_candidate_20261106';code=branch/'code';scripts=code/'LookCloser/scripts'
    sys.path.insert(0,str(scripts));from marshawn_tuning_controller import gpu_snapshot,identity,alive
    lock=(a/'clothing_103_watcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=a/'clothing_103_watcher.json';state=dict(pid=os.getpid(),poll_seconds=60,phase='waiting',step=15188,preserved={},audits=[])
    write(state_path,state)
    while len(state['preserved'])<3:
        for arm in 'ABC':
            if arm in state['preserved']:continue
            run=branch/('run_'+arm);cp=run/'checkpoints/step-000015188.ckpt';receipt=run/'occupancy/step-000015188_grid.json';raster=run/'renders/step-000015188/metrics.json'
            if not all(p.exists()for p in [cp,receipt,raster]):continue
            folder=a/('sh_geometry_'+arm+'_015188');folder.mkdir(exist_ok=True);target=folder/'checkpoint.ckpt'
            if not target.exists():os.link(cp,target)
            for src in [run/'worker/lookcloser/frozen/config.yml',receipt,receipt.with_suffix('.npz')]:shutil.copyfile(src,folder/src.name)
            shutil.copytree(run/'renders/step-000015188',folder/'renders',dirs_exist_ok=True)
            archive=dict(step=15188,source_checkpoint=str(cp),checkpoint_sha256=sha(target),preserved_hardlink=True,
                         original_spec_sha256=sha(branch/('spec_'+arm+'.json')),protected_at=time.time())
            write(folder/'archive_receipt.json',archive);state['preserved'][arm]=dict(folder=str(folder),**archive);write(state_path,state)
        if len(state['preserved'])<3:
            state['last_poll']=time.time();state['producer_steps']={arm:json.loads((branch/('run_'+arm)/'progress.json').read_text())['step']for arm in 'ABC'};write(state_path,state);time.sleep(60)
    state['phase']='auditing';write(state_path,state)
    query=a/'queries/clothing_103_predeclared.json';expected=json.loads(query.read_text());assert len(expected['pixels'])==103
    requests=[('SH_reference',5,branch/'run_A',a/'sh_corrected_015188',c/'sh_corrected_code')]
    requests += [(arm,gpu,branch/('run_'+arm),Path(state['preserved'][arm]['folder']),code)for arm,gpu in zip('ABC',[5,6,7])]
    for name,gpu,producer,folder,frozen in requests:
        trace=folder/'clothing103_trace';health_path=folder/'clothing103_producer_health.json'
        if (trace/'rays/diagnostic_summary.json').exists():state['audits'].append(dict(name=name,reused_complete_trace=True));write(state_path,state);continue
        guard=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=json.loads((producer/'progress.json').read_text());process=identity(before['pid']);devices=gpu_snapshot();assert process and alive(process)and any(x['pid']==before['pid']for x in devices[gpu]['processes'])
        health=dict(before_progress=before,producer_identity=process,before_gpu=devices[gpu],authorized_overlap=True,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
        write(health_path,health);state['active_audit']=name;write(state_path,state)
        command=[sys.executable,str(a/'audit_marshawn_structural.py'),'trace','--data',str(c/'data'),'--output',str(trace),'--queries',str(query),
                 '--receipt',str(folder/'step-000015188_grid.json'),'--checkpoint',str(folder/'checkpoint.ckpt'),'--config',str(folder/'config.yml'),'--code-root',str(frozen),
                 '--gpu',str(gpu),'--skip-wholegrid','--allow-readonly-overlap','--memory-cap-gib','10','--expected-sh','true','--expected-density','softplus','--expected-projection','legacy']
        env=os.environ.copy();env.update(PYTHONPATH=str(frozen)+':'+str(frozen/'LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (folder/'clothing103_trace.log').open('w')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=frozen/'LookCloser')
        after=json.loads((producer/'progress.json').read_text());health.update(after_progress=after,producer_alive_after=alive(process),after_gpu=gpu_snapshot()[gpu],returncode=result.returncode)
        write(health_path,health);state['audits'].append(dict(name=name,returncode=result.returncode,before_step=before['step'],after_step=after['step'],producer_alive_after=health['producer_alive_after']));write(state_path,state);guard.close()
        if result.returncode or not health['producer_alive_after']or after['step']<before['step']:
            state['phase']='audit_requires_attention';write(state_path,state);raise RuntimeError('Audit failure/producer health recorded separately; training not modified')
    state['phase']='traces_complete';state['finished']=time.time();write(state_path,state);print(json.dumps(state),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args())
