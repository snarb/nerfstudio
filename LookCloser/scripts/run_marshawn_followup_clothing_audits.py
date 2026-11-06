"""Three matched15188 native-exp103-ray audits; read-only bounded overlap."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def trace_one(args):
    c=args.campaign;branch=c/'exp_geometry_candidate_20261106';frozen=Path(args.code_root)
    sys.path.insert(0,str(frozen));sys.path.insert(0,str(frozen/'LookCloser/scripts'));sys.path.insert(0,str(branch))
    import nerfstudio.utils.eval_utils as eu
    import audit_marshawn_tuning_checkpoint as audit
    original=eu.eval_setup
    def checked(*a,**kw):
        config,pipe,cp,step=original(*a,**kw)
        actual=pipe.model.config.adaptive_coarse_step_size
        if step!=15188 or actual!=args.coarse_step:raise ValueError('Actual checkpoint step/coarse sampling differs')
        write(args.output/'actual_parameter_binding.json',dict(step=step,adaptive_coarse_step_size=actual,expected_coarse_step=args.coarse_step,
            config_sha256=sha(args.config),model_source_sha256=sha(frozen/'nerfstudio/models/lookcloser.py'),optimizer_updates=0))
        return config,pipe,cp,step
    eu.eval_setup=checked
    args.data=c/'data';args.skip_wholegrid=True;args.allow_readonly_overlap=True;args.memory_cap_gib=10.;args.expected_sh='true';args.expected_density='trunc_exp';args.expected_projection='legacy'
    try:audit.trace(args)
    finally:eu.eval_setup=original


def run(c):
    branch=c/'exp_geometry_candidate_20261106';a=c/'independent_occupancy';root=a/'exp_followup103_matched15k';root.mkdir(exist_ok=True)
    guard=(root/'runner.lock').open('a+')
    try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    sys.path.insert(0,str(branch/'code/LookCloser/scripts'));from marshawn_tuning_controller import gpu_snapshot,identity,alive
    state=dict(pid=os.getpid(),phase='running',audits=[]);write(root/'state.json',state)
    query=branch/'clothing_103_predeclared.json';assert sha(query)=='e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'
    requests=[('exp_reference',a/'structural_exp_hplus1_015188','exp_followup_actor',c/'structural_extension_code',.00625),
              ('actor',a/'exp_followup_actor_015188','exp_followup_actor',c/'exp_followup_code',.00625),
              ('step_half',a/'exp_followup_step_half_015188','exp_followup_step_half',c/'exp_followup_code',.003125)]
    for name,folder,job,frozen,coarse in requests:
        trace=folder/'clothing103_trace'
        if(trace/'rays/diagnostic_summary.json').exists():state['audits'].append(dict(name=name,reused=True));write(root/'state.json',state);continue
        producer=c/'jobs'/job/'run';receipt=json.loads((c/'jobs'/job/'receipt.json').read_text());gpu=int(receipt['gpu'])
        lock=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
        before=json.loads((producer/'progress.json').read_text());process=identity(before['pid']);devices=gpu_snapshot()
        if not process or not alive(process)or not any(x['pid']==before['pid']for x in devices[gpu]['processes']):raise ValueError('Qualified live producer GPU required')
        health=dict(before_progress=before,producer_identity=process,before_gpu=devices[gpu],authorized_overlap=True,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
        write(folder/'clothing103_producer_health.json',health);state['active']=name;write(root/'state.json',state)
        if trace.exists():trace.rename(folder/('incomplete_trace_'+str(int(time.time()))))
        cmd=[sys.executable,str(Path(__file__).resolve()),'trace','--campaign',str(c),'--code-root',str(frozen),'--coarse-step',str(coarse),'--output',str(trace),'--gpu',str(gpu),
            '--receipt',str(folder/'step-000015188_grid.json'),'--checkpoint',str(folder/'checkpoint.ckpt'),'--config',str(folder/'config.yml'),'--queries',str(query)]
        env=dict(os.environ,PYTHONPATH=f'{frozen}:{frozen}/LookCloser/scripts',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with(folder/'clothing103_trace.log').open('a')as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=frozen/'LookCloser')
        after=json.loads((producer/'progress.json').read_text());ok=r.returncode==0 and alive(process)and after['step']>=before['step']
        health.update(after_progress=after,producer_alive_after=alive(process),after_gpu=gpu_snapshot()[gpu],returncode=r.returncode,audit_environment_healthy=bool(ok));write(folder/'clothing103_producer_health.json',health);lock.close()
        state['audits'].append(dict(name=name,returncode=r.returncode,healthy=bool(ok),before_step=before['step'],after_step=after['step']));write(root/'state.json',state)
        if not ok:raise RuntimeError('Read-only audit requires attention; production not modified')
    subprocess.run([sys.executable,str(branch/'compare_marshawn_followup_clothing_traces.py'),'--campaign',str(c),'--output',str(root)],check=True)
    subprocess.run([sys.executable,str(branch/'summarize_marshawn_depth_residuals.py'),'--comparison',str(root/'comparison.json'),'--output',str(root/'depth_residuals.json')],check=True)
    state.update(phase='complete',time=time.time());write(root/'state.json',state);print(root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','trace']);p.add_argument('--campaign',type=Path,required=True)
    for name in ['code-root','output','receipt','checkpoint','config','queries']:p.add_argument('--'+name,type=Path)
    p.add_argument('--coarse-step',type=float);p.add_argument('--gpu',type=int);args=p.parse_args()
    if args.mode=='run':run(args.campaign.resolve())
    else:trace_one(args)
