"""Separate false production contracts; never modifies the sealed technical tree."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106');SC='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
T=C/'below_floor_density_warm_technical_plan_v1';TR=SC+'/'+T.name
P=C/'below_floor_density_warm_production_plan_v1';R=SC+'/'+P.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2)+'\n')


def main():
    P.mkdir(exist_ok=True);(P/'helpers').mkdir(exist_ok=True)
    for name in ['train_marshawn_below_floor_density_warm.py','marshawn_below_floor_density_prior.py','marshawn_tuning_controller.py']:
        shutil.copyfile(T/'helpers'/name,P/'helpers'/name)
    table='frozen_active_below_band_cells.npz';shutil.copyfile(T/table,P/table)
    cases=[('production_control0','control0_full64',5),('production_prior1e4','prior1e4_full64',6),('production_prior1e3','prior1e3_full64',7)]
    spec_hashes={};bound={};jobs=[]
    for name,source,gpu in cases:
        s=json.loads((T/'specs'/source/'spec.json').read_text());s.update(output=R+'/runs/'+name,mode='production',
            parent_checkpoint=R+'/anchor/checkpoint.ckpt',release=R+'/specs/'+name+'/root_release.json')
        s['frozen_source_hashes']={k.replace(TR,R):v for k,v in s['frozen_source_hashes'].items()}
        s['below_floor_density_prior']['table']=R+'/'+table
        file=P/'specs'/name/'spec.json';write(file,s);spec_hashes[R+'/specs/'+name+'/spec.json']=sha(file)
        release=dict(reviewer='root',production_authorized=False,technical_authorized=False,spec_sha256=sha(file))
        write(file.with_name('root_review_TEMPLATE.json'),release);write(file.with_name('root_release.json'),release)
        bound.update(s['frozen_source_hashes'])
        for key in ['parent_checkpoint','parent_spec','anchor_audit','confidence_manifest']:bound[s[key]]=s[key+'_sha256']
        bound[R+'/specs/'+name+'/spec.json']=sha(file)
        oldjob=json.loads((T/'technical_plan.json').read_text())['jobs'][0]
        env={k:v.replace(TR,R)for k,v in oldjob['env'].items()}
        jobs.append(dict(name=name,host='silver',gpu=gpu,
            command=[SC+'/python',R+'/helpers/train_marshawn_below_floor_density_warm.py','--spec',R+'/specs/'+name+'/spec.json'],
            cwd=R+'/helpers',env=env,dependencies=[],progress_files=[R+'/runs/'+name+'/progress.json'],
            metrics_file=R+'/runs/'+name+'/complete.json'))
    host=json.loads((T/'technical_plan.json').read_text())['hosts']['silver']
    host.update(controller=R+'/helpers/marshawn_tuning_controller.py',root=R+'/remote_queue')
    write(P/'production_plan.json',dict(schema_version=1,poll_seconds=15,hosts=dict(silver=host),jobs=jobs))
    G=P/'external_guard_below_floor_v1';G.mkdir(exist_ok=True)
    original=(T/'external_guard_below_floor_v1/guardian.py').read_text();text=original
    old="""        fresh_script+="for arm in ['control01', 'distortion03']:\\n s=json.loads((r/'specs'/('production_'+arm)/'spec.json').read_text())\\n assert not pathlib.Path(s['output']).exists(),s['output']\\n assert not (r/'remote_queue/jobs'/('distortion126316_production_'+arm)).exists(),arm\\n"
"""
    new="""        fresh_script+="for case,job_name in "+repr(list(policy['production_cases_and_jobs'].items()))+":\\n s=json.loads((r/'specs'/case/'spec.json').read_text())\\n assert not pathlib.Path(s['output']).exists(),s['output']\\n assert not (r/'remote_queue/jobs'/job_name).exists(),job_name\\n"
"""
    if text.count(old)!=1:raise ValueError('Exact qualified production freshness block missing')
    text=text.replace(old,new);(G/'guardian.py').write_text(text);compile(text,str(G/'guardian.py'),'exec')
    (G/'guardian.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='technical/guardian.py',tofile='production/guardian.py')))
    policy=json.loads((T/'external_guard_below_floor_v1/policy.json').read_text())
    policy.update(local_root=str(P),remote_root=R,controller_path=str(P/'helpers/marshawn_tuning_controller.py'),
        worker_path=R+'/helpers/train_marshawn_below_floor_density_warm.py',guardian_sha256=sha(G/'guardian.py'),
        phase_plans={'production':'production_plan.json'},phase_deadlines={'production':None},
        phase_spec_cases={'production':[q[0]for q in cases]},production_cases_and_jobs={q[0]:q[0]for q in cases},root_receipts={},prepared_bindings={})
    for file in [*(P/'helpers').glob('*.py'),P/table,P/'production_plan.json',G/'guardian.py',*(P/'specs').glob('*/spec.json')]:
        policy['prepared_bindings'][str(file.relative_to(P))]=sha(file);bound[str(file).replace(str(C),SC)]=sha(file)
    write(G/'policy.json',policy);bound[R+'/external_guard_below_floor_v1/policy.json']=sha(G/'policy.json')
    write(G/'production_guard_release.template.json',dict(reviewer='root',external_guard_authorized=False,
        policy_sha256=sha(G/'policy.json'),authorized_phases=['production'],technical_verification_sha256=None))
    write(P/'root_review_TEMPLATE.json',dict(reviewer='root',below_floor_production_authorized=False,
        exact_worker_bytes_sha256=sha(P/'helpers/train_marshawn_below_floor_density_warm.py'),
        parent_technical_root=str(T),technical_acceptance_required=True,external_guard_policy_sha256=sha(G/'policy.json'),
        production_plan_sha256=sha(P/'production_plan.json'),spec_sha256=spec_hashes,bound_files=bound,
        source_start='fresh original126316; never a technical checkpoint',private_CPU_seed=42,
        early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None,all_checkpoints_retained=True))
    print(json.dumps(dict(package=str(P),worker_identical=sha(P/'helpers/train_marshawn_below_floor_density_warm.py')==sha(T/'helpers/train_marshawn_below_floor_density_warm.py'),bindings=len(bound),CUDA_started=False)))


if __name__=='__main__':main()
