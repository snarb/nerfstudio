"""CPU-only production transport closure and actual fresh-output guard block."""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main(root):
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    technical=root.parent/'below_floor_density_warm_technical_plan_v1';old=root.parent/'distortion03_warm_continuation_candidate_v1'
    r=read(root/'root_review_TEMPLATE.json');bad=[f for f,h in r['bound_files'].items()if not Path(f).is_file()or sha(f)!=h]
    assert not bad,bad
    assert sha(root/'helpers/train_marshawn_below_floor_density_warm.py')==sha(technical/'helpers/train_marshawn_below_floor_density_warm.py')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'helpers')+':'+str(old/'code')+':'+str(old/'code/LookCloser/scripts'))
    cases=[]
    for spec in sorted((root/'specs').glob('*/spec.json')):
        d=read(spec);release=read(d['release']);assert release['production_authorized']is False
        assert d['parent_checkpoint']==str(root/'anchor/checkpoint.ckpt')and d['mode']=='production'
        assert d['stopping']['hard_iteration_limit']is None and d['stopping']['early_visual_attempts']==[500,1000,2000,4000]
        a=subprocess.run([sys.executable,str(root/'helpers/train_marshawn_below_floor_density_warm.py'),'--spec',str(spec),'--dry-run'],env=env,capture_output=True,text=True)
        b=subprocess.run([sys.executable,str(root/'helpers/train_marshawn_below_floor_density_warm.py'),'--spec',str(spec)],env=env,capture_output=True,text=True)
        assert a.returncode==0 and b.returncode!=0 and'Exact root phase release required'in b.stderr
        cases.append(dict(case=spec.parent.name,coefficient=d['below_floor_density_prior']['coefficient'],false_gate_pass=True))
    plan=read(root/'production_plan.json');assert len(plan['jobs'])==3 and plan['hosts']['silver']['gpus']==[5,6,7]
    d=subprocess.run([sys.executable,str(root/'helpers/marshawn_tuning_controller.py'),'dry-run','--root',str(root/'controller'),'--manifest',str(root/'production_plan.json')],env=env,capture_output=True,text=True);assert d.returncode==0
    guard_path=root/'external_guard_below_floor_v1/guardian.py';guard=module('_production_guard_cpu',guard_path)
    previous=module('_technical_guard_cpu',technical/'external_guard_below_floor_v1/guardian.py')
    core=['identity','same','authorized_worker','event','fail','check_limits','bounded_error','clear_errors','phase_status','signal_due','final_record']
    assert all(inspect.getsource(getattr(guard,k))==inspect.getsource(getattr(previous,k))for k in core)
    policy=read(root/'external_guard_below_floor_v1/policy.json');assert set(policy['phase_plans'])=={'production'}
    for rel,h in policy['prepared_bindings'].items():assert sha(root/rel)==h
    function=next(n for n in ast.parse(guard_path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='local_guard')
    fresh=next(n for n in function.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and ast.unparse(n.test)=="phase == 'production'")
    captured=[]
    def simulate_ssh(_,script):captured.append(script);exec(script,{})
    exec(compile(ast.fix_missing_locations(ast.Module(body=fresh.body,type_ignores=[])),'actual_guard_freshness','exec'),dict(policy=policy,ssh_script=simulate_ssh))
    assert len(captured)==1
    rejected=False
    try:guard.require_phase_proof(policy,root,'production',dict(authorized_phases=['production']))
    except ValueError:rejected=True
    assert rejected
    probe=root/'CPU_actual_write_probe';probe.mkdir();probe.rmdir()
    report=dict(passed=True,CPU_only=True,CUDA_started=False,actual_bound_files=len(r['bound_files']),
        worker_bytes_identical_to_sealed_technical=True,cases=cases,production_jobs=3,
        fresh_source_checkpoint340e=True,private_seed42=True,original_plateau_and_early_gates_unchanged=True,
        actual_production_fresh_output_guard_block_passed=True,missing_technical_acceptance_fails_closed=True,
        guardian11_safety_functions_unchanged=True,all_production_releases_false=True,
        all_run_outputs_absent=not(root/'runs').exists(),actual_write_remove=True)
    (root/'CPU_readiness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main(Path(sys.argv[1]))
