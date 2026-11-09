"""Actual staged closure, original1495/guardian-core and false-gate CPU audit."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main(root):
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    policy=read(root/'root_review_TEMPLATE.json');bad=[p for p,h in policy['bound_files'].items()if not Path(p).is_file()or sha(p)!=h]
    if bad:raise ValueError('Actual stage mismatch:'+str(bad))
    old=root.parent/'distortion03_warm_continuation_candidate_v1'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'helpers')+':'+str(old/'code')+':'+str(old/'code/LookCloser/scripts'))
    dryruns=[]
    for spec in sorted((root/'specs').glob('*/spec.json')):
        d=read(spec);release=read(d['release']);assert release['technical_authorized']is False and release['production_authorized']is False
        assert release['spec_sha256']==sha(spec)
        out=subprocess.run([sys.executable,str(root/'helpers/train_marshawn_below_floor_density_warm.py'),'--spec',str(spec),'--dry-run'],env=env,capture_output=True,text=True)
        fail=subprocess.run([sys.executable,str(root/'helpers/train_marshawn_below_floor_density_warm.py'),'--spec',str(spec)],env=env,capture_output=True,text=True)
        if out.returncode or fail.returncode==0 or'Exact root phase release required'not in fail.stderr:raise ValueError('CPU dryrun/false gate failed:'+str(spec))
        dryruns.append(dict(case=spec.parent.name,dryrun=out.returncode,false_release_failed_before_Torch=True))
    for phase,count in [('zero',1),('technical',9)]:
        plan=read(root/(phase+'_plan.json'));assert len(plan['jobs'])==count
        assert plan['hosts']['silver']['gpus']==[5,6,7]
        for j in plan['jobs']:
            assert j['gpu']in[5,6,7]and'--spec'in j['command']
            assert j['command'][1]==str(root/'helpers/train_marshawn_below_floor_density_warm.py')
        result=subprocess.run([sys.executable,str(root/'helpers/marshawn_tuning_controller.py'),'dry-run','--root',str(root/('controller_'+phase)),'--manifest',str(root/(phase+'_plan.json'))],env=env,capture_output=True,text=True)
        assert result.returncode==0
    guardian=load('_new_guard_CPU',root/'external_guard_below_floor_v1/guardian.py')
    previous=load('_old_guard_CPU',old/'external_guard_distortion03_v1/guardian.py')
    core=['identity','same','authorized_worker','event','fail','check_limits','bounded_error','clear_errors','phase_status','signal_due','final_record']
    assert all(inspect.getsource(getattr(guardian,n))==inspect.getsource(getattr(previous,n))for n in core)
    gp=read(root/'external_guard_below_floor_v1/policy.json')
    for rel,h in gp['prepared_bindings'].items():assert sha(root/rel)==h
    assert set(gp['phase_plans'])=={'zero','technical'}and gp['approved_gpu_indices']==[5,6,7]
    limit=dict(gp,deadline_seconds_per_invocation=1800)
    assert guardian.check_limits(16384,['u'],'u',1800,limit)==[]
    assert guardian.check_limits(16385,['v'],'u',1801,limit)==['memory_limit','physical_GPU_mismatch','deadline']
    expected=dict(pid=99999999,start='abc',boot='def',argv=['python','worker','spec'])
    actual=dict(expected,state='R');signals=[];state={}
    assert guardian.signal_due(state,'k',expected,'worker','spec',0,limit,lookup=lambda _:actual,send=lambda *x:signals.append(x))=='TERM'
    assert guardian.signal_due(state,'k',expected,'worker','spec',24,limit,lookup=lambda _:actual,send=lambda *x:signals.append(x))=='waiting'
    assert guardian.signal_due(state,'k',expected,'worker','spec',25,limit,lookup=lambda _:actual,send=lambda *x:signals.append(x))=='KILL'
    before=len(signals)
    assert guardian.signal_due({},'k',expected,'worker','spec',0,limit,lookup=lambda _:dict(actual,start='reused'),send=lambda *x:signals.append(x))=='absent_or_identity_mismatch'
    assert len(signals)==before
    # No actual process is signalled: every send/identity operation above is injected.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuids={x.split(',')[0].strip():x.split(',')[1].strip()for x in raw.splitlines()}
    assert all(uuids[k]==v for k,v in gp['approved_gpu_UUIDs'].items())
    sys.path[:0]=[str(old/'code/LookCloser/scripts')]
    import torch
    from train_marshawn_semantic_late_distortion import tree_digest
    anchor=read(old/'anchor_audit.json');checkpoint=root/'anchor/checkpoint.ckpt'
    assert sha(checkpoint)=='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert all(tree_digest(payload[k])==h for k,h in anchor['fullstate_hashes'].items())
    del payload
    probe=root/'CPU_actual_write_probe';probe.mkdir();probe.rmdir()
    if torch.cuda.is_initialized():raise ValueError('CPU stage audit initializedCUDA')
    report=dict(passed=True,CPU_only=True,CUDA_started=False,actual_bound_files=len(policy['bound_files']),
        source_checkpoint_seven_states_exact=True,source_checkpoint_sha256=sha(checkpoint),
        source1495_files_unchanged=True,worker_dryruns=dryruns,zero_jobs=1,technical_jobs=9,
        approved_actual_UUIDs=gp['approved_gpu_UUIDs'],guardian_core_functions_byte_identical=core,
        guardian_memory_UUID_deadline_TERM_grace_PID_reuse_tests_pass=True,
        all_per_spec_releases_false=True,no_production_plan=True,actual_write_remove=True,
        all_run_outputs_absent=all(not Path(read(p)['output']).exists()for p in(root/'specs').glob('*/spec.json')))
    (root/'CPU_readiness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    main(Path(sys.argv[1]))
