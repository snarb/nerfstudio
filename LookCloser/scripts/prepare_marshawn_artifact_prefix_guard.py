"""Orchestration-only generation: reuse tau v3 local controller / remote watchdog."""
import argparse,ast,copy,difflib,hashlib,importlib.util,json,shutil
from pathlib import Path

GEN='external_guard_prefix196_v1'
ARMS=['control','tau001','tau01']
RECEIPTS={'root_four_prefix_qualification.json':'0850f67b215affe471aca1975a07a5eea4952325074751aed7084f2b68cf988f'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def replace_function(text,name,new):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name);lines=text.splitlines(True);return ''.join(lines[:node.lineno-1])+new+'\n'+''.join(lines[node.end_lineno:])


def main(root,base,remote,uuids):
    out=root/GEN;out.mkdir(exist_ok=False)
    for name,d in RECEIPTS.items():assert sha(root/name)==d
    science=read(root/'source_manifest.json');assert all(sha(root/'code'/p)==d for p,d in science['files_sha256'].items())
    source=(base/'guardian.py').read_text();text=source.replace('external_guard_v3',GEN).replace('diagnostic','production').replace("['control','old7369','seven10481']",repr(ARMS)).replace('zero_summary.json','zero_verification.json').replace('quarter91128_tau_production_','prefix126316_production_')
    text=text.replace("read(root/'"+GEN+"/guard_release.json')","read(root/'"+GEN+"'/(phase+'_guard_release.json'))")
    text=replace_function(text,'check_limits',"""def check_limits(memory,physical_uuids,expected_uuid,age,policy):
    reasons=[]
    if memory>policy['process_cap_MiB']:reasons.append('memory_limit')
    if any(u!=expected_uuid for u in physical_uuids):reasons.append('physical_GPU_mismatch')
    deadline=policy['deadline_seconds_per_invocation']
    if deadline is not None and age>deadline:reasons.append('deadline')
    return reasons
""")
    text=replace_function(text,'require_phase_proof',"""def require_phase_proof(policy,root,phase,release):
    if phase not in release.get('authorized_phases',[]) or phase not in policy['phase_plans']:
        raise ValueError('Phase not authorized by exact external guard release')
    for name,digest in policy['root_receipts'].items():
        if sha(Path(root)/name)!=digest or release.get(name.removesuffix('.json')+'_sha256')!=digest:
            raise ValueError('Exact root source/CPU receipt required')
    if phase in ('technical','production'):
        name='zero_verification.json' if phase=='technical' else 'technical_verification.json'
        path=Path(root)/name;expected=release.get(name.removesuffix('.json')+'_sha256')
        if not expected or sha(path)!=expected or read(path).get('passed') is not True:
            raise ValueError('Exact passing predecessor phase proof required')
""")
    text=text.replace("policy=verify_policy(policy_path);root=Path(policy['remote_root']);out=root/'"+GEN+"'/phase","policy=verify_policy(policy_path);policy=dict(policy,deadline_seconds_per_invocation=policy['phase_deadlines'][phase]);root=Path(policy['remote_root']);out=root/'"+GEN+"'/phase")
    text=text.replace("r=read(receipt);w=r.get('worker')","r=read(receipt);w=r.get('worker')\n                if r.get('gpu_uuid') and (r.get('gpu') not in policy['approved_gpu_indices'] or r['gpu_uuid']!=policy['approved_gpu_UUIDs'].get(str(r['gpu']))):fail(state,'unapproved_GPU_lease',job=name)",1)
    compile(text,str(out/'guardian.py'),'exec');(out/'guardian.py').write_text(text);(out/'guardian.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile='proven_tau_v3_guardian',tofile=GEN+'/guardian.py')))
    paths={};plan_hashes={}
    for phase in ('zero','technical','production'):
        plan=copy.deepcopy(read(root/(phase+'_plan.json')));plan['hosts']['silver']['address']='ubuntu@silver-sable';plan['hosts']['silver']['gpus']=[0,1,2,3,5,6];write(out/(phase+'_plan.json'),plan);paths[phase]=GEN+'/'+phase+'_plan.json';plan_hashes[phase]=sha(out/(phase+'_plan.json'))
        assert plan['jobs']==read(root/(phase+'_plan.json'))['jobs']
    oldpolicy=read(base/'policy.json');policy={k:oldpolicy[k]for k in ['version','process_cap_MiB','worker_allocator_GiB','term_grace_seconds','heartbeat_timeout_seconds','max_consecutive_errors','max_error_seconds','poll_seconds']}
    policy.update(local_root=str(root),remote_root=remote,local_python=str(root.parent/'python'),controller_path=str(root.parent/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'),address='ubuntu@silver-sable',worker_path=remote+'/code/LookCloser/scripts/train_marshawn_artifact_prefix.py',guardian_sha256=sha(out/'guardian.py'),controller_sha256=sha(root.parent/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'),phase_plans=paths,phase_deadlines=dict(zero=900,technical=900,production=None),deadline_seconds_per_invocation=None,approved_gpu_indices=[0,1,2,3,5,6],approved_gpu_UUIDs={str(k):uuids[str(k)]for k in [0,1,2,3,5,6]},root_receipts=RECEIPTS,launch_origin='LOCAL_OLIVE_C_ONLY',science_snapshot_unchanged=True)
    bind=[root/'source_manifest.json',root/'confidence/manifest.json',root/'anchor_audit.json',root/'CPU_readiness.json',*[root/n for n in RECEIPTS],*list((root/'specs').rglob('spec.json')),*[out/(p+'_plan.json')for p in paths],out/'guardian.py']
    policy['prepared_bindings']={str(p.relative_to(root)):sha(p)for p in bind};write(out/'policy.json',policy)
    for phase in paths:
        template=dict(reviewer='root',external_guard_authorized=False,policy_sha256=sha(out/'policy.json'),authorized_phases=[phase],**{n.removesuffix('.json')+'_sha256':d for n,d in RECEIPTS.items()})
        if phase!='zero':template[('zero'if phase=='technical'else'technical')+'_verification_sha256']='ROOT_BIND_AFTER_PREDECESSOR_PASSES'
        write(out/(phase+'_guard_release.template.json'),template)
    unchanged=[];oldfunc={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)};newfunc={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    for name in ['identity','same','authorized_worker','event','fail','bounded_error','clear_errors','phase_status','signal_due','final_record','ssh_script']:
        assert oldfunc[name]==newfunc[name],name;unchanged.append(name)
    spec=importlib.util.spec_from_file_location('rgb_guard',out/'guardian.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    guard.require_local_origin(policy,out/'policy.json');guard.require_phase_proof(policy,root,'zero',dict(template,authorized_phases=['zero'],**{n.removesuffix('.json')+'_sha256':d for n,d in RECEIPTS.items()}))
    assert guard.check_limits(100,[],uuids['0'],1e9,dict(policy,deadline_seconds_per_invocation=None))==[]
    assert guard.check_limits(16385,[],uuids['0'],1,dict(policy,deadline_seconds_per_invocation=900))==['memory_limit']
    assert guard.check_limits(1,[],uuids['0'],901,dict(policy,deadline_seconds_per_invocation=900))==['deadline']
    remote_node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='remote_guard');assert 'controller_path'not in ast.get_source_segment(text,remote_node)
    write(out/'CPU_orchestration.json',dict(passed=True,original1495_science_unchanged=True,science_worker_sha256=sha(root/'code/LookCloser/scripts/train_marshawn_artifact_prefix.py'),unchanged_guard_core_functions=unchanged,local_origin_check=True,remote_guardian_only=True,plans_have_explicit_remote_address=True,job_commands_and_specs_unchanged=True,approved_GPU_UUIDs=policy['approved_gpu_UUIDs'],root_receipts=RECEIPTS,old_proofs_overwritten=False,GPU_started=False))
    write(out/'preservation.json',dict(base_guardian_sha256=sha(base/'guardian.py'),science_manifest_sha256=sha(root/'source_manifest.json'),confidence_manifest_sha256=sha(root/'confidence/manifest.json'),CPU_readiness_sha256=sha(root/'CPU_readiness.json'),old_phase_runner_sha256=None,policy_sha256=sha(out/'policy.json'),plans_sha256=plan_hashes,release_files_created=False))
    print(json.dumps(dict(policy_sha256=sha(out/'policy.json'),guardian_sha256=sha(out/'guardian.py'),launch_entry=str(out/'guardian.py'),generation=GEN)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--remote',required=True);p.add_argument('--UUIDs',type=Path,required=True);a=p.parse_args();main(a.root,a.base,a.remote,read(a.UUIDs))
