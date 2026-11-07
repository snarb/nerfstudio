"""Freeze an executable paired margin package; prepare releases, never grant them."""
import argparse,ast,copy,difflib,hashlib,json,os,shutil
from pathlib import Path
import numpy as np

ARMS=('original_mapped','source_margin002')
GEN='external_guard_margin_v1'
SOURCE_SHA='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
STOP=dict(minimum_attempts=4000,anchor_step=130316,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001,early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')

def prepare(c,root,remote):
    remote=Path(remote);parent=c/'semantic_quarter_distortion_candidate_20261107';rgb=c/'rgb_confidence_continuation_candidate_v1';previous=c/'independent_occupancy/quarter_distortion_source_margin_v1';original=read(parent/'source_manifest.json')['files_sha256']
    if(root/'code').exists():raise ValueError('Executable generation already exists; never silently re-freeze')
    assert sha(root/'protocol.json')=='659842196fa0351e48a5bf847433e582de48d08666b5486250f783126d2a083e'
    shutil.copytree(parent/'code',root/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    own=['marshawn_margin_continuation_policy.py','marshawn_margin_continuation_factory.py','train_marshawn_margin_continuation.py','verify_marshawn_margin_continuation_phase.py','export_marshawn_core_boundary_metrics_cpu.py','audit_marshawn_core_boundary_metrics_cpu.py']
    inherited=['train_marshawn_semantic_late_distortion.py','marshawn_gpu_isolation.py'];audited=['marshawn_source_point_admission.py','marshawn_source_margin_admission.py','marshawn_readonly_point_selector.py']
    for name in own+inherited+audited:
        target=root/'code/LookCloser/scripts'/name
        if target.exists():raise ValueError('Unexpected original source shadow: '+name)
        source=Path(__file__).with_name(name)if name in own else rgb/'code/LookCloser/scripts'/name if name in inherited else previous/'helpers'/name
        shutil.copy2(source,target)
    for rel,d in original.items():assert sha(root/'code'/rel)==d,rel
    shutil.copytree(rgb/'anchor',root/'anchor',ignore=shutil.ignore_patterns('checkpoint.ckpt'))
    anchor=read(rgb/'anchor_audit.json');oldremote=str(remote.parent/rgb.name)
    anchor['source_hashes']={p.replace(oldremote,str(remote)):d for p,d in anchor['source_hashes'].items()};anchor['render_directory']=anchor['render_directory'].replace(oldremote,str(remote));anchor['inherited_anchor_audit_sha256']=sha(rgb/'anchor_audit.json');write(root/'anchor_audit.json',anchor)
    source=root/'margin/source_grid.npz';source.parent.mkdir();shutil.copy2(previous/'source_grid.npz',source)
    traces=previous/'quarter_distortion_raw126316/native_A/contour9_trace/rays';points=[];sources={}
    for path in sorted(traces.glob('*_native.npz')):
        with np.load(path)as z:points.append(z['xyz'].astype(np.float32))
        sources[str(path)]=sha(path)
    points=np.concatenate(points);np.savez_compressed(root/'margin/probe_positions.npz',xyz=points)
    manifest=dict(kind='PAIRED_SOURCE_MARGIN002_RUNTIME',source_checkpoint_sha256=SOURCE_SHA,source_grid=str(remote/'margin/source_grid.npz'),source_grid_sha256=sha(source),geometric_Linf_margin=.002,numerical_epsilon=2e-7,outside_unknown='original_numerical_epsilon',helper_sha256={n:sha(root/'code/LookCloser/scripts'/n)for n in audited},probe_positions=str(remote/'margin/probe_positions.npz'),probe_positions_sha256=sha(root/'margin/probe_positions.npz'),probe_provenance=sources,probe_usage='Existing nine diagnostic native rays, no train targets or optimizer gradient. Never sampled for training.',source_support_TRAIN_only=True,original_loss_weights_and_proposal_pools_unchanged=True)
    write(root/'margin/manifest.json',manifest)
    for arm,phase in [('original_mapped','native_A'),('source_margin002','margin002')]:
        src=previous/'quarter_distortion_raw126316'/phase/'renders/step-000126316';dst=root/'zero_references'/arm;shutil.copytree(src,dst/'renders');reference=dict(arm=arm,source_checkpoint_sha256=SOURCE_SHA,render_directory=str(remote/'zero_references'/arm/'renders'),render_sha256={p.name:sha(p)for p in(dst/'renders').glob('*.png')},metrics=read(dst/'renders/metrics.json'),artifacts_sha256={str(remote/'zero_references'/arm/'renders'/p.name):sha(p)for p in(dst/'renders').iterdir()if p.is_file()},inherited_readonly_complete_sha256=sha(previous/'quarter_distortion_raw126316/complete.json'))
        write(dst/'reference.json',reference)
    files={str(p.relative_to(root/'code')):sha(p)for p in(root/'code').rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix!='.pyc'}
    write(root/'source_manifest.json',dict(original1495_source_manifest_sha256=sha(parent/'source_manifest.json'),unchanged_original_files=len(original),files_sha256=files,added_helpers=sorted(set(files)-set(original)),changed_original_files=[]))
    bound={str(remote/'code'/p):d for p,d in files.items()};worker=remote/'code/LookCloser/scripts/train_marshawn_margin_continuation.py'
    for phase in ['zero','technical','production']:
        jobs=[]
        for arm in ARMS:
            dest=root/'specs'/f'{phase}_{arm}';reference=root/'zero_references'/arm/'reference.json'
            spec=dict(output=str(remote/'runs'/f'{phase}_{arm}'),parent_checkpoint=str(remote/'anchor/checkpoint.ckpt'),parent_checkpoint_sha256=SOURCE_SHA,parent_spec=str(remote/'anchor/spec.json'),parent_spec_sha256=sha(root/'anchor/spec.json'),anchor_audit=str(remote/'anchor_audit.json'),anchor_audit_sha256=sha(root/'anchor_audit.json'),mode=phase,arm=arm,release=str(remote/'specs'/f'{phase}_{arm}/root_release.json'),worker_sha256=sha(root/'code/LookCloser/scripts'/worker.name),frozen_source_hashes=bound,source_render_sha256={p.name:sha(p)for p in(root/'anchor/renders').glob('*.png')},margin_manifest=str(remote/'margin/manifest.json'),margin_manifest_sha256=sha(root/'margin/manifest.json'),stopping=STOP,density_selector=arm,zero_reference=str(remote/'zero_references'/arm/'reference.json'),zero_reference_sha256=sha(reference))
            write(dest/'spec.json',spec);write(dest/'root_release.template.json',dict(reviewer='root',technical_authorized=False,production_authorized=False,spec_sha256=sha(dest/'spec.json'),margin_manifest_sha256=spec['margin_manifest_sha256'],zero_reference_sha256=spec['zero_reference_sha256']))
            for segment in ['first32','resume32']if phase=='technical'else['']:
                name=f'margin126316_{phase}_{arm}'+('_'+segment if segment else'');command=[str(remote.parent/'python'),str(worker),'--spec',str(remote/'specs'/f'{phase}_{arm}/spec.json')]
                if segment=='first32':command+=['--stop-after-attempts','32']
                jobs.append(dict(name=name,host='silver',command=command,cwd=str(remote/'code/LookCloser'),env=dict(PYTHONPATH=str(remote/'code')+':'+str(remote/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',LOOKCLOSER_GPU_LEASE_RECEIPT=str(remote/'remote_queue/jobs'/name/'receipt.json')),dependencies=[f'margin126316_{phase}_{arm}_first32']if segment=='resume32'else[],progress_files=[str(remote/'runs'/f'{phase}_{arm}/progress.json')],metrics_file=str(remote/'runs'/f'{phase}_{arm}'/('latest.json'if segment=='first32'else'complete.json'))))
        plan=dict(schema_version=1,poll_seconds=30,hosts=dict(silver=dict(address='ubuntu@silver-sable',root=str(remote/'remote_queue'),python=str(remote.parent/'python'),controller=str(remote.parent/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=[0,1,2,3,5,6])),jobs=jobs);write(root/f'{phase}_plan.json',plan)
    shutil.copy2(Path(__file__).with_name('audit_marshawn_margin_continuation_cpu.py'),root/'audit_cpu.py')
    write(root/'preparation.json',dict(CPU_only=True,source_checkpoint_sha256=SOURCE_SHA,source_actual_Adam=anchor['actual_Adam'],remote_root=str(remote),local_root=str(root),immutable_parent_files=len(original),added_helpers=len(files)-len(original),GPU_started=False,releases_created=False,root_protocol_sha256=sha(root/'protocol.json'),production_fresh_parent=True,one_scientific_factor='SourcePad1+.002 density selector; original mapped traversal retained'))


def guardian(c,root,remote):
    remote=Path(remote);base=c/'rgb_confidence_continuation_candidate_v1/external_guard_rgb_v1';source=(base/'guardian.py').read_text();text=source.replace('external_guard_rgb_v1',GEN).replace("['original', 'bag6874', 'bag6874_delta2', 'bag6874_delta4']",repr(list(ARMS))).replace('rgb126316_production_','margin126316_production_');out=root/GEN;out.mkdir(exist_ok=False);(out/'guardian.py').write_text(text)
    (out/'guardian.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile=str(base/'guardian.py'),tofile=GEN+'/guardian.py')))
    policy=read(base/'policy.json');policy.update(local_root=str(root),remote_root=str(remote),worker_path=str(remote/'code/LookCloser/scripts/train_marshawn_margin_continuation.py'),guardian_sha256=sha(out/'guardian.py'),phase_plans={p:GEN+'/'+p+'_plan.json'for p in ['zero','technical','production']},root_receipts={'protocol.json':sha(root/'protocol.json'),'CPU_readiness.json':sha(root/'CPU_readiness.json')})
    for phase in policy['phase_plans']:shutil.copy2(root/(phase+'_plan.json'),out/(phase+'_plan.json'))
    bindings=[root/'source_manifest.json',root/'margin/manifest.json',root/'anchor_audit.json',root/'protocol.json',root/'CPU_readiness.json',*[p for p in(root/'specs').rglob('spec.json')],*[p for p in(root/'zero_references').rglob('reference.json')],*[out/(p+'_plan.json')for p in policy['phase_plans']],out/'guardian.py']
    policy['prepared_bindings']={str(p.relative_to(root)):sha(p)for p in bindings};write(out/'policy.json',policy)
    for phase in policy['phase_plans']:
        release=dict(reviewer='root',external_guard_authorized=False,policy_sha256=sha(out/'policy.json'),authorized_phases=[phase],protocol_sha256=sha(root/'protocol.json'),CPU_readiness_sha256=sha(root/'CPU_readiness.json'))
        if phase!='zero':release[('zero'if phase=='technical'else'technical')+'_verification_sha256']='ROOT_BIND_AFTER_PREDECESSOR_PASS'
        write(out/(phase+'_guard_release.template.json'),release)
    a={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)};b={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    keys=['identity','same','authorized_worker','event','fail','bounded_error','clear_errors','phase_status','signal_due','final_record','ssh_script','check_limits','require_phase_proof']
    for k in keys:assert a[k]==b[k],k
    write(out/'CPU_orchestration.json',dict(passed=True,unchanged_guard_core_functions=keys,inherited_guardian_sha256=sha(base/'guardian.py'),policy_sha256=sha(out/'policy.json'),worker_science_unchanged=True,launch_origin='LOCAL_OLIVE_ONLY',remote_guardian_only=True,production_deadline=None,GPU_started=False))
    print(json.dumps(dict(policy_sha256=sha(out/'policy.json'),guardian_sha256=sha(out/'guardian.py'),launch_entry=str(out/'guardian.py'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','guardian']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--remote',required=True);a=p.parse_args();globals()[a.phase](a.campaign,a.root,a.remote)
