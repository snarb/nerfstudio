"""Prepare matched four-arm confidence continuation; never authorize or launch GPU work."""
import argparse,difflib,hashlib,json,os,shutil
from pathlib import Path
import numpy as np
from analyze_marshawn_soft_rgb_profiles import components

ARMS=('original','bag6874','bag6874_delta2','bag6874_delta4')
SOURCE_SHA='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
STOP=dict(minimum_attempts=4000,anchor_step=130316,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001,early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def asha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(c,out,remote):
    out.mkdir(parents=True,exist_ok=False);parent=c/'semantic_quarter_distortion_candidate_20261107';inventory=json.loads((parent/'source_manifest.json').read_text())['files_sha256']
    shutil.copytree(parent/'code',out/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel,d in inventory.items():assert sha(out/'code'/rel)==d,rel
    helpers=['marshawn_rgb_confidence_policy.py','marshawn_rgb_confidence_factory.py','train_marshawn_rgb_confidence.py','train_marshawn_semantic_late_distortion.py','marshawn_gpu_isolation.py','audit_marshawn_rgb_confidence_anchor.py','audit_marshawn_rgb_confidence_package_cpu.py','verify_marshawn_rgb_confidence_phase.py']
    for name in helpers:
        p=out/'code/LookCloser/scripts'/name
        if p.exists():raise ValueError('New helper unexpectedly shadows original source: '+name)
        shutil.copy2(Path(__file__).with_name(name),p)
    profile=c/'soft_rgb_band_profiles_CPU_v1';summary=json.loads((profile/'summary.json').read_text());source={r['camera']:r for r in summary['rows']};levels=np.load(profile/'frequency_levels.npz');proof=c/'native_semantic_grid_20000/bag_qualification/train_bag_qualification.json';bags={r['camera']:r for r in json.loads(proof.read_text())['records']};records=[];(out/'confidence').mkdir()
    for name,record in source.items():
        sempath=c/'data/semantics'/f'{name}.npz';assert sha(sempath)==record['main_semantic_sha256']
        with np.load(sempath)as z:sem={k:z[k]for k in ('W64','core','band','valid','black','grey')}
        with np.load(profile/'profiles'/f'{name}.npz')as z:mult2=z['multiplier_delta2'];mult4=z['multiplier_delta4']
        original=sem['W64'];bag=original.copy();ex=np.zeros(original.shape,bool)
        if name in bags:
            b=bags[name];assert sha(b['exclusion_HD'])==b['exclusion_sha256']and sha(b['unknown_HD'])==b['unknown_sha256'];ex=np.load(b['exclusion_HD']);shell=np.load(b['unknown_HD']);assert not(ex&shell).any()and not(ex&sem['black']).any();bag[ex]=0
        weights=dict(original=original,bag6874=bag,bag6874_delta2=bag*mult2,bag6874_delta4=bag*mult4);archive=out/'confidence'/f'{name}.npz';np.savez_compressed(archive,**weights)
        records.append(dict(camera=name,archive=archive.name,archive_sha256=sha(archive),original_semantic_sha256=sha(sempath),source_RAW_sha256=record['source_raw_sha256'],source_profile_archive_sha256=record['profile_archive_sha256'],qualified_interior_pixels=int(ex.sum()),weights_sha256={k:asha(w)for k,w in weights.items()},proposal_means={k:components(w,sem,levels[name]).tolist()for k,w in weights.items()}))
    assert sum(r['qualified_interior_pixels']for r in records)==6874
    manifest=dict(kind='MATCHED_TRAIN63_RGB_CONFIDENCE_6874_RAW_DELTA2_DELTA4',uses_EVAL=False,arms=list(ARMS),records=records,source_profile_summary_sha256=sha(profile/'summary.json'),source_qualification_sha256=sha(proof),source_extra352_review_sha256=sha(profile/'bag_interior_review_v2/review.json'),root_extra352_manual_review='PENDING_NOT_AUTHORIZATION',profile='CurrentW64 times .1+.9exp(-d_out^2/(2delta^2)); original RAWsignedEDT calibrated centre map, delta2/4HD; existing qualified6874 interior RGBzero regardless main grey label. Original pools/labels/FAS unchanged.',fixed_density_selector='original_mapped',data_ready_sha256=sha(c/'data/ready.json'),transforms_sha256=sha(c/'data/transforms.json'))
    write(out/'confidence/manifest.json',manifest)
    (out/'anchor').mkdir();archive=c/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin'
    for name in ('config.yml','spec.json','source_manifest.json','grid.json','grid.npz','archive_receipt.json'):shutil.copy2(archive/name,out/'anchor'/name)
    shutil.copytree(archive/'renders',out/'anchor/renders')
    files={str(p.relative_to(out/'code')):sha(p)for p in(out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    assert all(files[k]==d for k,d in inventory.items());write(out/'source_manifest.json',dict(original1495_source_manifest_sha256=sha(parent/'source_manifest.json'),unchanged_original_files=len(inventory),files_sha256=files,added_helpers=sorted(set(files)-set(inventory)),changed_original_files=[]))
    write(out/'preparation.json',dict(CPU_only=True,source_checkpoint_sha256=SOURCE_SHA,remote_root=remote,local_root=str(out),remote_source_checkpoint=str(Path(remote).parent/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/checkpoint.ckpt'),profile_manifest_sha256=sha(out/'confidence/manifest.json'),source_code_unchanged=True,stopping=STOP,density_selector='original_mapped',GPU_started=False,releases_created=False))
    print(out)


def finalize(out,remote):
    audit=out/'anchor_audit.json';assert audit.exists();remote=Path(remote);inventory=json.loads((out/'source_manifest.json').read_text())['files_sha256'];bound={str(remote/'code'/name):d for name,d in inventory.items()};renders={p.name:sha(p)for p in(out/'anchor/renders').glob('*.png')if p.name in{c+x for c in('DC','HC','KA')for x in('.png','_alpha.png')}};worker=remote/'code/LookCloser/scripts/train_marshawn_rgb_confidence.py';controller=remote.parent/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'
    for phase in ('zero','technical','production'):
        jobs=[]
        for arm in ARMS:
            specroot=out/'specs'/f'{phase}_{arm}';spec=dict(output=str(remote/'runs'/f'{phase}_{arm}'),parent_checkpoint=str(remote/'anchor/checkpoint.ckpt'),parent_checkpoint_sha256=SOURCE_SHA,parent_spec=str(remote/'anchor/spec.json'),parent_spec_sha256=sha(out/'anchor/spec.json'),anchor_audit=str(remote/'anchor_audit.json'),anchor_audit_sha256=sha(audit),mode=phase,arm=arm,release=str(remote/'specs'/f'{phase}_{arm}/root_release.json'),worker_sha256=sha(out/'code/LookCloser/scripts/train_marshawn_rgb_confidence.py'),frozen_source_hashes=bound,source_render_sha256=renders,confidence_manifest=str(remote/'confidence/manifest.json'),confidence_manifest_sha256=sha(out/'confidence/manifest.json'),stopping=STOP,density_selector='original_mapped');write(specroot/'spec.json',spec);write(specroot/'root_release.template.json',dict(reviewer='root',technical_authorized=False,production_authorized=False,spec_sha256=sha(specroot/'spec.json'),qualified6874_source_reviewed=False,confidence_manifest_sha256=spec['confidence_manifest_sha256']))
            phases=('first32','resume32')if phase=='technical'else('',)
            for segment in phases:
                name=f'rgb126316_{phase}_{arm}'+('_'+segment if segment else'');command=[str(remote.parent/'python'),str(worker),'--spec',str(remote/'specs'/f'{phase}_{arm}/spec.json')]
                if segment=='first32':command+=['--stop-after-attempts','32']
                run=remote/'runs'/f'{phase}_{arm}'
                jobs.append(dict(name=name,host='silver',command=command,cwd=str(remote/'code/LookCloser'),env=dict(PYTHONPATH=str(remote/'code')+':'+str(remote/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',LOOKCLOSER_GPU_LEASE_RECEIPT=str(remote/'remote_queue/jobs'/name/'receipt.json')),dependencies=[f'rgb126316_{phase}_{arm}_first32']if segment=='resume32'else[],progress_files=[str(run/'progress.json')],metrics_file=str(run/('latest.json'if segment=='first32'else'complete.json'))))
        # Dispatcher and reused phase guardian execute on silver itself; no self-SSH.
        plan=dict(schema_version=1,poll_seconds=30,hosts=dict(silver=dict(root=str(remote/'remote_queue'),python=str(remote.parent/'python'),controller=str(controller),gpus=list(range(8)))),jobs=jobs);write(out/f'{phase}_plan.json',plan)
    original=Path(__file__).with_name('run_marshawn_focused_geometry_phase.py').read_text();guard=original.replace('diagnostic','production').replace("root/'specs'/(phase+'_'+arm+'.json')","root/'specs'/(phase+'_'+arm)/'spec.json'").replace('train_marshawn_focused_clothing_geometry.py','train_marshawn_rgb_confidence.py').replace('or age>limit','or (limit is not None and age>limit)')
    guard=guard.replace("    policy = read(root/'phase_policy.json')","    if not str(root.resolve()).startswith('/cache/ubuntu/lookcloser/'):\n        raise ValueError('Run reused phase guardian from remote SC; no local GPU/self-SSH ambiguity')\n    if not read(root/'CPU_readiness.json').get('passed'):\n        raise ValueError('CPU preparation gate required')\n    policy = read(root/'phase_policy.json')")
    (out/'phase_runner.py').write_text(guard);(out/'resource_guard_reuse.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),guard.splitlines(True),fromfile='existing_focused_phase_guard',tofile='confidence_phase_guard')))
    bindings={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and(p.parent.name.startswith(('zero_','technical_','production_'))or p.name in('manifest.json','anchor_audit.json','source_manifest.json','zero_plan.json','technical_plan.json','production_plan.json','phase_runner.py'))and not p.name.endswith('.template.json')}
    write(out/'phase_policy.json',dict(arms=list(ARMS),controller=str(controller),max_seconds_per_worker=dict(zero=900,technical=900,production=None),bindings_sha256={str(remote/rel):d for rel,d in bindings.items()},resource_guard='Reused focused phase guard; only phase/arm/path parameterization. Separate from scientific worker; production has no wall/iteration quality cap.',parent_guard_sha256=sha(Path(__file__).with_name('run_marshawn_focused_geometry_phase.py'))))
    write(out/'readiness_status.json',dict(source_bound_prepared=True,CPU_readiness='PENDING',zero_FHD_seven_states='PENDING_NOT_RUN',technical32_plus32_Adam='PENDING_NOT_RUN',root_extra352_source_review='PENDING',production_authorized=False,sourcepoint_selector_added=False,production_must_start_fresh_parent=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','finalize']);p.add_argument('--campaign',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--remote',required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.campaign,a.output,a.remote)
    else:finalize(a.output,a.remote)
