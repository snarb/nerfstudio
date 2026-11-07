"""Build executable196-prefix three-arm CPU package; no GPU/releases/jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

ARMS=('control','tau001','tau01')
STOP=dict(minimum_attempts=4000,anchor_step=130316,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001,early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None)
SOURCE_SHA='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')


def prepare(c,out,remote):
    out.mkdir(parents=True,exist_ok=False); parent=c/'semantic_quarter_distortion_candidate_20261107';rgb=c/'rgb_confidence_continuation_candidate_v1'
    original=read(parent/'source_manifest.json')['files_sha256']
    shutil.copytree(parent/'code',out/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    helpers=['marshawn_artifact_prefix_factory.py','marshawn_artifact_prefix_rgb.py','marshawn_artifact_prefix_tau.py','train_marshawn_artifact_prefix.py',
             'train_marshawn_semantic_late_distortion.py','marshawn_gpu_isolation.py','audit_marshawn_artifact_prefix_CPU.py','verify_marshawn_artifact_prefix_phase.py']
    for name in helpers:
        target=out/'code/LookCloser/scripts'/name
        if target.exists():raise ValueError('Do not overwrite original source: '+name)
        shutil.copy2(Path(__file__).with_name(name),target)
    sealed=c/'quarter_seven_tau_candidate_v1/code/LookCloser/scripts/marshawn_clear_tau.py'
    shutil.copy2(sealed,out/'code/LookCloser/scripts/marshawn_artifact_prefix_sealed_tau.py')
    (out/'anchor').mkdir()
    for p in (rgb/'anchor').iterdir():
        if p.is_dir():shutil.copytree(p,out/'anchor'/p.name)
        elif p.name!='checkpoint.ckpt':shutil.copy2(p,out/'anchor'/p.name)
    rgb_remote=read(rgb/'preparation.json')['remote_root'];audit=read(rgb/'anchor_audit.json')
    # All seven source hash values remain exact; copied evidence paths use this generation.
    audit=json.loads(json.dumps(audit).replace(rgb_remote,remote));audit['upstream_anchor_audit_sha256']=sha(rgb/'anchor_audit.json')
    write(out/'anchor_audit.json',audit)
    (out/'confidence').mkdir();old_manifest=read(rgb/'confidence/manifest.json');records=[]
    for record in old_manifest['records']:
        name=record['camera']
        with np.load(rgb/'confidence'/record['archive'],allow_pickle=False)as z:w=z['original']
        target=out/'confidence'/f'{name}.npz';np.savez_compressed(target,original=w)
        records.append(dict(camera=name,archive=target.name,archive_sha256=sha(target),original_semantic_sha256=record['original_semantic_sha256'],
                            weights_sha256={'original':record['weights_sha256']['original']},proposal_means={'original':record['proposal_means']['original']}))
    write(out/'confidence/manifest.json',dict(kind='MATCHED_TRAIN63_ORIGINAL_RGB_W64_FOR_PREFIX196',uses_EVAL=False,records=records,
          parent_confidence_manifest_sha256=sha(rgb/'confidence/manifest.json'),RGB_weights_changed=False,sampling_pools_labels_FAS_changed=False))
    source=c/'actual_artifact_five_prefix_review_v1'
    for name in ['root_selected196_rays.json','root_four_prefix_qualification.json','proposal.json','floor_reference_context.json','actual_artifact_segment_coverage.json']:
        shutil.copy2(source/name,out/name)
    (out/'source_archives').mkdir()
    for name in ['AA','BA']:
        shutil.copy2(c/'native_semantic_grid_20000/occlusion_v2/W64/archives'/f'{name}.npz',out/'source_archives'/f'{name}.npz')
    files={str(p.relative_to(out/'code')):sha(p)for p in(out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'}
    assert all(files[k]==h for k,h in original.items())
    write(out/'source_manifest.json',dict(original1495_source_manifest_sha256=sha(parent/'source_manifest.json'),unchanged_original_files=len(original),files_sha256=files,added_helpers=sorted(set(files)-set(original)),changed_original_files=[]))
    write(out/'preparation.json',dict(CPU_only=True,source_checkpoint_sha256=SOURCE_SHA,remote_root=remote,local_root=str(out),
          source_checkpoint_remote=read(rgb/'preparation.json')['remote_source_checkpoint'],source_seven_state_proof_upstream=sha(rgb/'anchor_audit.json'),
          sealed_tau_source_sha256=sha(sealed),sealed_tau_copied_byte_identical=True,selected196_sha256=sha(out/'root_selected196_rays.json'),
          existing10481_overlap=14,new_rays=182,scope='Conditional soft prefixes only above empiricalZ.025. JA excluded; no hard cells/full-ray empty/alphaGT.',
          science_factors={'control':0.,'tau001':.001,'tau01':.01},new_factor='.01 versus proven .001; no integration/AMP/sigma changes',
          stopping=STOP,main_RGB='exact originalW64 and original proposal expected normalizer',density_selector='original_mapped',GPU_started=False,releases_created=False))
    remote_path=Path(remote);inventory=read(out/'source_manifest.json')['files_sha256'];bound={str(remote_path/'code'/p):h for p,h in inventory.items()}
    renders={p.name:sha(p)for p in(out/'anchor/renders').glob('*.png')}
    worker=out/'code/LookCloser/scripts/train_marshawn_artifact_prefix.py'
    for phase in ('zero','technical','production'):
        jobs=[]
        for arm in ARMS:
            sp=out/'specs'/f'{phase}_{arm}'/'spec.json'
            spec=dict(output=str(remote_path/'runs'/f'{phase}_{arm}'),parent_checkpoint=str(remote_path/'anchor/checkpoint.ckpt'),parent_checkpoint_sha256=SOURCE_SHA,
                parent_spec=str(remote_path/'anchor/spec.json'),parent_spec_sha256=sha(out/'anchor/spec.json'),anchor_audit=str(remote_path/'anchor_audit.json'),anchor_audit_sha256=sha(out/'anchor_audit.json'),
                mode=phase,arm=arm,release=str(remote_path/'specs'/f'{phase}_{arm}/root_release.json'),worker_sha256=sha(worker),frozen_source_hashes=bound,
                source_render_sha256=renders,confidence_manifest=str(remote_path/'confidence/manifest.json'),confidence_manifest_sha256=sha(out/'confidence/manifest.json'),
                targets=str(remote_path/'root_selected196_rays.json'),targets_sha256=sha(out/'root_selected196_rays.json'),root_qualification=str(remote_path/'root_four_prefix_qualification.json'),
                root_qualification_sha256=sha(out/'root_four_prefix_qualification.json'),stopping=STOP,density_selector='original_mapped')
            write(sp,spec);write(sp.with_name('root_release.template.json'),dict(reviewer='root',technical_authorized=False,production_authorized=False,
                  spec_sha256=sha(sp),qualified196_prefixes_reviewed=True,root_qualification_sha256=spec['root_qualification_sha256']))
            for segment in (('first32','resume32')if phase=='technical'else('',)):
                name=f'prefix126316_{phase}_{arm}'+('_'+segment if segment else'');run=remote_path/'runs'/f'{phase}_{arm}'
                command=[str(remote_path.parent/'python'),str(remote_path/'code/LookCloser/scripts/train_marshawn_artifact_prefix.py'),'--spec',str(remote_path/'specs'/f'{phase}_{arm}/spec.json')]
                if segment=='first32':command+=['--stop-after-attempts','32']
                jobs.append(dict(name=name,host='silver',command=command,cwd=str(remote_path/'code/LookCloser'),env=dict(PYTHONPATH=str(remote_path/'code')+':'+str(remote_path/'code/LookCloser/scripts'),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',LOOKCLOSER_GPU_LEASE_RECEIPT=str(remote_path/'remote_queue/jobs'/name/'receipt.json')),
                      dependencies=[f'prefix126316_{phase}_{arm}_first32']if segment=='resume32'else[],progress_files=[str(run/'progress.json')],metrics_file=str(run/('latest.json'if segment=='first32'else'complete.json'))))
        write(out/f'{phase}_plan.json',dict(schema_version=1,poll_seconds=30,hosts=dict(silver=dict(address='ubuntu@silver-sable',root=str(remote_path/'remote_queue'),python=str(remote_path.parent/'python'),
              controller=str(remote_path.parent/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'),gpus=[0,1,2,3,5,6])),jobs=jobs))
    write(out/'CURRENT_STATUS.json',dict(CPU_preparation='STAGED_AUDIT_PENDING',root_source196='QUALIFIED_CONDITIONAL_SOFT_PREFIX_ONLY',zero='NOT_RUN',technical='NOT_RUN',production='NOT_RELEASED',source_checkpoint_sha256=SOURCE_SHA,GPU_started=False))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--remote',required=True)
    a=p.parse_args();prepare(a.campaign,a.output,a.remote)
