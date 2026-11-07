"""Current-frame person candidates only, using the immutable SAM3 main prefix.

Canonical model construction and global set_image/set_text_prompt run unchanged.
A read-only select observer saves the native prediction and stops before native
boundary refinement. No canonical LOCAL matte is written, replaced or adopted.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time
from types import SimpleNamespace
import marshawn_sam3_assist as io


class Captured(Exception):pass


def pack(campaign,frame,camera,task,remote_task,remote_code,remote_overlay):
    spec=io.pack(campaign,frame,camera,task,remote_task,remote_code,remote_overlay)
    spec.update(mode='PERSON_CANDIDATES_ONLY',candidate_adapter_sha256=io.sha(__file__))
    path=Path(task)/'candidate_task.json'
    if path.exists() and io.read(path)!=spec:raise ValueError('Immutable candidate task changed')
    io.write(path,spec);return spec


def run(spec_path):
    import numpy as np
    spec=io.read(spec_path);root=Path(spec['remote_task']);code=Path(spec['source_code'])
    if io.sha(__file__)!=spec['candidate_adapter_sha256'] or io.sha(io.__file__)!=spec['adapter_sha256']:raise ValueError('External adapter source changed')
    reference=io.validate_task(spec,root)
    if io.sha(code.parent/'code_seal.json')!=spec['code_seal_sha256']:raise ValueError('Frozen code seal differs')
    module=io.canonical(code)
    if io.sha(module.__file__)!=spec['canonical_mask_sha256']:raise ValueError('Canonical mask worker differs')
    if io.sha(code/'LookCloser/scripts/marshawn_video_prep_mask_components.py')!=spec['cleanup_sha256']:raise ValueError('Canonical component helper differs')
    checkpoint=Path(spec['overlay'])/'weights/sam3.pt'
    if io.sha(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('SAM3 checkpoint differs')
    overlay=Path(spec['overlay']);sys.path[:0]=[str(overlay/'src'),str(overlay/'deps')]
    import torch
    from marshawn_gpu_isolation import verify_cuda,require_controller_lease
    gpu=require_controller_lease(os.environ['SAM3_ASSIST_LEASE_RECEIPT']);gpu.update(verify_cuda(gpu['expected_physical_uuid'],torch))
    os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']))
    portable=root/'campaign';source=portable/'sources'/str(spec['frame'])/spec['camera'];source.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'source_portable_receipt.json',source/'receipt.json')
    def seed_io(campaign,frame,camera):
        if Path(campaign)!=portable or frame!=spec['frame'] or camera!=spec['camera']:raise ValueError('Wrong candidate seed request')
        return {a:v.copy() for a,v in reference.items()},spec['reference']['frame'],spec['reference']['archive_sha256']
    module.seed=seed_io;original_select=module.select;out=root/'person_candidates';out.mkdir(exist_ok=True)
    def capture(masks,reference_inputs):
        # Selection is an unqualified hint only; failure must never discard the
        # actual current-frame global person prediction needed for identity QA.
        try:selected,stats=original_select(masks,reference_inputs);error=None
        except Exception as exc:selected=None;stats=None;error=str(exc)
        archive=out/(spec['camera']+'.npz');np.savez_compressed(archive,masks=masks)
        io.write(out/(spec['camera']+'.json'),dict(frame=spec['frame'],camera=spec['camera'],
            source_receipt_sha256=spec['portable_source_receipt_sha256'],
            original_source_receipt_sha256=spec['original_source_receipt_sha256'],
            raw_native_image_sha256=spec['raw_native_image_sha256'],raw_sha256=spec['raw_sha256'],
            checkpoint_sha256=spec['checkpoint_sha256'],canonical_mask_sha256=spec['canonical_mask_sha256'],
            archive_sha256=io.sha(archive),masks_sha256=io.array_sha(masks),shape=list(masks.shape),selected=selected,
            identity=stats,identity_error=error,fighter_identity_status='UNQUALIFIED_HINT_ONLY',
            identity_reference_frame=spec['reference']['frame'],identity_reference_sha256=spec['reference']['archive_sha256'],
            no_NeRF_state_loaded=True,native_refinement_skipped=True,
            observer='Global native person prediction captured unchanged at canonical select input; no matte publication'))
        raise Captured()
    module.select=capture;started=time.time()
    try:module.main(SimpleNamespace(root=portable,frame=spec['frame'],cameras=[spec['camera']],checkpoint=checkpoint))
    except Captured:pass
    else:raise ValueError('Global candidate observer did not execute')
    path=out/(spec['camera']+'.npz');receipt=out/(spec['camera']+'.json')
    proof=dict(schema=1,frame=spec['frame'],camera=spec['camera'],mode='PERSON_CANDIDATES_ONLY',task_sha256=io.sha(spec_path),
        candidate_adapter_sha256=io.sha(__file__),adapter_sha256=io.sha(io.__file__),
        canonical_mask_sha256=spec['canonical_mask_sha256'],cleanup_sha256=spec['cleanup_sha256'],checkpoint_sha256=io.sha(checkpoint),
        original_source_receipt_sha256=spec['original_source_receipt_sha256'],portable_source_receipt_sha256=io.sha(source/'receipt.json'),
        raw_native_image_sha256=io.sha(root/'raw_native.png'),reference_blob_sha256=io.sha(root/'reference.npz'),
        archive_sha256=io.sha(path),receipt_sha256=io.sha(receipt),gpu=gpu,torch_version=torch.__version__,seconds=time.time()-started,
        native_mattes_published=False,prepared_publication_allowed=False)
    io.write(root/'operation_receipt.json',proof);return proof


def verify(task):
    import numpy as np
    task=Path(task);spec=io.read(task/'candidate_task.json');returned=task/'returned';proof=io.read(returned/'operation_receipt.json')
    for key,value in dict(mode='PERSON_CANDIDATES_ONLY',frame=spec['frame'],camera=spec['camera'],
        task_sha256=io.sha(task/'candidate_task.json'),candidate_adapter_sha256=spec['candidate_adapter_sha256'],adapter_sha256=spec['adapter_sha256'],
        canonical_mask_sha256=spec['canonical_mask_sha256'],cleanup_sha256=spec['cleanup_sha256'],checkpoint_sha256=spec['checkpoint_sha256'],
        original_source_receipt_sha256=spec['original_source_receipt_sha256'],portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],
        raw_native_image_sha256=spec['raw_native_image_sha256'],reference_blob_sha256=spec['reference']['blob_sha256'],
        native_mattes_published=False,prepared_publication_allowed=False).items():
        if proof.get(key)!=value:raise ValueError('Candidate operation source proof differs: '+key)
    camera=spec['camera'];source=returned/'person_candidates';record=io.read(source/(camera+'.json'))
    if io.sha(source/(camera+'.npz'))!=proof['archive_sha256'] or io.sha(source/(camera+'.json'))!=proof['receipt_sha256']:raise ValueError('Candidate artifact SHA differs')
    for key,value in dict(frame=spec['frame'],camera=camera,source_receipt_sha256=spec['portable_source_receipt_sha256'],
        raw_native_image_sha256=spec['raw_native_image_sha256'],checkpoint_sha256=spec['checkpoint_sha256'],
        canonical_mask_sha256=spec['canonical_mask_sha256'],identity_reference_frame=spec['reference']['frame'],
        identity_reference_sha256=spec['reference']['archive_sha256']).items():
        if record.get(key)!=value:raise ValueError('Candidate record source differs')
    with np.load(source/(camera+'.npz'),allow_pickle=False) as z:
        if z['masks'].dtype!=np.bool_ or z['masks'].shape[1:]!=(4320,8192) or io.array_sha(z['masks'])!=record['masks_sha256']:raise ValueError('Native candidate tensor differs')
    return proof


def collect(campaign,task):
    task=Path(task);spec=io.read(task/'candidate_task.json');proof=verify(task);camera=spec['camera']
    source=task/'returned/person_candidates';target=Path(campaign)/'frames'/('%06d'%spec['frame'])/'person_candidates_sg';target.mkdir(exist_ok=True)
    original=Path(campaign)/'sources'/str(spec['frame'])/camera/'receipt.json'
    if io.sha(original)!=spec['original_source_receipt_sha256']:raise ValueError('LOCAL original source changed')
    record=io.read(source/(camera+'.json'))
    if (target/(camera+'.json')).exists():
        old=io.read(target/(camera+'.json'))
        if old['archive_sha256']!=proof['archive_sha256'] or io.sha(target/(camera+'.npz'))!=proof['archive_sha256']:raise ValueError('Existing SG candidates changed')
        return dict(reused_verified_candidate=True,frame=spec['frame'],camera=camera)
    provenance=target/'provenance';provenance.mkdir(exist_ok=True)
    shutil.copy2(source/(camera+'.json'),provenance/(camera+'.remote.json'));shutil.copy2(task/'returned/operation_receipt.json',provenance/(camera+'.operation.json'))
    shutil.copy2(source/(camera+'.npz'),target/(camera+'.npz'))
    io.write(target/(camera+'.json'),dict(record,source_receipt_sha256=spec['original_source_receipt_sha256'],
        assist_mapping=dict(remote_candidate_receipt_sha256=proof['receipt_sha256'],operation_receipt_sha256=io.sha(task/'returned/operation_receipt.json'),
            portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],original_source_receipt_sha256=spec['original_source_receipt_sha256'])))
    return dict(published_candidate_only=True,frame=spec['frame'],camera=camera)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','run','collect','verify'])
    p.add_argument('--campaign',type=Path);p.add_argument('--frame',type=int);p.add_argument('--camera');p.add_argument('--task-root',type=Path)
    p.add_argument('--remote-task');p.add_argument('--remote-code');p.add_argument('--remote-overlay');p.add_argument('--spec',type=Path);a=p.parse_args()
    if a.action=='pack':value=pack(a.campaign,a.frame,a.camera,a.task_root,a.remote_task,a.remote_code,a.remote_overlay)
    elif a.action=='run':value=run(a.spec)
    elif a.action=='collect':value=collect(a.campaign,a.task_root)
    else:value=verify(a.task_root)
    print(json.dumps(value),flush=True)
