"""External portable I/O adapter for the SEALED canonical native SAM3 worker.

No frozen campaign file is edited. Model setup, detection, selection, component
qualification and native refinement execute the canonical main() unchanged.
The sole module override replaces seed-file I/O with exact precomputed inputs.
"""
from pathlib import Path
import argparse,hashlib,importlib,json,os,shutil,sys,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def array_sha(value):
    import numpy as np
    value=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str(value.dtype).encode());h.update(json.dumps(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def canonical(code):
    code=Path(code).resolve();sys.path[:0]=[str(code),str(code/'LookCloser/scripts')]
    module=importlib.import_module('marshawn_video_prep_masks')
    if Path(module.__file__).resolve()!=code/'LookCloser/scripts/marshawn_video_prep_masks.py':raise ValueError('Canonical worker import escaped frozen code')
    return module


def pack(campaign,frame,camera,task_root,remote_task,remote_code,remote_overlay,canary=False):
    import numpy as np
    campaign=Path(campaign).resolve();out=Path(task_root).resolve();out.mkdir(parents=True,exist_ok=True)
    module=canonical(campaign/'code');receipt_path=campaign/'sources'/str(frame)/camera/'receipt.json';receipt=read(receipt_path)
    image=Path(receipt['raw_native_image'])
    if receipt['frame']!=frame or receipt['camera']!=camera or sha(image)!=receipt['raw_native_image_sha256']:raise ValueError('Wrong native source')
    reference,previous,reference_sha=module.seed(campaign,frame,camera)
    if frame-previous>60 or previous>=frame:raise ValueError('Stale/nonprior identity reference')
    for actor in ('black','grey'):
        if reference[actor].shape!=(270,512) or reference[actor].dtype!=np.bool_:raise ValueError('Canonical 512x270 bool identity reference required')
    old=campaign/'frames'/('%06d'%frame)/'native_masks'/(camera+'.json')
    if canary:
        record=read(old)
        if record['identity_reference_frame']!=previous or record['identity_reference_sha256']!=reference_sha:
            raise ValueError('Current canonical seed differs from LOCAL canary reference')
        if record['source_receipt_sha256']!=sha(receipt_path) or record['mask_code_sha256']!=sha(module.__file__):raise ValueError('Canary source/helper differs')
    blob=out/'reference.npz'
    if not blob.exists():np.savez_compressed(blob,**reference)
    with np.load(blob,allow_pickle=False) as stored:
        if any(not np.array_equal(stored[a],reference[a]) for a in reference):raise ValueError('Immutable identity blob changed')
    original=out/'source_original_receipt.json'
    if not original.exists():shutil.copy2(receipt_path,original)
    if sha(original)!=sha(receipt_path):raise ValueError('Immutable original receipt changed')
    # Only the native PNG is large. Dereference a local input link when staging.
    local_png=out/'raw_native.png'
    if not local_png.exists():local_png.symlink_to(image)
    remote=Path(remote_task);portable=dict(receipt,raw_native_image=str(remote/'raw_native.png'))
    write(out/'source_portable_receipt.json',portable)
    spec=dict(schema=1,frame=frame,camera=camera,remote_task=str(remote),source_code=remote_code,
        overlay=remote_overlay,canonical_mask_sha256=sha(module.__file__),
        cleanup_sha256=sha(campaign/'code/LookCloser/scripts/marshawn_video_prep_mask_components.py'),
        code_seal_sha256=sha(campaign/'code_seal.json'),checkpoint_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
        original_source_receipt_sha256=sha(original),portable_source_receipt_sha256=sha(out/'source_portable_receipt.json'),
        raw_native_image_sha256=sha(image),raw_sha256=receipt['raw_sha256'],
        reference=dict(frame=previous,archive_sha256=reference_sha,blob_sha256=sha(blob),
                       arrays_sha256={a:array_sha(reference[a]) for a in reference}),
        adapter_sha256=sha(__file__),capture_person_candidates=True,canary=canary,no_NeRF_state_loaded=True)
    if (out/'task.json').exists() and read(out/'task.json')!=spec:raise ValueError('Immutable assist task changed')
    write(out/'task.json',spec);return spec


def validate_task(spec,root):
    import numpy as np
    root=Path(root)
    for name,expected in [('source_original_receipt.json',spec['original_source_receipt_sha256']),
                          ('source_portable_receipt.json',spec['portable_source_receipt_sha256']),
                          ('raw_native.png',spec['raw_native_image_sha256']),
                          ('reference.npz',spec['reference']['blob_sha256'])]:
        if sha(root/name)!=expected:raise ValueError('Assist input changed: '+name)
    original=read(root/'source_original_receipt.json');portable=read(root/'source_portable_receipt.json')
    if {k:v for k,v in original.items() if k!='raw_native_image'}!={k:v for k,v in portable.items() if k!='raw_native_image'}:
        raise ValueError('Portable source adapter changed scientific source metadata')
    if portable['raw_native_image']!=str(Path(spec['remote_task'])/'raw_native.png'):raise ValueError('Portable source image points elsewhere')
    reference={}
    with np.load(root/'reference.npz',allow_pickle=False) as source:
        for actor in ('black','grey'):
            value=source[actor].copy()
            if value.shape!=(270,512) or value.dtype!=np.bool_ or array_sha(value)!=spec['reference']['arrays_sha256'][actor]:
                raise ValueError('Identity input tensor changed')
            reference[actor]=value
    return reference


def remote_run(spec_path):
    from types import SimpleNamespace
    spec=read(spec_path);root=Path(spec['remote_task'])
    if sha(__file__)!=spec['adapter_sha256']:raise ValueError('Assist adapter code changed')
    reference=validate_task(spec,root);code=Path(spec['source_code'])
    if sha(code.parent/'code_seal.json')!=spec['code_seal_sha256']:raise ValueError('Wrong frozen campaign release')
    if sha(code/'LookCloser/scripts/marshawn_video_prep_masks.py')!=spec['canonical_mask_sha256']:raise ValueError('Canonical model/mask math changed')
    if sha(code/'LookCloser/scripts/marshawn_video_prep_mask_components.py')!=spec['cleanup_sha256']:raise ValueError('Canonical component math changed')
    module=canonical(code);checkpoint=Path(spec['overlay'])/'weights/sam3.pt'
    if sha(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('SAM3 checkpoint differs')
    overlay=Path(spec['overlay']);sys.path[:0]=[str(overlay/'src'),str(overlay/'deps')]
    import torch
    from marshawn_gpu_isolation import verify_cuda,require_controller_lease
    gpu=require_controller_lease(os.environ['SAM3_ASSIST_LEASE_RECEIPT']);gpu.update(verify_cuda(gpu['expected_physical_uuid'],torch))
    if 'LOOKCLOSER_GPU_LEASE_FD' not in os.environ:raise ValueError('Inherited global GPU lease FD required')
    os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']))
    portable_campaign=root/'campaign';source=portable_campaign/'sources'/str(spec['frame'])/spec['camera'];source.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'source_portable_receipt.json',source/'receipt.json')
    def seed_io(campaign,frame,camera):
        if Path(campaign)!=portable_campaign or frame!=spec['frame'] or camera!=spec['camera']:raise ValueError('Identity blob requested for another view/frame')
        return {a:v.copy() for a,v in reference.items()},spec['reference']['frame'],spec['reference']['archive_sha256']
    module.seed=seed_io
    # Read-only I/O observer: preserve the original select arguments/returns and
    # every array value, while retaining all same-instant people for crew audit.
    candidate_root=portable_campaign/'frames'/('%06d'%spec['frame'])/'person_candidates'
    candidate_root.mkdir(parents=True,exist_ok=True)
    original_select=module.select
    def select_observer(masks,reference_inputs):
        import numpy as np
        selected,stats=original_select(masks,reference_inputs)
        archive=candidate_root/(spec['camera']+'.npz')
        np.savez_compressed(archive,masks=masks)
        write(candidate_root/(spec['camera']+'.json'),dict(frame=spec['frame'],camera=spec['camera'],
            source_receipt_sha256=spec['portable_source_receipt_sha256'],
            original_source_receipt_sha256=spec['original_source_receipt_sha256'],
            raw_native_image_sha256=spec['raw_native_image_sha256'],
            checkpoint_sha256=spec['checkpoint_sha256'],canonical_mask_sha256=spec['canonical_mask_sha256'],
            archive_sha256=sha(archive),masks_sha256=array_sha(masks),selected=selected,identity=stats,
            identity_reference_frame=spec['reference']['frame'],identity_reference_sha256=spec['reference']['archive_sha256'],
            observer='original select called unchanged; full native input masks saved read-only; no extra inference or RNG'))
        return selected,stats
    module.select=select_observer
    start=time.time();module.main(SimpleNamespace(root=portable_campaign,frame=spec['frame'],cameras=[spec['camera']],checkpoint=checkpoint))
    result=portable_campaign/'frames'/('%06d'%spec['frame'])/'native_masks';record=read(result/(spec['camera']+'.json'))
    import numpy as np
    with np.load(result/(spec['camera']+'.npz'),allow_pickle=False) as z:arrays={k:array_sha(z[k]) for k in z.files}
    proof=dict(schema=1,frame=spec['frame'],camera=spec['camera'],task_sha256=sha(spec_path),
        canonical_mask_sha256=spec['canonical_mask_sha256'],cleanup_sha256=spec['cleanup_sha256'],
        adapter_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),
        original_source_receipt_sha256=spec['original_source_receipt_sha256'],portable_source_receipt_sha256=sha(source/'receipt.json'),
        raw_native_image_sha256=sha(root/'raw_native.png'),reference_blob_sha256=sha(root/'reference.npz'),
        identity_reference_frame=spec['reference']['frame'],identity_reference_sha256=spec['reference']['archive_sha256'],
        archive_sha256=sha(result/(spec['camera']+'.npz')),remote_mask_receipt_sha256=sha(result/(spec['camera']+'.json')),
        arrays_sha256=arrays,gpu=gpu,torch_version=torch.__version__,seconds=time.time()-start,
        operation='Only paths/seed input IO and readonly candidate observation adapted; canonical math unchanged')
    candidates=candidate_root/(spec['camera']+'.npz');candidate_receipt=candidate_root/(spec['camera']+'.json')
    proof['person_candidates']=dict(archive_sha256=sha(candidates),receipt_sha256=sha(candidate_receipt))
    if record['source_receipt_sha256']!=proof['portable_source_receipt_sha256']:raise ValueError('Worker used a different portable source receipt')
    write(root/'operation_receipt.json',proof);return proof


def compare(campaign,task_root):
    import numpy as np
    root=Path(task_root);spec=read(root/'task.json');remote=root/'returned';proof=read(remote/'operation_receipt.json')
    camera=spec['camera'];local=Path(campaign)/'frames'/('%06d'%spec['frame'])/'native_masks';result=remote/'native_masks'
    expected=dict(task_sha256=sha(root/'task.json'),canonical_mask_sha256=spec['canonical_mask_sha256'],
        cleanup_sha256=spec['cleanup_sha256'],checkpoint_sha256=spec['checkpoint_sha256'],adapter_sha256=spec['adapter_sha256'],
        original_source_receipt_sha256=spec['original_source_receipt_sha256'],portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],
        raw_native_image_sha256=spec['raw_native_image_sha256'],reference_blob_sha256=spec['reference']['blob_sha256'])
    for key,value in expected.items():
        if proof[key]!=value:raise ValueError('Remote operation mapping proof differs: '+key)
    if sha(result/(camera+'.npz'))!=proof['archive_sha256'] or sha(result/(camera+'.json'))!=proof['remote_mask_receipt_sha256']:
        raise ValueError('Returned result bytes changed')
    record=read(result/(camera+'.json'));old=read(local/(camera+'.json'));arrays={}
    with np.load(result/(camera+'.npz'),allow_pickle=False) as got,np.load(local/(camera+'.npz'),allow_pickle=False) as reference:
        if set(got.files)!=set(reference.files):raise ValueError('Canonical output schema differs')
        for key in got.files:
            a,b=got[key],reference[key]
            if array_sha(a)!=proof['arrays_sha256'][key]:raise ValueError('Result array proof differs')
            arrays[key]=dict(equal=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)))
    fields=['identity','component_qualification','tiles','native_cleanup_version','mask_code_sha256','checkpoint_sha256','identity_reference_frame','identity_reference_sha256']
    metadata={key:old[key]==record[key] for key in fields}
    report=dict(frame=spec['frame'],camera=camera,arrays=arrays,metadata_equal=metadata,
        all_equal=all(r['equal'] for r in arrays.values()) and all(metadata.values()),
        source_png_sha_equal=old['raw_native_image_sha256']==proof['raw_native_image_sha256'],
        canary_local_source_receipt_sha256=old['source_receipt_sha256'],
        original_source_receipt_sha256=proof['original_source_receipt_sha256'],
        nominal_direct_comparison='not stored by original LOCAL run; compare refined arrays and identity/component/tile receipts',
        remote_operation_receipt_sha256=sha(remote/'operation_receipt.json'))
    write(root/'comparison.json',report);return report


def collect(campaign,task_root):
    """Publish only absent outputs, retaining original remote receipt unchanged."""
    import fcntl
    root=Path(task_root);spec=read(root/'task.json');returned=root/'returned';proof=read(returned/'operation_receipt.json')
    # Reuse proof validation against an existing LOCAL result for canaries only.
    camera=spec['camera'];destination=Path(campaign)/'frames'/('%06d'%spec['frame'])/'native_masks';destination.mkdir(parents=True,exist_ok=True)
    lock=(destination/(camera+'.assist.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (destination/(camera+'.json')).exists():raise ValueError('Existing LOCAL output is never overwritten by assist')
    source=returned/'native_masks';record=read(source/(camera+'.json'))
    for key in ('canonical_mask_sha256','cleanup_sha256','checkpoint_sha256','adapter_sha256','raw_native_image_sha256'):
        if proof[key]!=spec[key]:raise ValueError('Untrusted remote operation: '+key)
    if proof['task_sha256']!=sha(root/'task.json') or proof['reference_blob_sha256']!=spec['reference']['blob_sha256']:raise ValueError('Remote identity/source mapping differs')
    if sha(source/(camera+'.npz'))!=proof['archive_sha256'] or sha(source/(camera+'.json'))!=proof['remote_mask_receipt_sha256']:raise ValueError('Returned mask changed')
    original_receipt=Path(campaign)/'sources'/str(spec['frame'])/camera/'receipt.json'
    if sha(original_receipt)!=spec['original_source_receipt_sha256']:raise ValueError('LOCAL source changed during assist')
    if record['source_receipt_sha256']!=spec['portable_source_receipt_sha256'] or record['mask_code_sha256']!=spec['canonical_mask_sha256']:raise ValueError('Canonical remote worker source differs')
    provenance=destination/'assist_provenance';provenance.mkdir(exist_ok=True)
    shutil.copy2(source/(camera+'.json'),provenance/(camera+'.remote_mask.json'))
    shutil.copy2(returned/'operation_receipt.json',provenance/(camera+'.operation.json'))
    if proof.get('person_candidates'):
        candidate_source=returned/'person_candidates';candidate_target=destination.parent/'person_candidates';candidate_target.mkdir(exist_ok=True)
        if sha(candidate_source/(camera+'.npz'))!=proof['person_candidates']['archive_sha256'] or sha(candidate_source/(camera+'.json'))!=proof['person_candidates']['receipt_sha256']:
            raise ValueError('Fresh person candidates changed')
        candidate_record=read(candidate_source/(camera+'.json'))
        if candidate_record['source_receipt_sha256']!=spec['portable_source_receipt_sha256']:raise ValueError('Candidate source mapping differs')
        shutil.copy2(candidate_source/(camera+'.json'),provenance/(camera+'.remote_candidates.json'))
        shutil.copy2(candidate_source/(camera+'.npz'),candidate_target/(camera+'.npz'))
        write(candidate_target/(camera+'.json'),dict(candidate_record,source_receipt_sha256=spec['original_source_receipt_sha256'],
            assist_mapping=dict(remote_candidate_receipt_sha256=proof['person_candidates']['receipt_sha256'],
                operation_receipt_sha256=sha(returned/'operation_receipt.json'),
                portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],
                original_source_receipt_sha256=spec['original_source_receipt_sha256'])))
    mapped=dict(record,source_receipt_sha256=spec['original_source_receipt_sha256'],
        assist_mapping=dict(original_remote_receipt_sha256=proof['remote_mask_receipt_sha256'],
            operation_receipt_sha256=sha(returned/'operation_receipt.json'),
            portable_source_receipt_sha256=spec['portable_source_receipt_sha256'],
            original_source_receipt_sha256=spec['original_source_receipt_sha256'],
            mapping='Only source receipt IO path changes; image SHA and exact identity inputs equal'))
    temporary=destination/(camera+'.assist.partial.npz');shutil.copy2(source/(camera+'.npz'),temporary)
    if sha(temporary)!=record['archive_sha256']:raise ValueError('Copied mask changed')
    temporary.replace(destination/(camera+'.npz'));write(destination/(camera+'.json'),mapped)
    return dict(published=True,frame=spec['frame'],camera=camera,archive_sha256=record['archive_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','run','compare','collect','check'])
    p.add_argument('--campaign',type=Path);p.add_argument('--frame',type=int);p.add_argument('--camera')
    p.add_argument('--task-root',type=Path);p.add_argument('--remote-task');p.add_argument('--remote-code');p.add_argument('--remote-overlay');p.add_argument('--spec',type=Path);p.add_argument('--canary',action='store_true');a=p.parse_args()
    if a.action=='pack':result=pack(a.campaign,a.frame,a.camera,a.task_root,a.remote_task,a.remote_code,a.remote_overlay,a.canary)
    elif a.action=='run':result=remote_run(a.spec)
    elif a.action=='compare':result=compare(a.campaign,a.task_root)
    elif a.action=='collect':result=collect(a.campaign,a.task_root)
    else:
        spec=read(a.spec);validate_task(spec,Path(a.spec).parent);result={'CPU_contract':'PASS'}
    print(json.dumps(result),flush=True)
