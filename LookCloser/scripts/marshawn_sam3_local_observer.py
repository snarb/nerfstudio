"""Canonical LOCAL native masks plus read-only current-person observation."""
from pathlib import Path
import argparse,os,shutil,sys
from types import SimpleNamespace
import marshawn_sam3_assist as io


def run(a):
    import numpy as np
    V=a.campaign.resolve();out=a.output_campaign.resolve() if a.output_campaign else V;code=V/'code'
    module=io.canonical(code);original_seed=module.seed;original_select=module.select;context={}
    overlay=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay')
    sys.path[:0]=[str(overlay/'src'),str(overlay/'deps')]
    checkpoint=overlay/'weights/sam3.pt'
    if io.sha(checkpoint)!='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e':raise ValueError('Wrong canonical SAM3 checkpoint')
    import torch
    from marshawn_gpu_isolation import require_controller_lease,verify_cuda
    gpu=require_controller_lease(os.environ['LOCAL_SAM3_LEASE_RECEIPT']);gpu.update(verify_cuda(gpu['expected_physical_uuid'],torch))
    os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']))
    if out!=V:
        for camera in a.cameras:
            destination=out/'sources'/str(a.frame)/camera;destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(V/'sources'/str(a.frame)/camera/'receipt.json',destination/'receipt.json')
    def seed_observer(campaign,frame,camera):
        if Path(campaign)!=out or frame!=a.frame or camera not in a.cameras:raise ValueError('Unexpected LOCAL seed IO')
        result=original_seed(V,frame,camera)
        context.update(camera=camera,reference_frame=result[1],reference_sha256=result[2])
        return result
    def select_observer(masks,reference):
        camera=context['camera'];p=V/'sources'/str(a.frame)/camera/'receipt.json';r=io.read(p)
        selected=stats=error=None
        try:selected,stats=original_select(masks,reference)
        except Exception as exc:error=str(exc)
        target=out/'frames'/('%06d'%a.frame)/'person_candidates';target.mkdir(parents=True,exist_ok=True)
        archive=target/(camera+'.npz');np.savez_compressed(archive,masks=masks)
        io.write(target/(camera+'.json'),dict(frame=a.frame,camera=camera,source_receipt_sha256=io.sha(p),
            raw_native_image_sha256=r['raw_native_image_sha256'],checkpoint_sha256=io.sha(checkpoint),
            canonical_mask_sha256=io.sha(module.__file__),observer_adapter_sha256=io.sha(__file__),
            archive_sha256=io.sha(archive),masks_sha256=io.array_sha(masks),selected=selected,identity=stats,
            identity_error=error,fighter_identity_status='UNQUALIFIED_HINT_ONLY',
            identity_reference_frame=context['reference_frame'],identity_reference_sha256=context['reference_sha256'],
            observer='Original seed/select called unchanged; native input saved read-only, no additional inference/RNG'))
        if error is not None:raise ValueError(error)
        return selected,stats
    module.seed=seed_observer;module.select=select_observer
    module.main(SimpleNamespace(root=out,frame=a.frame,cameras=a.cameras,checkpoint=checkpoint))
    io.write(out/'frames'/('%06d'%a.frame)/'local_observer_ready.json',dict(frame=a.frame,cameras=a.cameras,gpu=gpu,
        canonical_mask_sha256=io.sha(module.__file__),observer_adapter_sha256=io.sha(__file__),prepared_publication_allowed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--frame',type=int,required=True)
    p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--output-campaign',type=Path);run(p.parse_args())
