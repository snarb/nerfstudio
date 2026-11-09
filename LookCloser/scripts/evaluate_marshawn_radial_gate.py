"""Exact paused-checkpoint HD gate; float-domain radiometric comparisons only."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_float(scorer, gt, pred):
    """Reuse existing AlexNet/SSIM engine, retaining unquantized float RGB."""
    if gt.shape != pred.shape or min(gt.shape[:2]) < 64:
        raise ValueError('Matched native RGB crops >=64 pixels required')
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite metric input')
    torch = scorer.torch
    a, b = [torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].to(
        scorer.device, dtype=torch.float32) for x in [gt,pred]]
    with torch.inference_mode():
        mse = (a-b).square().mean()
        result = dict(psnr=float(-10*torch.log10(mse)) if float(mse)>0 else None,
                      ssim=float(scorer.ssim(a,b,data_range=1.)), lpips=float(scorer.lpips(a,b)))
    scorer.lpips.reset()
    return result


def png(path, array):
    Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8)).save(path)


def panel(path, images, labels, max_width=None):
    ims=[Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)) for x in images]
    if max_width:
        for im in ims:
            im.thumbnail((max_width, max_width))
    w,h=ims[0].size
    out=Image.new('RGB',(len(ims)*w,h+25),'#171717')
    draw=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(zip(ims,labels)):
        out.paste(im,(i*w,25));draw.text((i*w+3,5),label,fill='white')
    out.save(path,quality=95)


def run(a):
    plan=json.loads(a.plan.read_text())
    for path,digest in plan['bindings'].items():
        if sha(path)!=digest:
            raise ValueError('Bound evaluator/input changed '+path)
    gate=json.loads(Path(plan['gate_file']).read_text())
    checkpoint=Path(plan['checkpoint']).resolve()
    if checkpoint!=Path(gate['checkpoint']).resolve() or sha(checkpoint)!=gate['checkpoint_sha256']:
        raise ValueError('Exact fully-written gate checkpoint mismatch')
    proc=Path('/proc')/str(gate['worker_pid'])/'stat'
    if not a.allow_terminal:
        if not proc.exists() or proc.read_text().split()[2] not in ['T','t']:
            raise ValueError('Expected same worker SIGSTOP at gate; no automatic stop/CONT is performed')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from nerfstudio.utils import eval_utils
    from probe_marshawn_occupancy import digest_model
    import render_marshawn_previews as preview
    original=eval_utils.eval_setup
    captured={}
    def setup(*args,**kwargs):
        result=original(*args,**kwargs)
        pipe=result[1]
        expected=pipe.model.scene_aabb.reshape(2,3)
        for box in [pipe.model.field.aabb,pipe.model.occupancy_grid.aabbs[0].reshape(2,3),
                    pipe.datamanager.train_dataparser_outputs.scene_box.aabb,
                    pipe.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb]:
            torch.testing.assert_close(box,expected.to(box),rtol=0,atol=1e-6)
        captured.update(pipe=pipe,before=digest_model(pipe.model),
                        active=pipe.model.occupancy_grid.binaries.detach().cpu().clone())
        return result
    eval_utils.eval_setup=setup
    try:
        preview.run(SimpleNamespace(config=Path(plan['config']),checkpoint=checkpoint,
            output=a.output/'native_HD',data=None,code_root=Path(plan['code_root']),
            max_width=1920,chunk_size=512,camera=None,bbox=None,
            skip_strict_bounds=True,display_transform='identity'))
    finally:
        eval_utils.eval_setup=original
    pipe=captured['pipe']
    if digest_model(pipe.model)!=captured['before'] or not torch.equal(
            pipe.model.occupancy_grid.binaries.cpu(),captured['active']):
        raise ValueError('Read-only renderer changed field/grid; do not accept artifacts')
    # No density, occupancy or helper query is made after rendering.
    del pipe
    captured.clear()
    torch.cuda.empty_cache()
    from summarize_marshawn_campaign import RoiScorer
    from marshawn_radial_common_domain import prediction_in_old_scalar_domain
    scorer=RoiScorer(a.device)
    fit=json.loads(Path(plan['fit']).read_text())
    old_contract=json.loads(Path(plan['old_contract']).read_text())
    old_ev={r['camera']:r['exposure_correction_ev'] for r in old_contract['rows']}
    tf=json.loads((Path(plan['data'])/'transforms.json').read_text())
    frames={f['physical_camera']:f for f in tf['frames']}
    old_tf=json.loads((Path(plan['old_data'])/'transforms.json').read_text())
    old_frames={f['physical_camera']:f for f in old_tf['frames']}
    for name in ['own_predictions','common_old_predictions','raw_common','contacts']:
        (a.output/name).mkdir()
    rows=[]
    for camera in ['DC','HC','KA']:
        frame=frames[camera]
        z=np.load(a.output/'native_HD'/('eval_'+camera+'_buffers.npz'))
        pred,gt=z['rgb'],z['gt']
        loaded=np.asarray(Image.open(Path(plan['data'])/frame['file_path']).convert('RGB'),dtype=np.float32)/255.
        if pred.shape!=(1012,1920,3) or not np.array_equal(gt,loaded):
            raise ValueError('Wrong native GT/raster')
        common,saturated=prediction_in_old_scalar_domain(pred,frame,fit,old_ev[camera])
        common_gt,_=prediction_in_old_scalar_domain(gt,frame,fit,old_ev[camera])
        old_gt=np.asarray(Image.open(Path(plan['old_data'])/old_frames[camera]['file_path']).convert('RGB'),dtype=np.float32)/255.
        reference=np.load(Path(plan['reference16k'])/('eval_'+camera+'_buffers.npz'))
        if not np.array_equal(reference['gt'],old_gt):
            raise ValueError('Reference16k GT is not actual OLD scalar target')
        previous=reference['rgb']
        png(a.output/'own_predictions'/(camera+'.png'),pred)
        png(a.output/'common_old_predictions'/(camera+'.png'),common)
        np.savez_compressed(a.output/'raw_common'/(camera+'.npz'),prediction=common,
            GT_old=old_gt,inverse_new_GT=common_gt,reference16k=previous)
        row=dict(camera=camera,full_own=score_float(scorer,gt,pred),
            full_common_old=score_float(scorer,old_gt,common),
            full_reference16k_old=score_float(scorer,old_gt,previous),
            saturated_noninvertible_channels=saturated,
            new_GT_inverse_vs_old_codes=dict(mean=float(np.mean(abs(common_gt-old_gt))*255),
                p99=float(np.quantile(abs(common_gt-old_gt),.99)*255),max=float(np.max(abs(common_gt-old_gt))*255)),
            mean_opacity=float(z['opacity'].mean()),regions={})
        panel(a.output/'contacts'/(camera+'_own.jpg'),[gt,pred],['NEW radial GT','candidate OWN'],640)
        panel(a.output/'contacts'/(camera+'_common_old.jpg'),[old_gt,previous,common],
              ['OLD scalar GT','actual MVS16k OLD','candidate inverse -> OLD'],480)
        rois={'floor':[850,650,1450,980]}
        if camera=='KA':
            rois['black_wide']=[0,552,428,994]
        if camera=='DC':
            rois['old_camera_veil']=[1400,0,1920,450]
        for label,(x0,y0,x1,y1) in rois.items():
            crop=lambda image:image[y0:y1,x0:x1]
            row['regions'][label]=dict(bbox_xyxy=[x0,y0,x1,y1],
                own=score_float(scorer,crop(gt),crop(pred)),
                common_old=score_float(scorer,crop(old_gt),crop(common)),
                reference16k_old=score_float(scorer,crop(old_gt),crop(previous)))
            panel(a.output/'contacts'/(camera+'_'+label+'.jpg'),
                [crop(gt),crop(pred),crop(old_gt),crop(previous),crop(common)],
                ['NEW radial GT','candidate OWN','OLD scalar GT','MVS16k OLD','candidate inverse OLD'])
        rows.append(row)
        (a.output/'progress_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    averages={branch:{k:float(np.mean([r[branch][k] for r in rows])) for k in ['psnr','ssim','lpips']}
              for branch in ['full_own','full_common_old','full_reference16k_old']}
    receipt=dict(kind='READONLY_EXACT_NATIVE_HD_RADIAL_GATE',checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),plan_sha256=sha(a.plan),script_sha256=sha(__file__),
        original_field_and_actual_active_grid_unchanged=True,occupancy_dilation=False,
        raw_float_predictions=True,GT_fitting=False,GT_compositing=False,rows=rows,average=averages,
        metric_definition='RGB range1 FP32 PSNR; torchmetrics SSIM data_range1; AlexNet LPIPS normalizeTrue; full native raster and unchanged rectangular crops; no masks/resizing',
        selection='Formal selector remains native CSV own-corrected all3; common-old metrics only compare runs',
        caveats=['Actual MVS16k differs in initialization/history/near/prior/MVS loss; comparison is not radial-only causality.',
                 'Inverse mapping on quantized NEW GT differs from OLD GT; errors measured above, no refit.',
                 'Fully saturated display channels cannot be uniquely inverted; counts reported.'])
    (a.output/'report.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(averages),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--allow-terminal',action='store_true',help='Explicitly allow a terminal worker; checkpoint is still hash-bound')
    run(p.parse_args())
