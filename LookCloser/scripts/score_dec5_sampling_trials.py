#!/usr/bin/env python3
"""Held-out hair/face crops for sealed, equal-step native sampling trials."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-cameras',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    meta=json.loads((a.data/'transforms.json').read_text());frames=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    refs={f['physical_camera']:f for f in json.loads(a.reference_cameras.read_text())['frames']}
    metric=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval();rows=[]
    for arm in ['control','boundary65','frequency','frequency_boundary65']:
        for step in [7999,9999]:
            evaluation=a.root/arm/f'eval_{step:06d}'
            if not evaluation.exists():continue
            for index,frame in enumerate(frames):
                archive=evaluation/f'{index}.npz'
                if not archive.exists():continue
                prediction=np.load(archive)['rgb'];gt=cv2.imread(str(a.data/frame['file_path']))[::3,::3,::-1].astype(np.float32)/255
                if prediction.shape!=gt.shape:raise ValueError('Native evaluation stride mismatch')
                ref=refs[frame['physical_camera']];np.testing.assert_allclose(frame['transform_matrix'],ref['transform_matrix'],atol=1e-10,rtol=0)
                probability=np.load(a.reference/(frame['physical_camera']+'.npz'))['probability'].astype(np.float32)/65535
                h,w=gt.shape[:2];y,x=np.mgrid[:h,:w];u=((x*3+.5-frame['cx'])/frame['fl_x']*ref['fl_x']+ref['cx']-.5).astype(np.float32)
                v=((y*3+.5-frame['cy'])/frame['fl_y']*ref['fl_y']+ref['cy']-.5).astype(np.float32)
                values=cv2.remap(probability,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                for label,channel in [('hair',0),('face',1)]:
                    mask=values[...,channel]>.5;y0,x0=np.nonzero(mask)
                    if len(x0)<100:raise ValueError('Insufficient reference hair/face evidence')
                    yy=slice(max(0,y0.min()-4),min(h,y0.max()+5));xx=slice(max(0,x0.min()-4),min(w,x0.max()+5))
                    target=gt[yy,xx];local=mask[yy,xx];pred=np.where(local[...,None],prediction[yy,xx],target)
                    t,g=[torch.as_tensor(np.ascontiguousarray(z),device='cuda').permute(2,0,1)[None] for z in [pred,target]]
                    mse=np.square(prediction[mask]-gt[mask]).mean()
                    with torch.no_grad():lp=float(metric(t,g));ss=float(structural_similarity_index_measure(t,g,data_range=1.))
                    rows.append(dict(arm=arm,step=step,camera=frame['physical_camera'],roi=label,
                        psnr=float(-10*np.log10(max(mse,1e-12))),ssim=ss,lpips=lp,pixels=int(mask.sum()),
                        prediction_sha256=sha(archive)))
    summary=[]
    for arm in sorted({r['arm'] for r in rows}):
        for step in [7999,9999]:
            for roi in ['hair','face']:
                values=[r for r in rows if r['arm']==arm and r['step']==step and r['roi']==roi]
                if len(values)==3:summary.append(dict(arm=arm,step=step,roi=roi,**{k:float(np.mean([r[k] for r in values])) for k in ['psnr','ssim','lpips']}))
    atomic_json(a.output,dict(summary=summary,per_view=rows,metric_domain='stride3 native prediction; GT-composited masked tight hair/face crop for SSIM/LPIPS; PSNR only on semantic pixels',
        reference_cameras_sha256=sha(a.reference_cameras),uses_validation_rgb_for_metrics_only=True,
        checkpoint_selection_unchanged=True,prototype_geometry_improvement_not_established=True))


if __name__=='__main__':main()
