#!/usr/bin/env python3
"""Prepare matched incremental-run metrics and GT/control/candidate crop sheets."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def descends_from(checkpoint, origin):
    seen=set()
    while checkpoint is not None:
        path=Path(checkpoint).resolve()
        if path==Path(origin).resolve():
            return path.is_file()
        if path in seen:
            raise ValueError('Cyclic checkpoint ancestry')
        seen.add(path)
        checkpoint=read(path.parents[4]/'request.json').get('resume')
    return False


def review(control, candidate, output, step, control_eval=None, candidate_eval=None, common_origin=None):
    if output.exists():
        raise FileExistsError('Preserve previous reviews; choose a new directory')
    requests=[read(run/'request.json') for run in [control,candidate]]
    results=[read(run/'result.json') for run in [control,candidate]]
    for key in ['seed','data','independent_rng_streams','deterministic_torch']:
        if requests[0].get(key)!=requests[1].get(key):
            raise ValueError(f'Unmatched comparison: {key}')
    origin=requests[0].get('comparison_origin_checkpoint',requests[0].get('resume'))
    if common_origin is not None:
        origin=str(common_origin.resolve())
        if not all(descends_from(r.get('resume'),origin) for r in requests):
            raise ValueError('Explicit common origin is not an ancestor of both checkpoints')
    elif requests[0].get('resume')!=requests[1].get('resume'):
        if not origin or origin!=requests[1].get('comparison_origin_checkpoint') or not all(
            descends_from(r.get('resume'),origin) for r in requests
        ):
            raise ValueError('Different parents without verified common checkpoint ancestry')
    if results[0]['transforms_sha256']!=results[1]['transforms_sha256']:
        raise ValueError('Different calibration or split')
    evaluation_dirs=[control_eval or control/f'eval_{step:06d}',candidate_eval or candidate/f'eval_{step:06d}']
    metrics=[read(path/'metrics.json') for path in evaluation_dirs]
    for key in ['near_plane','eval_use_training_near_plane','adaptive_coarse_step_size',
                'max_steps_per_ray','corrected_arm_allocator']:
        effective=[m.get('evaluation_model',{}).get(key,(r.get('model') or {}).get(key))
                   for m,r in zip(metrics,requests)]
        if effective[0]!=effective[1]:
            raise ValueError(f'Different evaluation settings: {key}')
    if any(m['step']!=step for m in metrics):
        raise ValueError('Evaluation step differs from requested comparison')
    if any(m['eval_stride']!=1 for m in metrics):
        raise ValueError('Quality comparisons require native resolution')
    keys=[[(v['split'],v['index'],v['image']) for v in m['per_view']] for m in metrics]
    if keys[0]!=keys[1] or len(keys[0])!=6:
        raise ValueError('Require the same three held-out and three train-review views')
    receipts=[[json.loads(l) for l in (run/'ray_batch_receipts.jsonl').read_text().splitlines()]
              for run in [control,candidate]]
    if len(receipts[0])!=32 or len(receipts[1])!=32 or not all(
        all(a[k]==b[k] for k in ['step','indices_sha256','rgb_sha256'])
        for a,b in zip(*receipts)
    ):
        raise ValueError('First32 pixel/RGB batches do not match')
    output.mkdir(parents=True)
    labels=['face','hand','hair','clothing']
    deltas=[]
    for a,b in zip(metrics[0]['per_view'],metrics[1]['per_view']):
        deltas.append(dict(split=a['split'],image=a['image'],
            rois={label:{k:b['rois'][label][k]-a['rois'][label][k]
                         for k in ['psnr','ssim','lpips']} for label in labels}))
    for label in labels:
        sheet=Image.new('RGB',(780,225*6),'#222222');draw=ImageDraw.Draw(sheet)
        for row,(split,index,_) in enumerate(keys[0]):
            name=f'{split}_{index:03d}_{label}.png'
            images=[Image.open(path/name).convert('RGB') for path in evaluation_dirs]
            if images[0].size!=images[1].size or images[0].width%2:
                raise ValueError(f'Different crop dimensions: {name}')
            width,height=images[0].size
            if not np.array_equal(np.asarray(images[0])[:,:width//2],np.asarray(images[1])[:,:width//2]):
                raise ValueError(f'Different GT: {name}')
            panel=Image.new('RGB',(width*3//2,height));panel.paste(images[0],(0,0))
            panel.paste(images[1].crop((width//2,0,width,height)),(width,0))
            panel.thumbnail((770,200));sheet.paste(panel,(5,row*225+22))
            draw.text((5,row*225+3),name+' GT | control | candidate',fill='white')
        sheet.save(output/f'{label}.jpg')
    interior=[]
    sheet=Image.new('RGB',(780,225*6),'#222222');draw=ImageDraw.Draw(sheet)
    for row,(split,index,_) in enumerate(keys[0]):
        name=f'{split}_{index:03d}_face.png'
        panels=[Image.open(path/name).convert('RGB') for path in evaluation_dirs]
        width,height=panels[0].size;face_width=width//2
        box=(face_width//4,height//4,3*face_width//4,3*height//4)
        gt=panels[0].crop(box)
        predictions=[p.crop((box[0]+face_width,box[1],box[2]+face_width,box[3])) for p in panels]
        psnr=[]
        for prediction in predictions:
            mse=np.mean(((np.asarray(prediction,dtype=float)-np.asarray(gt,dtype=float))/255)**2)
            psnr.append(float(-10*np.log10(max(mse,1e-12))))
        interior.append(dict(split=split,index=index,relative_box=box,control_psnr=psnr[0],
                             candidate_psnr=psnr[1],delta_psnr=psnr[1]-psnr[0]))
        panel=Image.new('RGB',(gt.width*3,gt.height))
        for i,im in enumerate([gt]+predictions):panel.paste(im,(i*gt.width,0))
        panel.thumbnail((770,200));sheet.paste(panel,(5,row*225+22))
        draw.text((5,row*225+3),name+' inner face GT | control | candidate',fill='white')
    sheet.save(output/'face_interior.jpg')
    summary=dict(control=str(control),candidate=str(candidate),step=step,visual_review='pending',
        evaluation_directories=[str(path) for path in evaluation_dirs],
        comparison_origin_checkpoint=origin,
        first32_pixel_rgb_batches_identical=True,transforms_sha256=results[0]['transforms_sha256'],
        full_metrics=[{k:m[f'eval_all_{k}'] for k in ['psnr','ssim','lpips']} for m in metrics],
        full_delta={k:metrics[1][f'eval_all_{k}']-metrics[0][f'eval_all_{k}'] for k in ['psnr','ssim','lpips']},
        inner_face_8bit_psnr=interior,
        inner_face_scope='Supplemental fixed central half of original face ROI; saved8bit RGB; no masks; not checkpoint selection',
        per_view_roi_deltas=deltas,worker_seconds_including_eval=[r['elapsed'] for r in results],
        note='Prepared evidence only; visual acceptance and repeat robustness require separate review')
    (output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['full_metrics','full_delta','visual_review']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,default=8000)
    p.add_argument('--control-eval',type=Path)
    p.add_argument('--candidate-eval',type=Path)
    p.add_argument('--common-origin',type=Path,
                   help='Verify an older shared checkpoint when declared branch origins differ')
    a=p.parse_args()
    review(a.control,a.candidate,a.output,a.step,a.control_eval,a.candidate_eval,a.common_origin)
