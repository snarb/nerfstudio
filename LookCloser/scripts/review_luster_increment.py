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


def review(control, candidate, output, step):
    if output.exists():
        raise FileExistsError('Preserve previous reviews; choose a new directory')
    requests=[read(run/'request.json') for run in [control,candidate]]
    results=[read(run/'result.json') for run in [control,candidate]]
    for key in ['seed','data','independent_rng_streams']:
        if requests[0].get(key)!=requests[1].get(key):
            raise ValueError(f'Unmatched comparison: {key}')
    origin=requests[0].get('comparison_origin_checkpoint',requests[0].get('resume'))
    if requests[0].get('resume')!=requests[1].get('resume'):
        if not origin or origin!=requests[1].get('comparison_origin_checkpoint') or not all(
            descends_from(r.get('resume'),origin) for r in requests
        ):
            raise ValueError('Different parents without verified common checkpoint ancestry')
    if results[0]['transforms_sha256']!=results[1]['transforms_sha256']:
        raise ValueError('Different calibration or split')
    metrics=[read(run/f'eval_{step:06d}/metrics.json') for run in [control,candidate]]
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
            images=[Image.open(run/f'eval_{step:06d}'/name).convert('RGB')
                    for run in [control,candidate]]
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
    summary=dict(control=str(control),candidate=str(candidate),step=step,visual_review='pending',
        comparison_origin_checkpoint=origin,
        first32_pixel_rgb_batches_identical=True,transforms_sha256=results[0]['transforms_sha256'],
        full_metrics=[{k:m[f'eval_all_{k}'] for k in ['psnr','ssim','lpips']} for m in metrics],
        full_delta={k:metrics[1][f'eval_all_{k}']-metrics[0][f'eval_all_{k}'] for k in ['psnr','ssim','lpips']},
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
    a=p.parse_args()
    review(a.control,a.candidate,a.output,a.step)
