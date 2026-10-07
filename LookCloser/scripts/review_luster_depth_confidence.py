#!/usr/bin/env python3
"""Save native RGB/depth/validity crops for the measured-depth visual gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import colormaps


def review(data, confidence, request_path):
    manifest=json.loads((confidence/'manifest.json').read_text())
    request=json.loads(request_path.read_text())
    meta=json.loads((data/'transforms.json').read_text())
    train=set(meta['train_filenames'])
    frames={Path(r['file_path']).name:r for r in meta['frames'] if r['file_path'] in train}
    receipts=[]
    previews=[]
    for name,boxes in request['roi_boxes'].items():
        if name not in manifest['maps'] or name not in frames:
            continue
        rgb=np.asarray(Image.open(data/frames[name]['file_path']).convert('RGB'))
        with np.load(confidence/manifest['maps'][name]['path'],allow_pickle=False) as p:
            depth=p['depth'];valid=p['valid'];uncertainty=p['uncertainty']
        for label,box in boxes.items():
            x0,y0,x1,y1=box
            d=depth[y0:y1,x0:x1];v=valid[y0:y1,x0:x1]
            measured=np.isfinite(d)&(d>0)
            if measured.any():
                low,high=np.quantile(d[measured],[.02,.98])
                tone=(d-low)/max(high-low,1e-6)
            else:
                low=high=0.;tone=np.zeros_like(d)
            color=np.rint(colormaps['viridis'](np.clip(tone,0,1))[...,:3]*255).astype('uint8')
            color[~measured]=0
            trusted=color.copy();trusted[~v]=0
            panels=np.concatenate([rgb[y0:y1,x0:x1],color,trusted],axis=1)
            crop=Image.fromarray(panels)
            crop.save(confidence/f'{Path(name).stem}_{label}_review.png')
            if label=='face':
                crop.thumbnail((990,380));previews.append((name,crop))
            receipts.append(dict(image=name,roi=label,box=box,measured_fraction=float(measured.mean()),
                valid_fraction=float(v.mean()),depth_color_range=[float(low),float(high)],
                median_relative_uncertainty=float(np.median(uncertainty[y0:y1,x0:x1][v]/d[v])) if v.any() else None))
    contact=Image.new('RGB',(1000,sum(im.height+30 for _,im in previews)),(25,25,25))
    draw=ImageDraw.Draw(contact);offset=0
    for name,im in previews:
        draw.text((5,offset+5),f'{name}: RGB | measured camera-z | trusted only',fill='white')
        contact.paste(im,(0,offset+25));offset+=im.height+30
    contact.save(confidence/'face_review.jpg')
    (confidence/'roi_coverage.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(dict(faces=[r for r in receipts if r['roi']=='face'],review=str(confidence/'face_review.jpg'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--confidence',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    review(args.data,args.confidence,args.request)
