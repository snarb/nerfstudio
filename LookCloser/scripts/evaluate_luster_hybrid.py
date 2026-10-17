"""Evaluation-only native person/ROI protocol. Never imported by the predictor."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from luster_hybrid_common import (ORIGIN, TARGETS, bootstrap, geometry, image, log,
                                  native_rows, read, sha, source_path, write)

# Source-inspected boxes in the historical HD coordinate system; fixed before candidates.
EXTRA = {
    7: dict(skin=[710,280,850,385], eyes=[715,237,853,300], hair=[650,155,925,410],
            jewelry=[660,55,950,420], embroidery=[600,750,820,900],
            cuffs=[880,630,1020,765], nails=[750,505,880,602], lace=[915,670,1000,745]),
    12:dict(skin=[702,300,738,370], eyes=None, hair=[430,235,650,1260],
            jewelry=[630,265,680,355], embroidery=[550,720,795,910],
            cuffs=[905,700,985,760], nails=[980,735,1060,805], lace=[920,707,970,757]),
    95:dict(skin=[540,445,650,510], eyes=[530,360,715,440], hair=[210,70,740,380],
            jewelry=[420,0,760,320], embroidery=[280,555,620,820],
            cuffs=[780,1180,1300,1370], nails=[800,1270,910,1390], lace=[785,1195,935,1275]),
    150:dict(skin=[0,625,115,710], eyes=[0,550,170,635], hair=[0,80,390,600],
             jewelry=[0,0,325,510], embroidery=[0,1060,155,1430],
             cuffs=[370,1580,650,1910], nails=[130,1850,300,1920], lace=[400,1810,650,1920]),
}


def protocol(root):
    dest=root/'evaluation/protocol.json'
    if dest.exists():return read(dest)
    masks=read(ORIGIN/'evaluation_protocol_v2/masks.json')
    rois=read(ORIGIN/'evaluation_protocol/rois.json')['boxes']
    extra=read(ORIGIN/'evaluation_protocol/extra_rois.json')['boxes']
    original=read(ORIGIN/'data/unmasked/transforms.json')
    rows=native_rows(); result={}
    for c in TARGETS:
        hd=next(r for r in original['frames'] if r['camera_id']==c)
        row=rows[c];sx=row['w']/hd['w'];sy=row['h']/hd['h']
        boxes={**rois[str(c)],**extra[str(c)],**EXTRA[c]}
        boxes['person']=[0,0,hd['w'],hd['h']]
        native={k:None if v is None else [max(0,int(np.floor(v[0]*sx))),max(0,int(np.floor(v[1]*sy))),
               min(row['w'],int(np.ceil(v[2]*sx))),min(row['h'],int(np.ceil(v[3]*sy)))] for k,v in boxes.items()}
        mask=next(r for r in masks['rows'] if r['camera']==c)
        assert sha(mask['source'])==mask['source_sha256'] and sha(mask['mask'])==mask['mask_sha256']
        result[str(c)]=dict(boxes=native,**mask)
    p=dict(usage='evaluation_only',resolution='native',created=time.time(),cameras=result,
           metrics='spatial AlexNet LPIPS / PSNR / SSIM; neutral .5 exterior; original masks v2',
           validation=[12,95,150],demonstration=[7],target_ratio=.60,
           code_sha256=sha(Path(__file__)))
    write(dest,p)
    return p


def compute(root, candidate, cameras, regions=None, panels=False, device='cuda', output_tag=None):
    bootstrap(root)
    from luster_quality_protocol import masked_metrics
    import lpips
    p=protocol(root)
    perceptual=lpips.LPIPS(net='alex',spatial=True).eval().to(device)
    result=[];coverage=[];started=time.monotonic()
    with torch.inference_mode():
        for c in cameras:
            base=geometry(root,c)['rgb']
            directory=root/'candidates'/candidate
            prediction=base if candidate=='baseline' else np.load(directory/f'cam{c:03d}.npy',mmap_mode='r')
            e=p['cameras'][str(c)]
            assert sha(e['source'])==e['source_sha256'] and sha(e['mask'])==e['mask_sha256']
            gt=np.asarray(Image.open(e['source']).convert('RGB'),np.float32)/255
            mask=np.asarray(Image.open(e['mask']))>0
            assert prediction.shape==gt.shape and mask.shape==gt.shape[:2]
            for name,box in e['boxes'].items():
                if regions and name not in regions:continue
                if box is None:
                    result.append(dict(camera=c,region=name,status='not_visible'));continue
                x0,y0,x1,y1=box
                # Keeping 16-pixel context reproduces the full-image fixed-support evaluator.
                x0p=max(0,x0-16);y0p=max(0,y0-16);x1p=min(mask.shape[1],x1+16);y1p=min(mask.shape[0],y1+16)
                support=mask[y0p:y1p,x0p:x1p].copy()
                inside=np.zeros_like(support);inside[y0-y0p:y1-y0p,x0-x0p:x1-x0p]=True;support&=inside
                if not support.any():
                    result.append(dict(camera=c,region=name,status='empty_support'));continue
                pred=np.array(prediction[y0p:y1p,x0p:x1p]);ref=gt[y0p:y1p,x0p:x1p]
                scores=masked_metrics(torch.from_numpy(pred).to(device),torch.from_numpy(ref).to(device),torch.from_numpy(support).to(device),perceptual)
                result.append(dict(camera=c,region=name,status='measured',**scores))
                changed=np.any(pred!=np.asarray(base[y0p:y1p,x0p:x1p]),axis=-1)
                coverage.append(dict(camera=c,region=name,changed_fraction=float(changed[support].mean()),preserved_fraction=float((~changed[support]).mean())))
                if panels:
                    panel=np.concatenate([gt[y0:y1,x0:x1],base[y0:y1,x0:x1],prediction[y0:y1,x0:x1]],axis=1)
                    out=root/'evaluation'/candidate;out.mkdir(exist_ok=True)
                    image(out/f'cam{c:03d}_{name}_GT_LookCloser_hybrid.png',panel)
                    im=Image.fromarray(np.rint(panel.clip(0,1)*255).astype(np.uint8));im.thumbnail((1500,900));im.save(out/f'cam{c:03d}_{name}_preview.jpg',quality=94)
            log(root,'metrics_camera',candidate=candidate,camera=c,person=next((r for r in result if r['camera']==c and r['region']=='person'),None))
    person=[r for r in result if r['camera'] in (12,95,150) and r['region']=='person' and r['status']=='measured']
    means={k:float(np.mean([r[k] for r in person])) for k in ('lpips','psnr','ssim')} if len(person)==3 else None
    out=dict(candidate=candidate,resolution='native',rows=result,coverage=coverage,validation_mean=means,
             protocol_sha256=sha(root/'evaluation/protocol.json'),seconds=time.monotonic()-started,
             code_sha256=sha(Path(__file__)),device=device)
    write(root/'evaluation'/f'{output_tag or candidate}_metrics.json',out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',default='baseline')
    p.add_argument('--cameras',type=int,nargs='+',default=list(TARGETS))
    p.add_argument('--regions',nargs='+')
    p.add_argument('--panels',action='store_true')
    p.add_argument('--protocol-only',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output-tag')
    a=p.parse_args()
    if a.protocol_only:protocol(a.root)
    else:compute(a.root,a.candidate,a.cameras,a.regions,a.panels,a.device,a.output_tag)
