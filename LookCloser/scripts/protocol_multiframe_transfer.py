"""Freeze per-frame native evaluation support; never imported by transfer."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from luster_hybrid_common import FRAME,TARGETS,read,write,sha,source_path


def prepare(root):
    path=root/'evaluation/protocol.json'
    context=read(root/'context.json')
    original=read(context['original_metrics'])
    cameras={}
    for cid in TARGETS:
        measured=next(row for row in original['rows'] if row['camera']==cid)
        mp=Path(measured['mask'])
        assert sha(mp)==measured['mask_sha256']
        source=source_path(cid)
        assert sha(source)==measured['source_sha256']
        mask=np.asarray(Image.open(mp).convert('L'))>0
        with Image.open(source) as im:
            w,h=im.size
        assert mask.shape==(h,w) and mask.any()
        yy,xx=np.nonzero(mask)
        x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
        boxes={'person':[0,0,w,h]}
        # Evaluation-only spatial regions follow each frame's own support extent.
        # They are not claimed to be anatomical segmentations.
        for r in range(5):
            for c in range(3):
                boxes[f'spatial_r{r}_c{c}']=[
                    x0+(x1-x0)*c//3,y0+(y1-y0)*r//5,
                    x0+(x1-x0)*(c+1)//3,y0+(y1-y0)*(r+1)//5]
        cameras[str(cid)]=dict(boxes=boxes,source=str(source),source_sha256=sha(source),
            mask=str(mp),mask_sha256=sha(mp))
    result=dict(usage='evaluation_only',frame=FRAME,resolution='native',cameras=cameras,
        metrics='spatial AlexNet LPIPS / PSNR / SSIM; neutral .5 exterior; native per-frame video SAM support',
        validation=[12,95,150],demonstration=[7],target_ratio=.70,
        code_sha256=sha(Path(__file__)))
    if path.exists():
        assert read(path)==result,'Evaluation protocol changed'
    else:
        write(path,result)
    print(f'Native evaluation protocol frozen for {FRAME:06d}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    prepare(p.parse_args().root)
