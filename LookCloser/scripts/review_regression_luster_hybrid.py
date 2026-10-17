"""Evaluation-only review of the largest local LPIPS regressions, beyond named ROIs."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from luster_hybrid_common import geometry,read,write
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
protocol=read(a.root/'evaluation/protocol.json');out=a.root/'evaluation'/f'{a.name}_spatial';records=[]
for c in [7,12,95,150]:
    maps=np.load(out/f'cam{c:03d}_lpips_maps.npz');base_map=maps['baseline'];hybrid_map=maps['hybrid'];mask=maps['mask'];x0,y0,x1,y1=maps['crop'];tiles=[]
    for y in range(0,mask.shape[0]-127,64):
        for x in range(0,mask.shape[1]-127,64):
            sl=np.s_[y:y+128,x:x+128];m=mask[sl]
            if m.sum()<2048:continue
            delta=float((hybrid_map[sl]-base_map[sl])[m].mean())
            if delta>0:tiles.append((delta,x,y))
    chosen=[]
    for item in sorted(tiles,reverse=True):
        _,x,y=item
        if all(abs(x-xx)>=128 or abs(y-yy)>=128 for _,xx,yy in chosen):chosen.append(item)
        if len(chosen)==6:break
    gt=np.asarray(Image.open(protocol['cameras'][str(c)]['source']).convert('RGB'));base=geometry(a.root,c)['rgb'];pred=np.load(a.root/f'candidates/{a.name}/cam{c:03d}.npy',mmap_mode='r')
    canvas=Image.new('RGB',(768,280*len(chosen)),(25,25,25));draw=ImageDraw.Draw(canvas)
    for i,(delta,x,y) in enumerate(chosen):
        sl=np.s_[y+y0:y+y0+128,x+x0:x+x0+128]
        panel=np.concatenate([gt[sl],np.rint(base[sl].clip(0,1)*255).astype(np.uint8),np.rint(pred[sl].clip(0,1)*255).astype(np.uint8)],axis=1)
        panel=Image.fromarray(panel).resize((768,256),Image.Resampling.NEAREST)
        canvas.paste(panel,(0,280*i+24));draw.text((4,280*i+4),f'GT | LookCloser | hybrid; box {x+x0},{y+y0}; LPIPS delta {delta:+.5f}',fill='white')
        records.append(dict(camera=c,box=[int(x+x0),int(y+y0),int(x+x0+128),int(y+y0+128)],lpips_delta=delta))
    canvas.save(out/f'cam{c:03d}_worst_local_regressions.png')
write(out/'regression_tiles.json',dict(usage='evaluation_only_not_prediction',rows=records))
