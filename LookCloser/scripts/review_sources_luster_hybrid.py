"""Evaluation-only selected-photo provenance panels; never used in construction."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from luster_hybrid_common import geometry,read,write,sha

def review(root,name,regions):
    protocol=read(root/'evaluation/protocol.json');out=root/'evaluation'/f'{name}_sources';out.mkdir(exist_ok=True)
    rows=[];candidate=root/'candidates'/name
    for c in [7,12,95,150]:
        e=protocol['cameras'][str(c)];assert sha(e['source'])==e['source_sha256']
        gt=np.asarray(Image.open(e['source']).convert('RGB'));base=geometry(root,c)['rgb']
        pred=np.load(candidate/f'cam{c:03d}.npy',mmap_mode='r');labels=np.load(candidate/f'cam{c:03d}_sources.npy',mmap_mode='r');weights=np.load(candidate/f'cam{c:03d}_confidence.npy',mmap_mode='r')
        receipt=read(candidate/f'cam{c:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root))
        for region in regions:
            box=e['boxes'].get(region)
            if box is None:continue
            x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];lab=labels[sl];weight=weights[sl]
            texture=np.full((*lab.shape,3),.5,np.float32)
            for cid in np.unique(lab):
                if cid<0:continue
                d=warp_root/'warps'/f'cam{c:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');wx0,wy0,wx1,wy1=r['box']
                ix0,iy0=max(x0,wx0),max(y0,wy0);ix1,iy1=min(x1,wx1),min(y1,wy1)
                local=np.s_[iy0-y0:iy1-y0,ix0-x0:ix1-x0];chosen=lab[local]==cid
                rgb=np.load(d/'rgb.npy',mmap_mode='r')[iy0-wy0:iy1-wy0,ix0-wx0:ix1-wx0]
                texture[local][chosen]=rgb[chosen]
            def uint(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)
            palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32)
            colors=palette[np.maximum(lab,0)];colors[lab<0]=0
            tiles=[gt[sl],uint(base[sl]),uint(pred[sl]),uint(texture),uint(colors)]
            titles=['GT','LookCloser','hybrid','selected photo RGB','source IDs; black=fallback']
            # Preserve native scale in the PNG; preview scales all panels equally.
            h,w=lab.shape;panel=Image.new('RGB',(3*w,2*(h+24)),(25,25,25));draw=ImageDraw.Draw(panel)
            for i,(tile,title) in enumerate(zip(tiles,titles)):
                xx=(i%3)*w;yy=(i//3)*(h+24);panel.paste(Image.fromarray(tile),(xx,yy+24));draw.text((xx+3,yy+4),title,fill='white')
            panel.paste(Image.fromarray(uint(np.repeat(weight[...,None],3,-1))),(2*w,h+48));draw.text((2*w+3,h+28),'transfer weight',fill='white')
            panel.save(out/f'cam{c:03d}_{region}.png');panel.thumbnail((1500,1000));panel.save(out/f'cam{c:03d}_{region}_preview.jpg',quality=95)
            ids,counts=np.unique(lab,return_counts=True)
            rows.append(dict(camera=c,region=region,box=box,fallback_fraction=float((lab<0).mean()),full_weight_fraction=float((weight>=.999).mean()),sources={str(int(i)):int(n) for i,n in zip(ids,counts)},receipt_sha256=sha(candidate/f'cam{c:03d}_receipt.json')))
    write(out/'receipt.json',dict(usage='evaluation_only',candidate=name,code_sha256=sha(Path(__file__)),rows=rows));print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--regions',nargs='+',default=['nails','jewelry','embroidery']);a=p.parse_args();review(a.root,a.name,a.regions)
