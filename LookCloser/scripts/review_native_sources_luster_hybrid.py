"""Evaluation-only untouched native crops for the automatically selected photos.

ROI boxes only locate diagnostic crops. This script never builds a prediction,
changes source labels, or feeds these crops back into source selection.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from luster_hybrid_common import (assert_source,geometry,native_rows,project,read,
    sha,source_path,unproject,write)


def review(root,name,regions=('nails',),count=3):
    rows=native_rows();protocol=read(root/'evaluation/protocol.json')
    out=root/'evaluation'/f'{name}_sources';out.mkdir(exist_ok=True);records=[]
    for target in [7,12,95,150]:
        labels=np.load(root/'candidates'/name/f'cam{target:03d}_sources.npy',mmap_mode='r')
        g=geometry(root,target)
        for region in regions:
            box=protocol['cameras'][str(target)]['boxes'].get(region)
            if box is None:continue
            x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];lab=labels[sl]
            yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
            ids,counts=np.unique(lab[lab>=0],return_counts=True)
            chosen=sorted(zip(ids,counts),key=lambda x:(-x[1],x[0]))[:count]
            panels=[]
            for cid,n in chosen:
                cid=int(cid);assert_source(rows,target,cid)
                select=lab==cid;world=unproject(np.asarray(g['depth_z_q50'][sl])[select],rows[target],xy[select])
                uv,_=project(world,rows[cid]);assert np.isfinite(uv).all()
                photo_path=source_path(cid);photo=Image.open(photo_path).convert('RGB')
                low=np.floor(uv.min(axis=0)-12).astype(int);high=np.ceil(uv.max(axis=0)+13).astype(int)
                b=[int(max(0,low[0])),int(max(0,low[1])),int(min(photo.width,high[0])),int(min(photo.height,high[1]))]
                crop=photo.crop(b);stem=f'cam{target:03d}_{region}_native_src{cid:03d}'
                crop.save(out/f'{stem}.png')
                mask=np.zeros((crop.height,crop.width),np.uint8)
                points=np.rint(uv).astype(int)-np.array(b[:2]);inside=(points[:,0]>=0)&(points[:,0]<crop.width)&(points[:,1]>=0)&(points[:,1]<crop.height)
                points=points[inside];mask[points[:,1],points[:,0]]=255
                Image.fromarray(mask).save(out/f'{stem}_projected_samples.png')
                preview=crop.copy();preview.thumbnail((500,500));panels.append((cid,int(n),preview))
                records.append(dict(target=target,region=region,target_box=box,source=cid,
                    selected_target_pixels=int(n),source_box=b,source_path=str(photo_path),source_sha256=sha(photo_path),
                    crop=str(out/f'{stem}.png'),native_crop_size=[crop.width,crop.height]))
            if panels:
                canvas=Image.new('RGB',(500*len(panels),540),(25,25,25));draw=ImageDraw.Draw(canvas)
                for i,(cid,n,panel) in enumerate(panels):
                    draw.text((500*i+4,5),f'src{cid:03d}; {n} selected target pixels',fill='white')
                    draw.text((500*i+4,20),'Native photo crop; preview may be reduced',fill='white')
                    canvas.paste(panel,(500*i,40))
                canvas.save(out/f'cam{target:03d}_{region}_native_sources_preview.jpg',quality=95)
    write(out/'native_source_crops.json',dict(usage='evaluation_only_no_prediction_feedback',candidate=name,
        code_sha256=sha(Path(__file__)),selection='top existing source-label counts inside evaluation ROI',rows=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--regions',nargs='+',default=['nails']);p.add_argument('--count',type=int,default=3);a=p.parse_args()
    review(a.root,a.name,a.regions,a.count)
