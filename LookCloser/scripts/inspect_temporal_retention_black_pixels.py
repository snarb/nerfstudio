"""Expose every newly black RGB point without mistaking it for a ray miss."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from study_temporal_source_retention import ROOT,BASE,CLIPS
from joint_temporal_texture import read,sha,atomic_json
from review_temporal_source_retention import draw_pair


def main():
    dest=ROOT/'review/new_black';dest.mkdir(exist_ok=False)
    records=[];panels=[];bindings={}
    for frames in CLIPS.values():
        for f in frames:
            a=BASE/'frames'/f;b=ROOT/f/'frames'/f
            old=np.array(Image.open(a/'frame.png'));new=np.array(Image.open(b/'frame.png'))
            oa=np.rot90(np.array(Image.open(a/'source_ids.png')));nb=np.rot90(np.array(Image.open(b/'source_ids.png')))
            depth=np.rot90(np.load(b/'target_depth.npz')['depth']);r=read(b/'result.json')
            mask=(old.max(2)>0)&(new.max(2)==0)
            for y,x in np.argwhere(mask):
                box=[max(0,int(x)-45),max(0,int(y)-45),min(1080,int(x)+46),min(1920,int(y)+46)]
                # Mark the central sample without recoloring the queried pixel.
                im=draw_pair(old[box[1]:box[3],box[0]:box[2]],new[box[1]:box[3],box[0]:box[2]],f)
                d=ImageDraw.Draw(im);w=box[2]-box[0]
                for offset in [0,w]:d.rectangle((offset+x-box[0]-2,y-box[1]+22,offset+x-box[0]+2,y-box[1]+26),outline='magenta')
                path=dest/f'{f}_{x}_{y}.png';im.save(path);panels.append(path)
                records.append(dict(frame=f,portrait_xy=[int(x),int(y)],old_rgb=old[y,x].tolist(),
                    previous_source=int(oa[y,x]),new_source=int(nb[y,x]),
                    new_physical_camera=r['source_cameras'][nb[y,x]] if nb[y,x]<62 else None,
                    geometry_hit=bool(depth[y,x]>0),no_source=bool(nb[y,x]==255),crop=box,path=str(path)))
            for p in [a/'complete.json',b/'complete.json',a/'frame.png',b/'frame.png',b/'source_ids.png',b/'target_depth.npz']:
                bindings[str(p)]=sha(p)
    atomic_json(dest/'result.json',dict(records=records,input_hashes=bindings,
        images={str(p):sha(p) for p in panels},script_sha256=sha(__file__),visual_status='pending'))
    print('new black',len(records),'with source',sum(not r['no_source'] for r in records),flush=True)


if __name__=='__main__':main()
