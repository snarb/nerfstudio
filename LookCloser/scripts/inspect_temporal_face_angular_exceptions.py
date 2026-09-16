"""Expose every new black sample and explicitly show exact-zero-change frames."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_temporal_face_angular_control import ROOT,BASE,verify
from review_temporal_face_angular_control import frame_pair,pair
from build_train_hair_semantics import write,sha


def main():
    q=verify();dest=ROOT/'exceptions';assert not dest.exists();dest.mkdir();records=[];zeros=[];bindings={}
    for inv in q['inventory']:
        a,b,x,y,r=frame_pair(inv);frame=inv['frame']
        if not np.any(a!=b):
            np.testing.assert_array_equal(x,y)
            zeros.append(frame);im=pair(np.array(Image.fromarray(a).resize((360,640))),np.array(Image.fromarray(b).resize((360,640))),frame+' BIT-IDENTICAL')
            im.save(dest/(frame+'_zero_change.png'))
        mask=(a.max(2)>0)&(b.max(2)==0)
        if not mask.any():continue
        py,px=np.nonzero(mask);depth=np.rot90(np.load(r['depth_reference'])['depth'])
        assert (depth[py,px]>0).all() and (y[py,px]<62).all()
        box=[max(0,int(px.min())-45),max(0,int(py.min())-45),min(1080,int(px.max())+46),min(1920,int(py.max())+46)]
        images=[]
        for rgb in [a,b]:
            im=Image.fromarray(rgb).crop(box);draw=ImageDraw.Draw(im)
            for xx,yy in zip(px,py):
                dx=int(xx)-box[0];dy=int(yy)-box[1];draw.rectangle((dx-3,dy-3,dx+3,dy+3),outline='magenta')
            images.append(np.array(im))
        path=dest/(frame+'_new_black.png');pair(*images,frame).save(path)
        for xx,yy in zip(px,py):records.append(dict(frame=frame,portrait_xy=[int(xx),int(yy)],
            old_rgb=a[yy,xx].tolist(),new_rgb=b[yy,xx].tolist(),old_source=int(x[yy,xx]),new_source=int(y[yy,xx]),
            positive_depth=float(depth[yy,xx]),crop=str(path)))
        for p in [Path(inv['output'])/'result.json',Path(r['depth_reference'])]:bindings[str(p)]=sha(p)
    write(dest/'result.json',dict(new_black=records,bit_identical_frames=zeros,input_hashes=bindings,
        images={str(p):sha(p) for p in dest.glob('*.png')},visual_status='pending',script_sha256=sha(__file__)))
    print('black samples',len(records),'zero-change frames',zeros,flush=True)


if __name__=='__main__':main()
