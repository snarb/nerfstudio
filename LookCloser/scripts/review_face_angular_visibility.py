"""Matched native panels covering every changed pixel, with immutable receipts."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from study_multiview_face_prior import read,save,sha
from review_face_interior_visibility import panel


def main(roots):
    base=Path(read(roots[0]/'result.json')['baseline']);a=np.array(Image.open(base/'frame.png'))
    images=[Image.fromarray(a)];labels=['baseline'];changed=np.zeros(a.shape[:2],bool);records=[];bindings={}
    for root in roots:
        r=read(root/'result.json');assert r['baseline']==str(base)
        for n,h in r['hashes'].items():assert sha(root/n)==h;bindings[str(root/n)]=h
        b=np.array(Image.open(root/'frame.png'));images.append(Image.fromarray(b));labels.append(root.name)
        source=np.rot90(np.array(Image.open(root/'source_ids.png')));old=np.rot90(np.array(Image.open(base/'source_ids.png')))
        diff=source!=old;np.testing.assert_array_equal(a[~diff],b[~diff]);changed|=diff
        records.append(dict(root=str(root),source_changes=int(diff.sum()),rgb_changes=int(np.any(a!=b,2).sum()),
            new_black=int(((a.max(2)>0)&(b.max(2)==0)).sum()),result_sha256=sha(root/'result.json')))
    dest=roots[0].parent/'review';assert not dest.exists();dest.mkdir()
    panel([x.resize((360,640)) for x in images],labels,dest/'overview.png')
    panel([x.crop((780,450,1080,850)) for x in images],labels,dest/'face.png')
    panel([x.crop((875,595,960,765)) for x in images],labels,dest/'nose.png')
    Y,X=np.nonzero(changed);boxes=[]
    for tx,ty in sorted(set(zip((X//128).tolist(),(Y//128).tolist()))):
        box=[max(0,128*tx-12),max(0,128*ty-12),min(1080,128*(tx+1)+12),min(1920,128*(ty+1)+12)]
        name=f'changed_{tx}_{ty}.png';panel([x.crop(box) for x in images],labels,dest/name);boxes.append(dict(image=name,box=box))
    for p in [base/'complete.json',base/'frame.png',Path(__file__)]:bindings[str(p)]=sha(p)
    save(dest/'audit.json',dict(records=records,input_hashes=bindings,changed_tiles=boxes,
        images={str(p):sha(p) for p in dest.glob('*.png')},visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');a=p.parse_args();main(a.roots)
