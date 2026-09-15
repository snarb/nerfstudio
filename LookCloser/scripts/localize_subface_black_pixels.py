"""Post-hoc connected-component crops for newly black RGB after subface pruning."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from joint_temporal_texture import read,sha,atomic_json
from study_subface_free_space import ROOT,FRAME,DEPTH_ROOT
from review_measured_free_surface import VIEWS
from review_jaw_repair_transfer import verified_image,panel


def main():
    for view in VIEWS:
        roots=[ROOT/c/FRAME/'rgb'/view for c in ['refined','pruned']]
        images=[verified_image(p,FRAME)[0] for p in roots]
        new_black=(images[0].max(2)>0)&(images[1].max(2)==0)
        labels,count=ndimage.label(new_black,np.ones((3,3),int))
        sizes=np.bincount(labels.ravel());records=[]
        dest=ROOT/FRAME/'review'/view/'new_black';dest.mkdir(exist_ok=False)
        names=['subdivision only','subdivision + pruning']
        if view!='moving':
            images=[np.array(Image.open(DEPTH_ROOT/FRAME/'review'/view/'train_gt.png')),*images]
            names=['actual train GT',*names]
        for label in sorted(range(1,count+1),key=lambda i:-sizes[i]):
            y,x=np.where(labels==label)
            box=[max(0,int(x.min())-20),max(0,int(y.min())-20),min(1080,int(x.max())+21),min(1920,int(y.max())+21)]
            name=f'{label:03d}.png';panel(dest/name,images,names,box)
            records.append(dict(component=label,pixels=int(sizes[label]),box=box,path=str(dest/name),sha256=sha(dest/name)))
        atomic_json(dest/'result.json',dict(view=view,new_black_pixels=int(new_black.sum()),
            components=records,render_review_sha256=sha(dest.parent/'result.json'),
            script_sha256=sha(__file__),visual_status='pending',counts_not_quality_metrics=True))
        print(view,'new black',int(new_black.sum()),'components',count,flush=True)


if __name__=='__main__':main()
