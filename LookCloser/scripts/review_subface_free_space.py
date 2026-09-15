"""Matched original/subdivision-only/subdivision-pruned native RGB comparisons."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_subface_free_space import ROOT,FRAME,DEPTH_ROOT
from review_measured_free_surface import baseline,VIEWS
from review_jaw_repair_transfer import verified_image,panel

BOXES={'moving':(380,1400,680,1870),'H004_C005_1210SZ':(230,1080,530,1550),
       'K004_B005_1210DS':(40,1120,340,1590)}
HEADS={'moving':(150,900,850,1490),'H004_C005_1210SZ':(100,500,1000,1200),
       'K004_B005_1210DS':(100,450,950,1210)}


def review(view):
    roots=[baseline(FRAME,view)]+[ROOT/control/FRAME/'rgb'/view for control in ['refined','pruned']]
    images=[];results=[];requests=[];depths=[];bindings={}
    for root in roots:
        rgb,result=verified_image(root,FRAME);images.append(rgb);results.append(result)
        requests.append(read(root/'request.json'))
        depths.append(np.load(root/'frames'/FRAME/'target_depth.npz')['depth'])
        for path in [root/'request.json',root/'frames'/FRAME/'complete.json']:
            bindings[str(path)]=sha(path)
        complete=read(root/'frames'/FRAME/'complete.json')
        bindings.update({str(root/'frames'/FRAME/n):h for n,h in complete['hashes'].items()})
    for result in results[1:]:
        for key in ['camera','source_cameras','fixed_exposure']:assert result[key]==results[0][key]
    for request in requests[1:]:
        for key in ['profiles_sha256','exposure_sha256','calibration_sha256','recipe']:
            assert request[key]==requests[0][key]
    deltas=[]
    for ai,bi in [(0,1),(1,2),(0,2)]:
        a,b=depths[ai],depths[bi];both=(a>0)&(b>0)
        assert a.shape==b.shape==(1080,1920) and np.isfinite(a).all() and np.isfinite(b).all()
        changed=np.any(images[ai]!=images[bi],axis=2)
        new_black=(images[ai].max(2)>0)&(images[bi].max(2)==0)
        delta=np.abs(b-a)
        deltas.append(dict(pair=[ai,bi],lost_hits=int(((a>0)&(b==0)).sum()),
            gained_hits=int(((a==0)&(b>0)).sum()),nearer_over_1e6=int((both&(b<a-1e-6)).sum()),
            farther_over_1e6=int((both&(b>a+1e-6)).sum()),max_shared_depth_delta=float(delta[both].max(initial=0)),
            changed_rgb=int(changed.sum()),new_black_rgb=int(new_black.sum())))
    # Pure subdivision cannot materially alter the surface; pruned is a subset.
    assert deltas[0]['nearer_over_1e6']==deltas[0]['farther_over_1e6']==0
    assert deltas[1]['gained_hits']==deltas[1]['nearer_over_1e6']==0
    dest=ROOT/FRAME/'review'/view;dest.mkdir(parents=True,exist_ok=True)
    names=['original','subdivision only','subdivision + pruning']
    if view!='moving':
        gt=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png'
        bindings[str(gt)]=sha(gt);images=[np.array(Image.open(gt)),*images];names=['actual train GT',*names]
    panel(dest/'lipstick_native.png',images,names,BOXES[view])
    panel(dest/'head_native.png',images,names,HEADS[view])
    overview=[np.array(Image.fromarray(im).resize((540,960),Image.Resampling.LANCZOS)) for im in images]
    panel(dest/'overview.png',overview,names,(0,0,540,960))
    atomic_json(dest/'result.json',dict(view=view,bindings=bindings,comparisons=deltas,
        hashes={name:sha(dest/name) for name in ['lipstick_native.png','head_native.png','overview.png']},
        script_sha256=sha(__file__),matched_recipe=True,visual_status='pending',production_promoted=False,
        counts_not_quality_metrics=True))
    print(view,deltas,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--view',choices=VIEWS,required=True)
    review(parser.parse_args().view)
