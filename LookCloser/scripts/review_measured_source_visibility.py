"""Verify fixed geometry and inspect strict measured-source RGB admission."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_measured_source_visibility import ROOT,ARMS,old_root
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_measured_free_surface import VIEWS
from review_jaw_repair_transfer import verified_image,panel


def review():
    frame='000995';records=[]
    boxes={'moving':(380,1400,680,1870),'H004_C005_1210SZ':(230,1080,530,1550),
           'K004_B005_1210DS':(40,1120,340,1590)}
    heads={'moving':(150,900,850,1490),'H004_C005_1210SZ':(100,500,1000,1200),
           'K004_B005_1210DS':(100,450,950,1210)}
    for arm in ARMS:
        for view in VIEWS:
            root=ROOT/frame/arm/view;old=old_root(frame,arm,view)
            a,ar=verified_image(old,frame);b,br=verified_image(root,frame)
            for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert ar[key]==br[key]
            aq,bq=read(old/'request.json'),read(root/'request.json')
            for key in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert aq[key]==bq[key]
            assert bq['measured_source_visibility']['baseline_request_sha256']==sha(old/'request.json')
            ad=np.load(old/'frames'/frame/'target_depth.npz')['depth']
            bd=np.load(root/'frames'/frame/'target_depth.npz')['depth']
            np.testing.assert_array_equal(ad,bd)
            assert br['measured_source_visibility']['unknown_depth_admits'] is False
            images=[a,b];labels=['matched baseline '+arm,'positive measured source gate']
            if view!='moving':
                images.insert(0,np.asarray(Image.open(DEPTH_ROOT/frame/'review'/view/'train_gt.png')))
                labels.insert(0,'real train GT')
            out=root/'review'
            panel(out/'lipstick_native.png',images,labels,boxes[view])
            panel(out/'head_native.png',images,labels,heads[view])
            panel(out/'actor.png',[np.asarray(Image.fromarray(im).resize((540,960))) for im in images],labels,(0,0,540,960))
            old_black=~a.any(2);new_black=~b.any(2)
            slices={name:(slice(box[1],box[3]),slice(box[0],box[2])) for name,box in [('head',heads[view]),('lipstick_hand',boxes[view])]}
            counts={name:dict(introduced=int((new_black&~old_black)[s].sum()),
                removed=int((old_black&~new_black)[s].sum())) for name,s in slices.items()}
            records.append(dict(arm=arm,view=view,depth_arrays_identical=True,
                baseline_png_sha256=sha(old/'frames'/frame/'frame.png'),candidate_png_sha256=sha(root/'frames'/frame/'frame.png'),
                baseline_request_sha256=sha(old/'request.json'),candidate_request_sha256=sha(root/'request.json'),
                changed_rgb_pixels=int(np.any(a!=b,axis=2).sum()),black_pixel_changes=counts,
                hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},visual_status='pending'))
    atomic_json(ROOT/frame/'review_result.json',dict(records=records,counts_not_quality_metrics=True,
        production_promoted=False,heldout_used=False))


if __name__=='__main__':review()
