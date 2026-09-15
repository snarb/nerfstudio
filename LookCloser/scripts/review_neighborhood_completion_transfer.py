"""Frozen train jaw diagnostics for same-time, same-footprint geometry pairs."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from run_neighborhood_completion_transfer import ROOT
from review_jaw_repair_transfer import verified_image,panel
from diagnose_jaw_transfer_support import POLYGONS


def run(frame):
    output=ROOT/frame;dest=output/'review';dest.mkdir(exist_ok=False);records=[]
    for view in ['moving','F004_E005_1210FP']:
        roots=[output/'rgb'/view/v for v in ['baseline','repaired']];images=[];depth=[];metadata=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);metadata.append(r)
            depth.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:assert metadata[0][key]==metadata[1][key]
        record=dict(frame=frame,view=view,inputs={str(p/'frames'/frame/'frame.png'):sha(p/'frames'/frame/'frame.png') for p in roots},
            same_footprint=True,changed_rgb_pixels=int(np.any(images[0]!=images[1],axis=2).sum()))
        labels=['production + fixed footprint','completion + fixed footprint']
        if view!='moving':
            gtpath=Path('/mnt/data/dec5_jaw_repair_transfer/review')/frame/(view+'_gt.png');gt=np.array(Image.open(gtpath));record['gt_sha256']=sha(gtpath)
            for name,polygon in [('interior',POLYGONS[frame]),('edge',[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]])]:
                mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
                record[name]=dict(polygon=polygon,depth_misses=[int((mask&(d==0)).sum()) for d in depth],
                    black_rgb=[int((mask&(im.max(2)==0)).sum()) for im in images])
            images=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_head.png'),images,labels,(170,450,970,1350))
        panel(dest/(view+'_detail.png'),images,labels,(570,1070,780,1240) if view!='moving' else (370,1080,670,1320))
        records.append(record)
    atomic_json(dest/'result.json',dict(script_sha256=sha(__file__),records=records,visual_status='pending',full_frame_quality_metrics=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001193','001195'],required=True);run(p.parse_args().frame)
