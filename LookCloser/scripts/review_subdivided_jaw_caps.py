"""Matched native RGB/depth controls on frozen real-view diagnostic polygons."""
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image,panel
from diagnose_jaw_transfer_support import POLYGONS
from study_subdivided_jaw_caps import OUT,SOURCE


def run(large=False):
    frame='001193';dest=OUT/('review_large' if large else 'review');dest.mkdir(exist_ok=False)
    records=[]
    for view in ['moving','F004_E005_1210FP']:
        roots=[SOURCE/'rgb'/frame/view/'baseline',SOURCE/'rgb'/frame/view/'repaired',
               OUT/'planar/rgb'/frame/view/'repaired',OUT/'curved/rgb'/frame/view/'repaired']
        labels=['production','previous flat repair','subdivided flat','subdivided curved']
        if large:
            roots.append(Path('/mnt/data/dec5_large_curved_jaw_caps/curved/rgb')/frame/view/'repaired')
            labels.append('larger curved')
        images=[];metadata=[];depths=[]
        for root in roots:
            im,record=verified_image(root,frame);images.append(im);metadata.append(record)
            depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:
            assert all(r[key]==metadata[0][key] for r in metadata)
        stats=dict(view=view,inputs={str(root/'frames'/frame/'frame.png'):sha(root/'frames'/frame/'frame.png') for root in roots},
            rgb_changes_from_previous=[int(np.any(im!=images[1],axis=2).sum()) for im in images],
            depth_changes_from_previous=[int((d!=depths[1]).sum()) for d in depths])
        if view!='moving':
            gtpath=Path('/mnt/data/dec5_jaw_repair_transfer/review')/frame/(view+'_gt.png')
            gt=np.array(Image.open(gtpath));stats['gt_sha256']=sha(gtpath)
            for roi,polygon in [('interior',POLYGONS[frame]),('edge',[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]])]:
                mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
                stats[roi]=dict(polygon=polygon,pixels=int(mask.sum()),depth_misses=[int((mask&(d==0)).sum()) for d in depths],
                    rgb_black=[int((mask&(im.max(axis=2)==0)).sum()) for im in images])
            images=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_head.png'),images,labels,(170,450,970,1350))
        box=(570,1070,780,1240) if view!='moving' else (370,1080,670,1320)
        panel(dest/(view+'_detail.png'),images,labels,box);records.append(stats)
    atomic_json(dest/'result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        metric_scope='frozen train skin ROI coverage diagnostics only; no face/full-frame quality scores'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--large',action='store_true');run(p.parse_args().large)
