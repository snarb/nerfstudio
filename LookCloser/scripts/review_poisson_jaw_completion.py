"""Compare continuous-surface priors with production and the earlier curved cap."""
import numpy as np
import cv2
import argparse
from pathlib import Path
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_poisson_jaw_completion import OUT,FRAME
from review_jaw_repair_transfer import verified_image,panel
from diagnose_jaw_transfer_support import POLYGONS


def run(interpolated=False):
    dest=OUT/('review_interpolated' if interpolated else 'review');dest.mkdir(exist_ok=False);records=[]
    for view in ['moving','F004_E005_1210FP']:
        roots=[Path('/mnt/data/dec5_jaw_measured_mask_control/rgb')/FRAME/view/'baseline',
               Path('/mnt/data/dec5_subdivided_jaw_caps/curved/rgb')/FRAME/view/'repaired',
               OUT/'strict/rgb'/FRAME/view/'repaired',OUT/'anchored/rgb'/FRAME/view/'repaired']
        images=[];depths=[];metadata=[];labels=['production','small curved cap','Poisson direct depth','Poisson with anchors']
        if interpolated:
            roots.append(OUT/'interpolated/rgb'/FRAME/view/'repaired');labels.append('Poisson interpolation')
        for root in roots:
            image,record=verified_image(root,FRAME);images.append(image);metadata.append(record)
            depths.append(np.rot90(np.load(root/'frames'/FRAME/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:
            assert all(r[key]==metadata[0][key] for r in metadata)
        record=dict(view=view,inputs={str(p/'frames'/FRAME/'frame.png'):sha(p/'frames'/FRAME/'frame.png') for p in roots},
                    changed_rgb_from_production=[int(np.any(im!=images[0],axis=2).sum()) for im in images])
        if view!='moving':
            gtpath=Path('/mnt/data/dec5_jaw_repair_transfer/review')/FRAME/(view+'_gt.png');gt=np.array(Image.open(gtpath))
            record['gt_sha256']=sha(gtpath)
            for name,polygon in [('interior',POLYGONS[FRAME]),('edge',[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]])]:
                mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
                record[name]=dict(polygon=polygon,pixels=int(mask.sum()),depth_misses=[int((mask&(d==0)).sum()) for d in depths],
                    black_rgb=[int((mask&(im.max(axis=2)==0)).sum()) for im in images])
            images=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_head.png'),images,labels,(170,450,970,1350))
        panel(dest/(view+'_detail.png'),images,labels,(570,1070,780,1240) if view!='moving' else (370,1080,670,1320))
        records.append(record)
    atomic_json(dest/'result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        quality_metrics_computed=False,scope='frozen train skin coverage diagnostics, not heldout face metrics'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interpolated',action='store_true');run(p.parse_args().interpolated)
