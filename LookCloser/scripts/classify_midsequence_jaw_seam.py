"""Separate geometry misses, missing RGB support and dark source RGB in a GT jaw band."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from render_midsequence_jaw_completion import ROOT, baseline
from review_jaw_repair_transfer import verified_image, panel
from diagnose_midsequence_jaw_seam import VIEW, BOX

# Fixed after inspecting the real train GT coordinate preview at 001193.
# Shared across both neighboring instants, interior to face/neck, not a metric ROI.
POLYGON=[[480,1080],[550,1120],[610,1140],[680,1135],
         [680,1185],[620,1185],[550,1160],[480,1115]]


def run(frame):
    root=ROOT/frame/'seam_diagnosis'
    gt=np.asarray(Image.open(root/'train_gt.png'))
    mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.asarray(POLYGON,np.int32)],1)
    inside=mask.astype(bool); images=[gt]; labels=['real train GT']; summary=[]; arrays={}
    for arm,folder in [('production',baseline(frame,VIEW)),('completed',ROOT/frame/'rgb'/VIEW/'completed')]:
        rgb,receipt=verified_image(folder,frame); data=folder/'frames'/frame
        depth=np.rot90(np.load(data/'target_depth.npz')['depth']); ids=np.rot90(np.asarray(Image.open(data/'source_ids.png')))
        black=rgb.max(2)==0
        geometric=inside&(depth==0); invisible=inside&(depth>0)&(ids==255)
        source_black=inside&(depth>0)&(ids<62)&black
        arrays[arm+'_rgb']=rgb[inside];arrays[arm+'_depth']=depth[inside];arrays[arm+'_source_id']=ids[inside]
        im=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(im);draw.polygon(POLYGON,outline=(0,255,80),width=1)
        for criterion,color in [(geometric,(255,40,40)),(invisible,(30,170,255)),(source_black,(255,255,0))]:
            yy,xx=np.where(criterion)
            for x,y in zip(xx,yy):draw.point((int(x),int(y)),fill=color)
        images.extend([rgb,np.asarray(im)]);labels.extend([arm,arm+' red=geometry blue=no RGB yellow=source black'])
        summary.append(dict(arm=arm,request_sha256=sha(folder/'request.json'),
            complete_sha256=sha(data/'complete.json'),roi_pixels=int(inside.sum()),
            black_rgb=int((inside&black).sum()),geometry_misses=int(geometric.sum()),
            valid_geometry_no_source=int(invisible.sum()),valid_source_black_rgb=int(source_black.sum())))
    np.savez_compressed(root/'classification.npz',portrait_yx=np.column_stack(np.where(inside)),gt_rgb=gt[inside],**arrays)
    panel(root/'classification_native.png',images,labels,(460,1060,700,1210))
    atomic_json(root/'classification.json',dict(frame=frame,view=VIEW,polygon=POLYGON,
        polygon_chosen_from_train_GT_preview=True,quality_metrics=False,
        candidate_defined_metric_mask=False,not_a_complete_anatomical_hole_inventory=True,
        records=summary,script_sha256=sha(__file__),
        hashes={n:sha(root/n) for n in ['train_gt.png','classification.npz','classification_native.png']}))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=['001193','001195'])
    run(p.parse_args().frame)
