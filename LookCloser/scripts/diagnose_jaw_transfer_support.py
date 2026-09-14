"""Attribute remaining GT-drawn under-jaw skin misses before changing the gate.

Manual train-only polygons follow visible skin/shadow, not prediction coverage.
They are diagnostic annotations, not held-out face metrics or shape inputs.
"""
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image,ImageDraw
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,cameras
from study_jaw_repair_transfer import OUT
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

POLYGONS={
    '001193':[[610,1090],[655,1120],[693,1134],[710,1146],[706,1170],[680,1166],[635,1145],[610,1125]],
    '001195':[[610,1090],[655,1120],[693,1134],[710,1146],[706,1170],[680,1166],[635,1145],[610,1125]],
}


def run(output,frame,include_edge=False):
    folder=output/frame;spec=read(folder/'request.json');a=np.load(folder/'evidence.npz')
    gtpath=output/'review'/frame/'F004_E005_1210FP_gt.png';gt=np.array(Image.open(gtpath))
    polygon=POLYGONS[frame]
    if include_edge:
        polygon=[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],
                 [711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]]
    mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=mask.astype(bool)
    mesh=o3d.io.read_triangle_mesh(spec['source_mesh']);v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    candidate=np.concatenate([t,a['proposals']]);train,_,_=cameras(frame)
    camera=next(r for r in train if r['physical_camera']=='F004_E005_1210FP')
    d,ids,_=camera_depth(scene_for(v,candidate),camera);d,ids=np.rot90(d),np.rot90(ids)
    prior=output/'rgb'/frame/'F004_E005_1210FP'
    old=np.rot90(np.load(prior/'baseline/frames'/frame/'target_depth.npz')['depth'])
    final=np.rot90(np.load(prior/'repaired/frames'/frame/'target_depth.npz')['depth'])
    missing=mask&(old==0);remaining=mask&(final==0)
    covered=missing&np.isfinite(d)&(ids>=len(t))&(ids<len(candidate))
    which=ids[covered].astype(int)-len(t);local=np.unique(which)
    reasons=[]
    for i in local:
        reasons.append(dict(proposal=int(i),pixels=int((which==i).sum()),
            vertex_support_ok=bool((a['votes'][i,:3]>=2).sum()>=2),median_support_ok=bool(np.median(a['votes'][i])>=2),
            mask_support=int(a['mask_support'][i]),mask_veto=int(a['mask_outside'][i]),
            sample_far_veto=bool(a['free'][:,i,:].any()),retained=bool(i in a['retained_proposal_ids'])))
    overlay=gt.copy();overlay[missing]=[255,0,0];overlay[covered]=[0,255,255]
    pred=np.array(Image.open(prior/'repaired/frames'/frame/'frame.png'))
    box=(570,1070,780,1240);images=[gt,overlay,pred];panel=Image.new('RGB',(630,194));draw=ImageDraw.Draw(panel)
    for j,(im,label) in enumerate(zip(images,['GT: fixed skin polygon','red=miss; cyan=raw cap','final prediction'])):
        patch=Image.fromarray(im).crop(box)
        if j==0:
            dd=ImageDraw.Draw(patch);points=[(x-box[0],y-box[1]) for x,y in polygon]
            dd.line(points+[points[0]],fill='yellow',width=1)
        panel.paste(patch,(j*210,24));draw.text((j*210+2,3),label,fill='white')
    dest=output/('support_diagnosis_edge' if include_edge else 'support_diagnosis')/frame;dest.mkdir(parents=True,exist_ok=True);panel.save(dest/'native.png')
    atomic_json(dest/'result.json',dict(frame=frame,manual_train_skin_polygon=polygon,gt_sha256=sha(gtpath),
        annotation_role='posthoc diagnostic only; no shape/gate tuning',skin_pixels=int(mask.sum()),
        baseline_missing=int(missing.sum()),final_missing=int(remaining.sum()),raw_proposal_covers_missing=int(covered.sum()),
        uncovered_even_before_any_gate=int((missing&~np.isfinite(d)).sum()),proposal_reasons=reasons,
        panel_sha256=sha(dest/'native.png'),script_sha256=sha(__file__),face_metrics=False))
    print(frame,'skin misses',int(missing.sum()),'->',int(remaining.sum()),'raw cap covers',int(covered.sum()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=list(POLYGONS),required=True)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--include-edge',action='store_true')
    a=p.parse_args();run(a.output,a.frame,a.include_edge)
