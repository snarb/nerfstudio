"""Inspect real RGB at mask disagreements that reject visible under-jaw caps."""
from pathlib import Path
import argparse
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project,ROOT,exr,display
from study_jaw_repair_transfer import OUT,PARENT
from study_confidence_depth_prior import load_real
from diagnose_jaw_measured_depth import observed_at


def run(output,frame):
    record=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    maskroot=Path(record['source_masks']['root']);masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    spec=read(output/frame/'request.json');a=np.load(output/frame/'evidence.npz')
    evidence=read(output/'support_diagnosis_edge'/frame/'result.json')
    selected=[r['proposal'] for r in evidence['proposal_reasons'] if r['mask_veto']]
    rows,depths,receipt=load_real(Path(spec['depth_root']),frame)
    if receipt!=spec['depth_receipt']:raise ValueError('Changed measured depths')
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain'];details=[]
    dest=output/'mask_disagreement'/frame;dest.mkdir(parents=True,exist_ok=True)
    for ci,row in enumerate(rows):
        if not selected:continue
        # Three vertices and centroid: exactly the semantic gate's samples.
        points=a['points'][selected][:,[0,1,2,9]].reshape(-1,3)
        uv,z=project(points,[row]);uv,z=uv[0],z[0];xy=np.rint(uv).astype(int)
        available=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
        mask=masks[names.index(row['physical_camera'])].astype(bool);ids=np.flatnonzero(available)
        bad=ids[~mask[xy[ids,1],xy[ids,0]]]
        if not len(bad):continue
        rgb=np.rint(display(exr(row['file_path'])*gains[row['physical_camera']],gain)*255).clip(0,255).astype(np.uint8)
        context=dest/(row['physical_camera']+'_gt_portrait.png')
        Image.fromarray(np.rot90(rgb)).save(context)
        edge=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
        overlay=rgb.copy();overlay[edge]=[0,255,255]
        center=np.rint(np.median(uv[bad],axis=0)).astype(int);x,y=center;box=(x-90,y-90,x+90,y+90)
        panel=Image.new('RGB',(360,204));draw=ImageDraw.Draw(panel)
        for j,im in enumerate([rgb,overlay]):
            patch=Image.fromarray(im).crop(box)
            if j:
                dd=ImageDraw.Draw(patch)
                for u,v in uv[bad]:dd.ellipse((u-box[0]-1,v-box[1]-1,u-box[0]+1,v-box[1]+1),fill='red')
            panel.paste(patch.rotate(90),(180*j,24))
        draw.text((2,4),row['physical_camera']+' GT / mask+rejected',fill='white')
        path=dest/(row['physical_camera']+'.png');panel.save(path)
        _,zq,observed,valid=observed_at(points[bad],row,depths[ci]);dist=distance_transform_edt(~mask)
        details.append(dict(camera=row['physical_camera'],proposals=selected,rejected_point_count=len(bad),
            rejected_xy=uv[bad].tolist(),distance_outside_mask=dist[xy[bad,1],xy[bad,0]].tolist(),
            depth_residual=np.where(valid,observed-zq,0).tolist(),depth_available=valid.tolist(),
            gt_sha256=sha(row['file_path']),panel=path.name,panel_sha256=sha(path),
            context=context.name,context_sha256=sha(context)))
    atomic_json(dest/'result.json',dict(frame=frame,selected_proposals=selected,details=details,
        mask_sha256=sha(maskroot/'masks.npz'),source_evidence_sha256=sha(output/frame/'evidence.npz'),
        script_sha256=sha(__file__),geometry_changed=False,visual_status='requires_actual_review'))
    print(frame,[(r['camera'],r['distance_outside_mask'],r['depth_residual']) for r in details],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001193','001195'],required=True)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();run(a.output,a.frame)
