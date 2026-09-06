#!/usr/bin/env python3
"""Read-only train-camera time-offset correspondence audit; no synchronization fix.

Track reference-camera features through neighboring frames, then inspect frozen-
calibration epipolar residuals against secondary-camera images at five times.
Calibration bias and view-dependent feature localization can mimic a time offset;
this is not timestamp ground truth and never changes the prediction's inputs.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_source_epipolar_residuals import fundamental
from audit_fixed_camera_feature_geometry import mutual_matches,epipolar_distance


def fit_residual_trend(offsets,residuals):
    offsets=np.asarray(offsets,float);residuals=np.asarray(residuals,float)
    if (len(offsets)<3 or len(set(offsets))!=len(offsets) or not np.isfinite(offsets).all()
            or not np.isfinite(residuals).all() or offsets.shape!=residuals.shape):
        raise ValueError('Need at least three unique finite temporal observations')
    slope,intercept=np.polyfit(offsets,residuals,1)
    error=residuals-(slope*offsets+intercept)
    r2=1-float(error@error)/max(float(((residuals-residuals.mean())**2).sum()),1e-12)
    qualified=bool(abs(slope)>=.1 and r2>=.9)
    return dict(slope_pixels_per_available_frame=float(slope),intercept_pixels=float(intercept),r2=r2,
                qualified=qualified,zero_crossing_available_frames=float(-intercept/slope) if qualified else None)


def main():
    from nerfstudio.data.utils.data_utils import load_exr_image
    from convert_exr_nerfstudio_to_jpeg import tone_map
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--frames',nargs='+',required=True);p.add_argument('--reference-camera',required=True)
    p.add_argument('--secondary-cameras',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    names=[a.reference_camera,*a.secondary_cameras]
    if forbidden&set(names) or len(set(names))!=len(names):p.error('Unique train cameras only')
    if a.output.exists():p.error('Preserve existing audit')
    inventory=sorted([q.name for q in a.parent.iterdir() if q.is_dir() and len(q.name)==6 and q.name.isdigit()],key=int)
    calibration=json.loads(a.calibration.read_text());cameras={f['physical_camera']:{**calibration,**f} for f in calibration['frames']}
    cv2.setNumThreads(4);sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.01,edgeThreshold=12)
    hashes={str(a.calibration):sha256(a.calibration)};receipts=[];cache={};all_results=[]
    def load(frame_id,name):
        key=(frame_id,name)
        if key in cache:return cache[key]
        directory=a.parent/frame_id;transforms=directory/'transforms.json';payload=json.loads(transforms.read_text())
        hashes[str(transforms)]=sha256(transforms)
        candidates=[f for f in payload['frames'] if f['physical_camera']==name]
        if len(candidates)!=1:raise ValueError('Ambiguous physical camera')
        f=candidates[0]
        if f.get('mask_path') or not Path(f['file_path']).name.startswith('frame_train_'):raise ValueError('Unmasked train RGB required')
        path=directory/f['file_path'];hashes[str(path)]=sha256(path);rgb=load_exr_image(path)
        if rgb.shape[:2]!=(1080,1920) or not np.isfinite(rgb).all():raise ValueError('Invalid source RGB')
        sample=np.maximum(rgb[::8,::8,:3],0);lum=sample@np.array([.2126,.7152,.0722],np.float32)
        anchor=float(np.percentile(lum,70));gain=.18/(max(anchor,1e-8)*.82)
        display=tone_map(rgb,gain);memory=BytesIO();Image.fromarray(display).save(memory,format='JPEG',quality=98,subsampling=0)
        encoded=memory.getvalue();gray=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_GRAYSCALE)
        keypoints,desc=sift.detectAndCompute(gray,None);xy=np.array([k.pt for k in keypoints],np.float32)
        cache[key]=(gray,xy,desc)
        import hashlib
        receipts.append(dict(frame_id=frame_id,physical_camera=name,source=str(path),gain=gain,
            jpeg_sha256=hashlib.sha256(encoded).hexdigest(),features=len(xy)))
        return cache[key]
    for frame_id in a.frames:
        index=inventory.index(frame_id)
        if index<2 or index+2>=len(inventory):raise ValueError('Need symmetric five-frame window')
        window={offset:inventory[index+offset] for offset in range(-2,3)}
        reference,xy,desc=load(frame_id,a.reference_camera)
        tracked=[]
        for offset in [-1,1]:
            image=load(window[offset],a.reference_camera)[0]
            moved,ok,_=cv2.calcOpticalFlowPyrLK(reference,image,xy[:,None],None,winSize=(31,31),maxLevel=3)
            back,ok_back,_=cv2.calcOpticalFlowPyrLK(image,reference,moved,None,winSize=(31,31),maxLevel=3)
            good=ok[:,0].astype(bool)&ok_back[:,0].astype(bool)&(np.linalg.norm(back[:,0]-xy,axis=1)<.5)
            tracked.append((moved[:,0],good))
        good=tracked[0][1]&tracked[1][1]
        velocity=(tracked[1][0]-tracked[0][0])/2
        speed=np.linalg.norm(velocity,axis=1)
        for name in a.secondary_cameras:
            matrix=fundamental(cameras[a.reference_camera],cameras[name],.5)
            correspondences={};counts={}
            for offset,target_id in window.items():
                _,points,description=load(target_id,name)
                matched=mutual_matches(desc,description);counts[offset]=len(matched)
                residual=epipolar_distance(matrix,xy[matched[:,0]],points[matched[:,1]])
                for (i,j),e in zip(matched,residual):
                    correspondences.setdefault(int(i),{})[offset]=dict(residual=float(e),point=points[j].tolist())
            records=[]
            for i,matches in correspondences.items():
                if not good[i] or 0 not in matches or len(matches)<3:continue
                offsets=sorted(matches)
                trend=fit_residual_trend(offsets,[matches[t]['residual'] for t in offsets])
                records.append(dict(reference_index=i,reference_xy=xy[i].tolist(),reference_speed=float(speed[i]),
                    group='moving' if speed[i]>=.5 else 'near_static' if speed[i]<.15 else 'intermediate',
                    matches=matches,trend=trend,spatial_block=(xy[i]//128).astype(int).tolist()))
            summary=[]
            for group in ['moving','near_static','intermediate']:
                subset=[r for r in records if r['group']==group]
                common=[r for r in subset if len(r['matches'])==5]
                qualified=[r for r in subset if r['trend']['qualified']]
                summary.append(dict(group=group,tracks=len(subset),all_five_offset_tracks=len(common),qualified_linear_trends=len(qualified),
                    paired_median_absolute_epipolar_by_offset={t:float(np.median([abs(r['matches'][t]['residual']) for r in common])) for t in window} if common else {},
                    median_zero_crossing_available_frames=float(np.median([r['trend']['zero_crossing_available_frames'] for r in qualified])) if qualified else None))
            all_results.append(dict(frame_id=frame_id,secondary_camera=name,window=window,cross_camera_matches_by_offset=counts,
                summary=summary,records=records))
            print(json.dumps({k:v for k,v in all_results[-1].items() if k!='records'}),flush=True)
        # Only retain small grayscale inputs for the current frame's visual examples.
        cache.clear()
    a.output.mkdir(parents=True)
    result=dict(uses_eval_rgb=False,uses_semantic_masks=False,uses_mesh=False,changes_calibration=False,
        changes_prediction=False,changes_source_frame_identity=False,reference_camera=a.reference_camera,
        frame_inventory=inventory,frames=a.frames,results=all_results,ingest_receipts=receipts,input_hashes=hashes,
        interpretation=__doc__,time_unit='one available directory, two numeric source-frame IDs in this dataset',
        thresholds=dict(reference_moving_speed=.5,reference_near_static_speed=.15,lk_round_trip=.5,
                        cross_camera_mutual_sift_ratio=.7,min_temporal_observations=3,min_trend_slope=.1,min_trend_r2=.9))
    for name in ['audit_temporal_source_correspondence.py','audit_source_epipolar_residuals.py','audit_fixed_camera_feature_geometry.py','convert_exr_nerfstudio_to_jpeg.py']:
        path=Path(__file__).with_name(name);result['input_hashes'][str(path)]=sha256(path)
    atomic_json(a.output/'audit.json',result)


if __name__=='__main__':main()
