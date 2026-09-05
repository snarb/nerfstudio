#!/usr/bin/env python3
"""Independent train-JPEG feature geometry audit; no mesh, SfM or camera edits.

Fit a diagnostic fundamental matrix on spatially separated training matches and
compare its held-match residuals with the frozen calibration. This is not a new
calibration and is never used to generate a prediction or choose its sources.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from audit_source_epipolar_residuals import fundamental
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256

FORBIDDEN={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}


def epipolar_distance(f,a,b):
    ah=np.c_[a,np.ones(len(a))];bh=np.c_[b,np.ones(len(b))]
    lines=ah@f.T
    return (bh*lines).sum(1)/np.maximum(np.linalg.norm(lines[:,:2],axis=1),1e-12)


def mutual_matches(a,b,ratio=.7):
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def accepted(x,y):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2
                for m,n in [pair] if m.distance<ratio*n.distance}
    ab=accepted(a,b);ba=accepted(b,a)
    return np.asarray([(i,j) for i,j in ab.items() if ba.get(j)==i],np.int64).reshape(-1,2)


def holdout(a):
    tiles=np.floor(a/128).astype(np.int64)
    return ((tiles[:,0]*73856093)^(tiles[:,1]*19349663))%4==0


def summary(values):
    return {'count':len(values),'signed_median':float(np.median(values)),
            'absolute_median':float(np.median(np.abs(values))),
            'absolute_p90':float(np.quantile(np.abs(values),.9))} if len(values) else {'count':0}


def temporal_similarity(a,b):
    """Describe the near-identity match cluster, not a semantic static mask."""
    near=np.linalg.norm(a-b,axis=1)<5
    if near.sum()<20:return {'near_identity_matches':int(near.sum())},np.zeros(len(a),bool)
    affine,mask=cv2.estimateAffinePartial2D(a[near],b[near],method=cv2.RANSAC,ransacReprojThreshold=.5,
                                         maxIters=5000,confidence=.999,refineIters=10)
    accepted=np.zeros(len(a),bool)
    if affine is None:return {'near_identity_matches':int(near.sum()),'fit_failed':True},accepted
    accepted[np.where(near)[0]]=mask[:,0].astype(bool)
    pred=np.c_[a,np.ones(len(a))]@affine.T
    delta=b[accepted]-a[accepted]
    return {'near_identity_matches':int(near.sum()),'consistent_matches':int(accepted.sum()),
            'similarity_matrix':affine.tolist(),'median_displacement_xy':np.median(delta,axis=0).tolist(),
            'median_length':float(np.median(np.linalg.norm(delta,axis=1))),
            'median_fit_residual':float(np.median(np.linalg.norm(b[accepted]-pred[accepted],axis=1))),
            'coordinate_span_xy':np.ptp(a[accepted],axis=0).tolist()},accepted


def farthest_points(points,count):
    if not len(points):return []
    chosen=[int(np.argmin(points[:,0]+points[:,1]))]
    distance=np.full(len(points),np.inf)
    for _ in range(min(count,len(points))-1):
        distance=np.minimum(distance,np.linalg.norm(points-points[chosen[-1]],axis=1))
        chosen.append(int(distance.argmax()))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cameras',nargs='+',required=True);p.add_argument('--max-features',type=int,default=6000)
    p.add_argument('--reference-data',type=Path,default=None,
                   help='Audit same-camera temporal stability instead of cross-camera epipolar geometry.')
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing audit')
    if len(set(args.cameras))!=len(args.cameras) or FORBIDDEN&set(args.cameras):p.error('Unique train cameras only')
    payload=json.loads((args.data/'transforms.json').read_text())
    frames={f['physical_camera']:f for f in payload['frames']};train=set(payload.get('train_filenames',[]))
    selected=[frames[name] for name in args.cameras]
    for f in selected:
        if f.get('mask_path') or not Path(f['file_path']).name.startswith('frame_train_'):
            raise ValueError('Only unmasked train RGB may enter this diagnostic')
        if train and f['file_path'] not in train:raise ValueError('Selected source is outside train split')
    args.output.mkdir(parents=True);cv2.setNumThreads(4);cv2.setRNGSeed(0)
    sift=cv2.SIFT_create(nfeatures=args.max_features,contrastThreshold=.01,edgeThreshold=12)
    cached={};sources={}
    for name,f in zip(args.cameras,selected):
        path=(args.data/f['file_path']).resolve(strict=True)
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(f['h'],f['w']):raise ValueError('Source raster mismatch')
        key,desc=sift.detectAndCompute(gray,None)
        cached[name]=(np.asarray([k.pt for k in key]),desc)
        sources[name]={'path':str(path),'sha256':sha256(path),'features':len(key)}
        print(f'features camera={name} count={len(key)}',flush=True)
    if args.reference_data is not None:
        reference=json.loads((args.reference_data/'transforms.json').read_text())
        by_camera={f['physical_camera']:f for f in reference['frames']};rows=[]
        for name in args.cameras:
            f=by_camera[name]
            if f.get('mask_path') or not Path(f['file_path']).name.startswith('frame_train_'):
                raise ValueError('Unmasked train-only temporal reference required')
            path=(args.reference_data/f['file_path']).resolve(strict=True)
            key,desc=sift.detectAndCompute(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),None)
            xy=np.asarray([k.pt for k in key]);xy_b,desc_b=cached[name]
            matched=mutual_matches(desc,desc_b);a=xy[matched[:,0]];b=xy_b[matched[:,1]]
            stats,accepted=temporal_similarity(a,b)
            keep=np.where(accepted)[0];chosen=keep[farthest_points(a[keep],8)]
            im_a=Image.open(path).convert('RGB');im_b=Image.open(sources[name]['path']).convert('RGB')
            canvas=Image.new('RGB',(384,218*max(len(chosen),1)))
            for i,k in enumerate(chosen):
                for col,(im,point) in enumerate([(im_a,a[k]),(im_b,b[k])]):
                    x,y=np.rint(point).astype(int);crop=im.crop((x-96,y-96,x+96,y+96)).rotate(90)
                    ImageDraw.Draw(crop).ellipse((92,92,100,100),outline='red');canvas.paste(crop,(col*192,i*218+26))
                ImageDraw.Draw(canvas).text((3,i*218+5),f'dx,dy={b[k]-a[k]}',fill='white')
            canvas.save(args.output/f'{name}_temporal_crops.png')
            row={'physical_camera':name,'reference_path':str(path),'reference_sha256':sha256(path),**stats,
                 'correspondences':[{'reference':pa.tolist(),'current':pb.tolist(),'near_identity_inlier':bool(ok)}
                                    for pa,pb,ok in zip(a,b,accepted)]}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='correspondences'}),flush=True)
        atomic_json(args.output/'audit.json',{'uses_eval_rgb':False,'uses_mesh':False,'changes_calibration':False,
            'changes_prediction':False,'method':'Same-camera near-identity feature cluster; static objects require visual confirmation',
            'sources':sources,'reference_transforms_sha256':sha256(args.reference_data/'transforms.json'),
            'transforms_sha256':sha256(args.data/'transforms.json'),'script_sha256':sha256(Path(__file__)),
            'cameras':rows});return
    rows=[]
    for left,right in itertools.combinations(args.cameras,2):
        ka,da=cached[left];kb,db=cached[right]
        if da is None or db is None:continue
        pairs=mutual_matches(da,db);a=ka[pairs[:,0]];b=kb[pairs[:,1]];held=holdout(a)
        if (~held).sum()<50 or held.sum()<20:continue
        fitted,_=cv2.findFundamentalMat(a[~held],b[~held],cv2.USAC_MAGSAC,.75,.999,10000)
        if fitted is None or fitted.shape!=(3,3):continue
        fit_error=epipolar_distance(fitted,a,b);consistent=np.abs(fit_error)<.75
        fixed=epipolar_distance(fundamental(frames[left],frames[right]),a,b)
        check=held&consistent
        row={'left':left,'right':right,'mutual_matches':len(pairs),'fit_matches':int((~held).sum()),
             'held_matches':int(held.sum()),'held_independent_inliers':int(check.sum()),
             'frozen_calibration':summary(fixed[check]),'fitted_fundamental':summary(fit_error[check]),
             'diagnostic_fundamental_matrix':fitted.tolist(),
             'correspondences':[{'a':pa.tolist(),'b':pb.tolist(),'held':bool(h),'independent_inlier':bool(ok),
                                'frozen_signed_pixels':float(e),'fitted_signed_pixels':float(ef)}
                               for pa,pb,h,ok,e,ef in zip(a,b,held,consistent,fixed,fit_error)]}
        rows.append(row)
        # Native pairs selected by held-match residual, not using any eval image.
        choose=np.where(check)[0];choose=choose[np.argsort(-np.abs(fixed[choose]))][:6]
        if len(choose):
            im_a=Image.open(sources[left]['path']).convert('RGB');im_b=Image.open(sources[right]['path']).convert('RGB')
            canvas=Image.new('RGB',(384,218*len(choose)))
            for index,k in enumerate(choose):
                for col,(im,pt) in enumerate([(im_a,a[k]),(im_b,b[k])]):
                    x,y=np.rint(pt).astype(int);crop=im.crop((x-96,y-96,x+96,y+96)).rotate(90)
                    ImageDraw.Draw(crop).ellipse((92,92,100,100),outline='red')
                    canvas.paste(crop,(col*192,index*218+26))
                ImageDraw.Draw(canvas).text((3,index*218+5),f'fixed={fixed[k]:.2f}px fit={fit_error[k]:.2f}px',fill='white')
            canvas.save(args.output/f'{left}__{right}_held_crops.png')
        print(json.dumps({k:v for k,v in row.items() if k not in ('correspondences','diagnostic_fundamental_matrix')}),flush=True)
    atomic_json(args.output/'audit.json',{'uses_eval_rgb':False,'uses_mesh':False,'changes_calibration':False,
       'changes_prediction':False,'method':__doc__,'heldout_rule':'128px spatial blocks, modulo four',
       'sources':sources,'transforms_sha256':sha256(args.data/'transforms.json'),'script_sha256':sha256(Path(__file__)),
       'geometry_helper_sha256':sha256(Path(__file__).with_name('audit_source_epipolar_residuals.py')),'pairs':rows})


if __name__=='__main__':main()
