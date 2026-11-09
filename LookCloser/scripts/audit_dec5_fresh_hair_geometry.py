#!/usr/bin/env python3
"""Fresh train-only >=3-view SIFT hair tracks with fixed calibration; no fitting."""
from pathlib import Path
from itertools import combinations
import argparse
import cv2,numpy as np
from build_dec5_background_planes import intrinsic,triangulate
from dec5_hair import cv_pose,project
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def multiview_point(observations,frames,max_error=1.5):
    rows=[];centers=[]
    for uv,frame in zip(observations,frames):
        pose=cv_pose(frame);center=pose[:3,3];rotation=pose[:3,:3].T
        camera=intrinsic(frame)@np.c_[rotation,-rotation@center]
        rows.extend([uv[0]*camera[2]-camera[0],uv[1]*camera[2]-camera[1]]);centers.append(center)
    _,_,v=np.linalg.svd(rows,full_matrices=False);point=v[-1,:3]/v[-1,3]
    projected=[project(point[None],f) for f in frames]
    errors=np.array([np.linalg.norm(p[0][0]-uv) for p,uv in zip(projected,observations)])
    directions=point-np.array(centers);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    angle=float(np.arccos(np.clip((directions@directions.T).min(),-1,1)))
    if not np.isfinite(point).all() or any(p[1][0]<=0 for p in projected) or errors.max()>max_error or angle<np.deg2rad(3):return None
    return point,errors,angle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'fresh_sparse_hair';out.mkdir(exist_ok=True);cv2.setNumThreads(2);cv2.setRNGSeed(42)
    data=r/'frame/semantic_data';sem=r/'frame/semantics';meta=read(data/'transforms.json')
    train=[f for f in meta['frames'] if f['file_path'] in meta['train_filenames']]
    if len(train)!=64:raise ValueError('64 training cameras required')
    heldout=next(f for f in meta['frames'] if f['file_path'] in meta['val_filenames'])
    centers=np.array([cv_pose(f)[:3,3] for f in train]);order=np.argsort(np.linalg.norm(centers-cv_pose(heldout)[:3,3],axis=1),kind='stable')[:16]
    frames=[train[i] for i in order];cache=[];records=[];parents={}
    for i,f in enumerate(frames):
        image=cv2.imread(str(data/f['file_path']));prob=np.load(sem/(f['physical_camera']+'.npz'))['probability']
        mask=cv2.erode((prob[...,0]>.8*65535).astype(np.uint8),np.ones((5,5),np.uint8))
        keypoints,descriptors=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.04).detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask*255)
        uv=np.asarray([k.pt for k in keypoints],np.float64).reshape(-1,2)+.5;cache.append((uv,descriptors))
        records.append(dict(camera=f['physical_camera'],features=len(uv),rgb_sha256=sha(data/f['file_path']),semantic_sha256=sha(sem/(f['physical_camera']+'.npz'))))
        write(out/'progress.json',dict(stage='features',completed=i+1,total=16))
    def root(node):
        parents.setdefault(node,node)
        while parents[node]!=node:parents[node]=parents[parents[node]];node=parents[node]
        return node
    pairs=set();c=centers[order]
    for i in range(16):
        for j in np.argsort(np.linalg.norm(c-c[i],axis=1))[1:7]:pairs.add(tuple(sorted((i,int(j)))))
    pair_records=[]
    for n,(i,j) in enumerate(sorted(pairs)):
        audit={};points,error=triangulate(cache[i],cache[j],frames[i],frames[j],audit)
        for u,v in audit.pop('accepted_feature_pairs',[]):
            first,second=root((i,u)),root((j,v))
            if first!=second:parents[second]=first
        pair_records.append(dict(cameras=[frames[i]['physical_camera'],frames[j]['physical_camera']],accepted=len(points),**audit))
        write(out/'progress.json',dict(stage='matching',completed=n+1,total=len(pairs)))
    groups={}
    for node in list(parents):groups.setdefault(root(node),[]).append(node)
    xyz=[];max_errors=[];angles=[];track_records=[];ambiguous=0
    for nodes in groups.values():
        cameras=[n[0] for n in nodes]
        if len(cameras)!=len(set(cameras)):ambiguous+=1;continue
        if len(cameras)<3:continue
        result=multiview_point([cache[i][0][u] for i,u in nodes],[frames[i] for i,u in nodes])
        if result is None:continue
        point,error,angle=result;xyz.append(point);max_errors.append(error.max());angles.append(angle)
        track_records.append(dict(observations=[dict(camera=frames[i]['physical_camera'],feature=u,uv=cache[i][0][u].tolist()) for i,u in nodes],max_reprojection_pixels=float(error.max())))
    np.savez_compressed(out/'points.npz',xyz=np.asarray(xyz).reshape(-1,3),max_reprojection_pixels=np.asarray(max_errors),max_parallax_radians=np.asarray(angles))
    # Training-image overlays only; held-out RGB is never opened.
    for f in frames[:4]:
        image=cv2.imread(str(data/f['file_path']))
        if xyz:
            uv,z=project(np.asarray(xyz),f)
            for point,depth in zip(uv,z):
                if depth>0 and 0<=point[0]<image.shape[1] and 0<=point[1]<image.shape[0]:cv2.circle(image,tuple(np.round(point-.5).astype(int)),2,(0,255,0),-1)
        cv2.imwrite(str(out/(f['physical_camera']+'.jpg')),cv2.resize(np.rot90(image),(360,720),interpolation=cv2.INTER_AREA))
    write(out/'complete.json',dict(points=len(xyz),tracks=track_records,pairs=pair_records,features=records,
        ambiguous_camera_tracks_rejected=ambiguous,minimum_views=3,maximum_reprojection_pixels=1.5,minimum_parallax_degrees=3,
        data_sha256=sha(data/'transforms.json'),points_sha256=sha(out/'points.npz'),uses_validation_rgb=False,uses_test_rgb=False,
        poses_refined=False,inherited_geometry=False,independent_depth_ground_truth=False,production_promoted=False))
    print(dict(points=len(xyz),ambiguous_tracks_rejected=ambiguous))


if __name__=='__main__':main()
