#!/usr/bin/env python3
"""Fresh calibrated train-background triangulation and planar scene hypotheses.

No inherited subject geometry, pose fitting, held-out RGB, or Gaussian splats.
Planes remain an unpromoted background representation until held-out evaluation.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from dec5_hair import cv_pose,project
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def intrinsic(f):
    return np.array([[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1.]])


def features(data,semantics,f,contrast_threshold=.04):
    image=cv2.imread(str(data/f['file_path']))
    if image is None:raise ValueError('Missing train RGB')
    probability=np.load(semantics/(f['physical_camera']+'.npz'))['probability']
    mask=cv2.erode((probability[...,2]>round(.98*65535)).astype(np.uint8),np.ones((33,33),np.uint8))
    if mask.shape!=image.shape[:2]:mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    keypoints,descriptors=cv2.SIFT_create(nfeatures=8000,contrastThreshold=contrast_threshold).detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask*255)
    return np.array([k.pt for k in keypoints],np.float64)+.5,descriptors


def triangulate(a,b,fa,fb,audit=None):
    uv1,d1=a;uv2,d2=b
    if d1 is None or d2 is None:return np.empty((0,3)),np.empty(0)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    forward=matcher.knnMatch(d1,d2,k=2);backward=matcher.knnMatch(d2,d1,k=2)
    reverse={m.queryIdx:m.trainIdx for pair in backward if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance}
    pairs=[(m.queryIdx,m.trainIdx) for pair in forward if len(pair)==2 for m,n in [pair]
           if m.distance<.7*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    if audit is not None:audit['mutual_ratio_matches']=len(pairs)
    if len(pairs)<8:return np.empty((0,3)),np.empty(0)
    p1,p2=[uv[np.asarray(pairs)[:,i]] for i,uv in enumerate([uv1,uv2])]
    pose1,pose2=cv_pose(fa),cv_pose(fb);r1,r2=pose1[:3,:3],pose2[:3,:3];c1,c2=pose1[:3,3],pose2[:3,3]
    k1,k2=intrinsic(fa),intrinsic(fb);t=r2.T@(c1-c2);r=r2.T@r1
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    fundamental=np.linalg.inv(k2).T@cross@r@np.linalg.inv(k1)
    h1=np.c_[p1,np.ones(len(p1))];h2=np.c_[p2,np.ones(len(p2))]
    lines2=h1@fundamental.T;lines1=h2@fundamental
    sampson=np.square((h2*lines2).sum(-1))/(np.square(lines1[:,:2]).sum(-1)+np.square(lines2[:,:2]).sum(-1)+1e-20)
    if audit is not None:
        audit['sampson_error_median_pixels']=float(np.sqrt(np.median(sampson)))
        audit['sampson_error_p10_pixels']=float(np.sqrt(np.quantile(sampson,.1)))
    keep=sampson<1.5**2;p1,p2=p1[keep],p2[keep]
    if audit is not None:audit['epipolar_inliers']=int(keep.sum())
    if not len(p1):return np.empty((0,3)),np.empty(0)
    camera1=k1@np.c_[r1.T,-r1.T@c1];camera2=k2@np.c_[r2.T,-r2.T@c2]
    h=cv2.triangulatePoints(camera1,camera2,p1.T,p2.T).T
    points=h[:,:3]/h[:,3:]
    q1,z1=project(points,fa);q2,z2=project(points,fb)
    error=np.maximum(np.linalg.norm(q1-p1,axis=-1),np.linalg.norm(q2-p2,axis=-1))
    v1=points-c1;v2=points-c2
    cosine=(v1*v2).sum(-1)/np.maximum(np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1),1e-20)
    valid=np.isfinite(points).all(-1)&(z1>0)&(z2>0)&(error<1.5)&(cosine<np.cos(np.deg2rad(.5)))
    if audit is not None:audit['accepted_feature_pairs']=np.asarray(pairs)[keep][valid].tolist()
    return points[valid],error[valid]


def fit_planes(points,threshold):
    rng=np.random.RandomState(42);remaining=np.arange(len(points));planes=[]
    for _ in range(4):
        if len(remaining)<200:break
        cloud=points[remaining];best=None;best_count=0
        for attempt in range(1000):
            p=cloud[rng.choice(len(cloud),3,replace=False)];n=np.cross(p[1]-p[0],p[2]-p[0]);length=np.linalg.norm(n)
            if length<1e-10:continue
            n/=length;d=-n@p[0];inliers=np.abs(cloud@n+d)<threshold;count=int(inliers.sum())
            if count>best_count:best=inliers;best_count=count
        if best_count<max(200,.05*len(points)):break
        selected=cloud[best];origin=np.median(selected,axis=0)
        _,_,basis=np.linalg.svd(selected-origin,full_matrices=False);normal=basis[-1];d=-normal@origin
        best=np.abs(cloud@normal+d)<threshold;selected=cloud[best]
        uv=(selected-origin)@basis[:2].T;hull=ConvexHull(uv)
        planes.append(dict(normal=normal.tolist(),offset=float(d),origin=origin.tolist(),basis=basis[:2].tolist(),
            hull_equations=hull.equations.tolist(),inlier_points=len(selected),
            median_distance=float(np.median(np.abs(selected@normal+d)))))
        remaining=remaining[~best]
    return planes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','semantics','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--views',type=int,default=16);a=p.parse_args()
    if not 8<=a.views<=32:raise ValueError('Bounded train-view pilot required')
    a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);cv2.setRNGSeed(42)
    meta=read(a.data/'transforms.json');train=[f for f in meta['frames'] if f['file_path'] in meta['train_filenames']]
    validation=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(train)!=64 or len(validation)!=1:raise ValueError('Authorized64+1 sequence required')
    centers=np.stack([cv_pose(f)[:3,3] for f in train]);reference=cv_pose(validation[0])[:3,3]
    chosen=np.argsort(np.linalg.norm(centers-reference,axis=-1),kind='stable')[:a.views]
    frames=[train[i] for i in chosen];cache=[];records=[]
    request=dict(data_sha256=sha(a.data/'transforms.json'),views=a.views,mask='p_bg>.98 eroded16 semantic pixels',
                 camera_names=[f['physical_camera'] for f in frames],sift_features=8000)
    old=read(a.output/'request.json')
    if old is not None and old!=request:raise ValueError('Background fit inputs changed')
    write(a.output/'request.json',request)
    for i,f in enumerate(frames):
        name=f['physical_camera'];cache_path=a.output/('features_'+name+'.npz')
        inputs=dict(rgb_sha256=sha(a.data/f['file_path']),semantic_sha256=sha(a.semantics/(name+'.npz')))
        cached=read(cache_path.with_suffix('.json'))
        if cached==inputs and cache_path.exists():
            with np.load(cache_path) as saved:values=(saved['uv'],saved['descriptors'])
        else:
            values=features(a.data,a.semantics,f)
            if values[1] is None:raise ValueError('No background descriptors in '+name)
            np.savez_compressed(cache_path,uv=values[0],descriptors=values[1]);write(cache_path.with_suffix('.json'),inputs)
        cache.append(values)
        records.append(dict(camera=name,**inputs,features=len(cache[-1][0])))
        write(a.output/'feature_audit.json',records)
        write(a.output/'progress.json',dict(phase='features',completed=i+1,total=len(frames)))
    c=centers[chosen];pairs=set()
    for i in range(len(c)):
        for j in np.argsort(np.linalg.norm(c-c[i],axis=-1))[1:5]:pairs.add(tuple(sorted((i,int(j)))))
    xyz=[];errors=[];pair_records=[]
    for index,(i,j) in enumerate(sorted(pairs)):
        detail={};points,error=triangulate(cache[i],cache[j],frames[i],frames[j],detail);xyz.append(points);errors.append(error)
        pair_records.append(dict(cameras=[frames[i]['physical_camera'],frames[j]['physical_camera']],points=len(points),**detail))
        write(a.output/'pair_audit.json',pair_records)
        write(a.output/'progress.json',dict(phase='triangulation',completed=index+1,total=len(pairs)))
    points=np.concatenate(xyz);error=np.concatenate(errors)
    np.savez_compressed(a.output/'raw_points.npz',xyz=points,reprojection_error=error)
    write(a.output/'geometry_audit.json',dict(points=len(points),pairs=len(pair_records),records=records))
    if len(points)<500:raise ValueError('Insufficient fresh background geometry')
    # Reject numerically distant matches using the fixed rig's scale, not GT.
    center=np.median(centers,axis=0);radius=np.median(np.linalg.norm(centers-center,axis=-1))
    valid=np.linalg.norm(points-center,axis=-1)<20*radius;points=points[valid];error=error[valid]
    threshold=.005*radius;planes=fit_planes(points,threshold)
    if not planes:raise ValueError('No supported background plane')
    np.savez_compressed(a.output/'points.npz',xyz=points,reprojection_error=error)
    write(a.output/'complete.json',dict(planes=planes,frames=frames,records=records,pairs=pair_records,
        points=len(points),median_reprojection_error=float(np.median(error)),plane_threshold=threshold,
        data_sha256=sha(a.data/'transforms.json'),uses_validation_rgb=False,uses_test_rgb=False,
        heldout_pose_used_for_source_subset_only=True,inherited_subject_geometry=False,
        calibration_refined=False,production_promoted=False,implementation_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
