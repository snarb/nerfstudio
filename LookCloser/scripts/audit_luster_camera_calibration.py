#!/usr/bin/env python3
"""Train-only SIFT epipolar audit; never changes supplied camera calibration."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from seed_colmap_from_nerfstudio import world_to_camera


def fundamental(a,b):
    def k(row):
        return np.array([[row['fl_x'],0,row['cx']-.5],[0,row['fl_y'],row['cy']-.5],[0,0,1.]])
    relative=world_to_camera(b)@np.linalg.inv(world_to_camera(a))
    x,y,z=relative[:3,3]
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.linalg.inv(k(b)).T@skew@relative[:3,:3]@np.linalg.inv(k(a))


def main(data,output):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    meta=json.loads((data/'transforms.json').read_text())
    rows={r['camera_id']:r for r in meta['frames'] if r['file_path'] in meta['train_filenames']}
    features={};images={};results=[];panels=[]
    sift=cv2.SIFT_create(nfeatures=6000)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    for aid,bid in [(10,11),(97,98),(151,152),(94,151),(149,151),(151,153)]:
        for cid in [aid,bid]:
            if cid not in features:
                image=cv2.imread(str(data/rows[cid]['file_path']))
                features[cid]=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
                images[cid]=image
        ka,da=features[aid];kb,db=features[bid]
        forward=matcher.knnMatch(da,db,k=2)
        reverse=matcher.knnMatch(db,da,k=2)
        back={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.7*n.distance}
        matches=[m for m,n in forward if m.distance<.7*n.distance and back.get(m.trainIdx)==m.queryIdx]
        pa=np.array([ka[m.queryIdx].pt for m in matches]);pb=np.array([kb[m.trainIdx].pt for m in matches])
        if len(matches)<8:
            results.append(dict(cameras=[aid,bid],status='insufficient_feature_matches',matches=len(matches)))
            continue
        _,inlier=cv2.findFundamentalMat(pa,pb,cv2.USAC_MAGSAC,1.5,.999,10000)
        keep=inlier.reshape(-1).astype(bool) if inlier is not None else np.ones(len(matches),dtype=bool)
        a=np.c_[pa[keep],np.ones(keep.sum())];b=np.c_[pb[keep],np.ones(keep.sum())]
        f=fundamental(rows[aid],rows[bid]);l_b=a@f.T;l_a=b@f
        residual=np.abs(np.sum(b*l_b,axis=-1))
        error=.5*residual*(1/np.maximum(np.linalg.norm(l_b[:,:2],axis=-1),1e-12)+
                           1/np.maximum(np.linalg.norm(l_a[:,:2],axis=-1),1e-12))
        row=dict(cameras=[aid,bid],matches=len(matches),independent_inliers=int(keep.sum()),
                 median_epipolar_px=float(np.median(error)),p90_epipolar_px=float(np.quantile(error,.9)),
                 within_1px_fraction=float((error<=1).mean()),calibration_changed=False)
        results.append(row)
        retained=[m for m,k in zip(matches,keep) if k]
        panel=cv2.drawMatches(images[aid],ka,images[bid],kb,retained[::max(1,len(retained)//40)],None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        im=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));im.thumbnail((1400,600));panels.append(im)
        print(json.dumps(row),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'calibration_audit.json').write_text(json.dumps(dict(train_only=True,calibration_changed=False,pairs=results),indent=2)+'\n')
    if panels:
        sheet=Image.new('RGB',(1400,sum(im.height for im in panels)),(30,30,30))
        y=0
        for im in panels:sheet.paste(im,(0,y));y+=im.height
        sheet.save(output/'matches.jpg')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    main(args.data,args.output)
