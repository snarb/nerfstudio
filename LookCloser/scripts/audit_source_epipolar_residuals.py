#!/usr/bin/env python3
"""Train-only epipolar consistency of registered, mesh-warped patch centres.

Nonzero residual cannot be eliminated by changing point depth alone if the patch
correspondence is correct. It is not by itself proof of bad calibration: motion,
view-dependent appearance and registration error can violate that assumption.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes


def projection(frame,offset):
    k=np.array([[frame['fl_x'],0,frame['cx']-offset],[0,frame['fl_y'],frame['cy']-offset],[0,0,1.]])
    pose=np.asarray(frame['transform_matrix']);w2c=np.diag([1.,-1.,-1.,1.])@np.linalg.inv(pose)
    return k,w2c


def fundamental(a,b,offset=.5):
    ka,wa=projection(a,offset);kb,wb=projection(b,offset)
    relative=wb@np.linalg.inv(wa);t=relative[:3,3]
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    return np.linalg.inv(kb).T@cross@relative[:3,:3]@np.linalg.inv(ka)


def project(frame,world,offset=.5):
    k,w=projection(frame,offset);p=k@(w[:3,:3]@world+w[:3,3])
    return p/p[2]


def signed_epipolar_distance(f,a,b):
    line=f@a
    return float(b@line/max(np.linalg.norm(line[:2]),1e-15))


def peak_offset(scores,u,v):
    if min(u,v)<1 or u>=scores.shape[1]-1 or v>=scores.shape[0]-1:return np.zeros(2)
    yy,xx=np.indices((3,3));x=xx.ravel()-1;y=yy.ravel()-1
    design=np.c_[np.ones(9),x,y,x*x,x*y,y*y]
    c=np.linalg.lstsq(design,scores[v-1:v+2,u-1:u+2].ravel(),rcond=None)[0]
    h=np.array([[2*c[3],c[4]],[c[4],2*c[5]]])
    if np.linalg.eigvalsh(h).max()>=-1e-8:return np.zeros(2)
    delta=-np.linalg.solve(h,c[1:3])
    return delta if (np.abs(delta)<=1).all() else np.zeros(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('render','registration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing audit')
    audit=json.loads((a.render/'reprojection_audit.json').read_text())
    data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={Path(f['file_path']).resolve():f for f in data['frames']}
    sources=[frames[Path(r['source_image'])] for r in audit['sources']]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if forbidden&{f['physical_camera'] for f in sources}:raise ValueError('Held-out source forbidden')
    target=next(f for f in data['frames'] if f['file_path'] in data['val_filenames'])
    dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    target_row=next(r for r in dm['images'] if Path(r['image']).resolve()==Path(target['file_path']).resolve())
    depth=fill_small_consistent_depth_holes(load_depth(Path(target_row['depth'])),max_area=1000,
                                          boundary_radius=4,max_relative_plane_rmse=.015)[0]
    offset=float(audit['pixel_center_offset']);kt,wt=projection(target,offset)
    def world(x,y):
        z=float(map_coordinates(depth,[[y],[x]],order=1)[0]);q=np.linalg.inv(kt)@np.array([x,y,1.])*z
        return (np.linalg.inv(wt)@np.r_[q,1])[:3]
    gray=[np.asarray(Image.open(a.render/'source_warps'/f'source_{r:02d}.png').convert('RGB'),np.float32)
          @np.array([.2126,.7152,.0722],np.float32)/255 for r in range(len(sources))]
    registration=json.loads(a.registration.read_text());rows=[]
    for name,expected in registration['input_hashes'].items():
        if sha256(a.render/'source_warps'/name)!=expected:
            raise ValueError('Registration audit belongs to different source warps')
    for row in registration['patches']:
        if row['ncc_best']<.9 or max(abs(row['dx']),abs(row['dy']))>=8:continue
        x,y,s=row['x'],row['y'],row['source_rank']
        local=depth[y-8:y+9,x-8:x+9]
        if (local<=0).any() or np.ptp(np.log(local))>.0075:continue
        ref=gray[0][y-24:y+24,x-24:x+24];search=gray[s][y-32:y+32,x-32:x+32]
        scores=cv2.matchTemplate(search,ref,cv2.TM_CCOEFF_NORMED);v,u=np.unravel_index(scores.argmax(),scores.shape)
        delta=peak_offset(scores,u,v)+[u-8,v-8]
        pa=project(sources[0],world(x,y),offset);pb=project(sources[s],world(x+delta[0],y+delta[1]),offset)
        f=fundamental(sources[0],sources[s],offset)
        rows.append({**row,'subpixel_shift_xy':delta.tolist(),
                     'signed_epipolar_pixels':signed_epipolar_distance(f,pa,pb),
                     'physical_camera':sources[s]['physical_camera']})
    summary=[]
    for region in ('hand_neck','face'):
        for s in range(1,len(sources)):
            selected=[r['signed_epipolar_pixels'] for r in rows if r['region']==region and r['source_rank']==s]
            if selected:summary.append({'region':region,'source_rank':s,'physical_camera':sources[s]['physical_camera'],
                'patches':len(selected),'signed_median':float(np.median(selected)),
                'absolute_median':float(np.median(np.abs(selected))),'absolute_p90':float(np.quantile(np.abs(selected),.9))})
    atomic_json(a.output,{'uses_eval_rgb':False,'changes_calibration':False,'changes_prediction':False,
       'interpretation':__doc__,'summary':summary,'patches':rows,'script_sha256':sha256(Path(__file__)),
       'registration_sha256':sha256(a.registration),'render_audit_sha256':sha256(a.render/'reprojection_audit.json')})
    print(json.dumps(summary))


if __name__=='__main__':main()
