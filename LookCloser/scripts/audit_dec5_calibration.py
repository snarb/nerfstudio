#!/usr/bin/env python3
"""Train-only static-background calibration audit, independent of hair fitting.

SIFT cycles across three images qualify correspondences without consulting the
provided poses. Known-pose epipolar residuals then test those poses. No bundle
adjustment, held-out RGB, camera changes or inherited sparse points are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import time

import cv2
import numpy as np

from dec5_hair import cv_pose, VALIDATION, TEST
from export_nerfstudio_colmap_model import export_model
from run_dec5_hair_campaign import atomic_json, sha


def prepare(campaign, output):
    output.mkdir(parents=True,exist_ok=False)
    data=campaign/'dataset';payload=json.loads((data/'transforms.json').read_text())
    frames=[f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):
        raise ValueError('Requires train59, no validation/test images')
    records=[]
    for frame in frames:
        path=campaign/'semantics'/(frame['physical_camera']+'.npz')
        probability=np.load(path)['probability'][...,2].astype(np.float32)/65535
        mask=cv2.erode((probability>.98).astype(np.uint8),np.ones((3,3),np.uint8))*255
        dest=output/'masks'/(frame['file_path']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(dest),mask)
        records.append(dict(name=frame['file_path'],camera=frame['physical_camera'],mask_sha256=sha(dest),
                            background_fraction=float((mask>0).mean())))
    export_model(data,output/'fixed_model',split='train',camera_model='PINHOLE')
    (output/'train_images.txt').write_text('\n'.join(f['file_path'] for f in frames)+'\n')
    centers=np.stack([cv_pose(f)[:3,3] for f in frames]);pairs=set()
    for i in range(len(frames)):
        for j in np.argsort(np.linalg.norm(centers-centers[i],axis=-1))[1:13]:
            pairs.add(tuple(sorted((frames[i]['file_path'],frames[j]['file_path']))))
    (output/'pairs.txt').write_text('\n'.join(a+' '+b for a,b in sorted(pairs))+'\n')
    atomic_json(output/'prepare.json',dict(frames=frames,records=records,pairs=len(pairs),
        dataset_sha256=sha(data/'transforms.json'),train_only=True,uses_sparse_points=False,
        implementation_sha256=sha(__file__)))


def calibrate_database(output):
    db=sqlite3.connect(str(output/'database.db'))
    images=db.execute('SELECT image_id,name FROM images').fetchall()
    payload=json.loads((output/'prepare.json').read_text());frames={f['file_path']:f for f in payload['frames']}
    if {name for _,name in images}!=set(frames):raise ValueError('Feature database split mismatch')
    columns={r[1] for r in db.execute('PRAGMA table_info(cameras)')}
    if not {'camera_id','model','width','height','params','prior_focal_length'}<=columns:
        raise ValueError('Unsupported COLMAP camera schema')
    for image_id,name in images:
        f=frames[name];camera_id=10000+image_id
        params=np.asarray([f[k] for k in ('fl_x','fl_y','cx','cy')],np.float64)
        db.execute('INSERT OR REPLACE INTO cameras(camera_id,model,width,height,params,prior_focal_length) VALUES(?,?,?,?,?,?)',
                   (camera_id,1,f['w'],f['h'],params.tobytes(),1))
        db.execute('UPDATE images SET camera_id=? WHERE image_id=?',(camera_id,image_id))
    db.commit();db.close()
    atomic_json(output/'camera_import.json',dict(images=len(images),pinhole_model_id=1,
        per_image_intrinsics=True,dataset_sha256=payload['dataset_sha256'],poses_changed=False))


def execute(stage,campaign,output,colmap):
    db=output/'database.db'
    if stage=='features':
        command=[str(colmap),'feature_extractor','--database_path',str(db),'--image_path',str(campaign/'dataset'),
            '--image_list_path',str(output/'train_images.txt'),'--ImageReader.mask_path',str(output/'masks'),
            '--ImageReader.camera_model','PINHOLE','--SiftExtraction.use_gpu','1',
            '--SiftExtraction.num_threads','2','--SiftExtraction.max_num_features','8192']
    else:
        calibrate_database(output)
        command=[str(colmap),'matches_importer','--database_path',str(db),'--match_list_path',str(output/'pairs.txt'),
                 '--match_type','pairs','--SiftMatching.use_gpu','1','--SiftMatching.num_threads','2']
    start=time.monotonic()
    with (output/(stage+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(output/(stage+'_result.json'),dict(command=command,returncode=result.returncode,seconds=time.monotonic()-start))
    if result.returncode:raise RuntimeError(stage+' failed; inspect its log')


def fundamental(a,b):
    pa,pb=cv_pose(a),cv_pose(b)
    r=pb[:3,:3].T@pa[:3,:3];t=pb[:3,:3].T@(pa[:3,3]-pb[:3,3])
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    def k(f):return np.array([[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1.]])
    return np.linalg.inv(k(b)).T@cross@r@np.linalg.inv(k(a))


def distances(uv_a,uv_b,f):
    a=np.column_stack((uv_a,np.ones(len(uv_a))));b=np.column_stack((uv_b,np.ones(len(uv_b))))
    la=a@f.T;lb=b@f;e=np.abs((b*la).sum(-1))
    return .5*e*(1/np.maximum(np.linalg.norm(la[:,:2],axis=-1),1e-12)+
                  1/np.maximum(np.linalg.norm(lb[:,:2],axis=-1),1e-12))


def measure(output):
    db=sqlite3.connect(str(output/'database.db'));payload=json.loads((output/'prepare.json').read_text())
    byname={f['file_path']:f for f in payload['frames']};images={i:byname[n] for i,n in db.execute('SELECT image_id,name FROM images')}
    keys={i:np.frombuffer(data,np.float32).reshape(rows,cols)[:,:2] for i,rows,cols,data in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    matches={}
    for pair,rows,cols,data in db.execute('SELECT pair_id,rows,cols,data FROM two_view_geometries'):
        if rows>=30:
            i,j=divmod(pair,2147483647);matches[i,j]=np.frombuffer(data,np.uint32).reshape(rows,cols)
    db.close();neighbors={i:set() for i in images};maps={}
    for (i,j),a in matches.items():
        neighbors[i].add(j);neighbors[j].add(i)
        maps[i,j]=dict(a);maps[j,i]=dict(a[:,::-1])
    rows=[];pooled=[];camera_errors={i:[] for i in images}
    for (i,j),a in matches.items():
        common=sorted(neighbors[i]&neighbors[j],key=lambda k:len(maps[i,k])+len(maps[j,k]),reverse=True)[:4]
        cyclic=np.zeros(len(a),bool)
        for k in common:
            cyclic |= np.array([maps[i,k].get(x,-1)==maps[j,k].get(y,-2) for x,y in a])
        error=distances(keys[i][a[:,0]],keys[j][a[:,1]],fundamental(images[i],images[j]))
        selected=error[cyclic]
        row=dict(cameras=[images[i]['physical_camera'],images[j]['physical_camera']],verified_matches=len(a),
                 cycle_supported=int(cyclic.sum()),all_median_px=float(np.median(error)),
                 cycle_median_px=float(np.median(selected)) if len(selected) else None,
                 cycle_p95_px=float(np.quantile(selected,.95)) if len(selected) else None)
        rows.append(row)
        if len(selected)>=30:
            pooled.append(selected);camera_errors[i].extend(selected);camera_errors[j].extend(selected)
    if not pooled:raise ValueError('Insufficient independently cycle-supported background matches')
    pool=np.concatenate(pooled)
    cameras=[dict(camera=images[i]['physical_camera'],matches=len(v),median_px=float(np.median(v)),p95_px=float(np.quantile(v,.95)))
             for i,v in camera_errors.items() if v]
    atomic_json(output/'metrics.json',dict(pairs=rows,cameras=cameras,cycle_matches=len(pool),
        median_px=float(np.median(pool)),p95_px=float(np.quantile(pool,.95)),
        metric='symmetric epipolar pixel distance under original poses/intrinsics',
        correspondence_filter='COLMAP verified matches plus independent feature-ID three-view cycles',
        calibration_optimized=False,uses_validation_rgb=False,uses_test_rgb=False,
        prepare_sha256=sha(output/'prepare.json'),database_sha256=sha(output/'database.db'),implementation_sha256=sha(__file__)))
    print('background cycle matches=%d median_px=%.4f p95_px=%.4f'%(len(pool),np.median(pool),np.quantile(pool,.95)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','features','match','measure'))
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--colmap',type=Path,default=Path('/usr/local/bin/colmap'));a=p.parse_args()
    if a.stage=='prepare':prepare(a.campaign,a.output)
    elif a.stage=='measure':measure(a.output)
    else:execute(a.stage,a.campaign,a.output,a.colmap)


if __name__=='__main__':main()
