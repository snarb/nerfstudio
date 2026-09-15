"""Matched, resumable ray-angle / graph-label retention controls.

Starts with two native K/B views. Geometry and source masks are byte-preserved;
real target images are loaded only in the separate review action.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import render_smooth_temporal_mesh_video as engine
import study_native_texture_footprint as footprint
from view_consistent_source_quality import quality
from study_unwarped_head_texture import zero_registration
from wide_dynamic_camera_flight import install_source_masks
from surface_ray_source_prior import ray_weights,admit_quality,gather_relative
from joint_temporal_texture import read,sha,atomic_json
from render_midsequence_jaw_completion import baseline
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_surface_ray_source_prior')
FRAMES=['001193','001195']
VIEW='K004_B005_1210DS'
MODES=['axis_relative','ray','ray_relative']


def transformed(mode):
    source=footprint.transform(inspect.getsource(engine.render_one))
    axis="angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0]"
    ray="_ray_weights(points,centers,np.asarray(record['camera']['transform_matrix'])[:3,3],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])"
    replacements={
        'np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2':"_view_quality((directions*normal).sum(-1),length,'incidence2')",
        'np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2':"_view_quality((direction*normal[f]).sum(-1),length,'incidence2')*"+(axis+'[:,None]' if mode=='axis_relative' else ray),
    }
    if mode!='axis_relative':replacements['_early_quality(quality,'+axis+')']='_admit_quality(quality,'+ray.replace('(points,','(centroid,')+')'
    for old,new in replacements.items():
        if source.count(old)!=1:raise ValueError('Unexpected renderer statement: '+old)
        source=source.replace(old,new)
    return source


def install(mode):
    source=transformed(mode)
    engine.__dict__.update(_view_quality=quality,_early_quality=footprint.early_quality,
        angle_weights=footprint.angle_weights,_ray_weights=ray_weights,_admit_quality=admit_quality,
        _snap_centers=footprint.snap_centers,_relevant_tap=footprint.relevant_tap,
        sample=footprint.sample_native,bounded_warp=zero_registration)
    if mode!='ray':engine.gather_hard_rgb=lambda rgb,w,p:gather_relative(rgb,w,p,.5)
    exec(compile(source,__file__+':'+mode,'exec'),engine.__dict__)
    install_source_masks(engine)
    return hashlib.sha256(source.encode()).hexdigest()


def prepare():
    for frame in FRAMES:
        old=baseline(frame,VIEW);parent=engine.verify_request(old)
        verified_image(old,frame)
        for mode in MODES:
            q=deepcopy(parent)
            q.update(ray_source_control=dict(mode=mode,relative_weight=.5 if mode!='ray' else 0,
                execution_sha256=hashlib.sha256(transformed(mode).encode()).hexdigest(),
                baseline=str(old),baseline_request_sha256=sha(old/'request.json')),
                geometry_changed=False,source_masks_unchanged=True,partial_diagnostic_only=True,
                full_video_candidate=False,artifact_free_approval=False)
            for name in [Path(__file__).name,'surface_ray_source_prior.py']:
                q['script_hashes'][name]=sha(Path(__file__).with_name(name))
            out=ROOT/frame/mode;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
            if (out/'request.json').exists() and read(out/'request.json')!=q:raise ValueError('Request mismatch')
            atomic_json(out/'request.json',q)


def render(frame,mode):
    out=ROOT/frame/mode;q=engine.verify_request(out)
    assert install(mode)==q['ray_source_control']['execution_sha256']
    engine.torch.set_num_threads(2);engine.render(out,[frame])


def supervise():
    jobs=[(f,m) for f in FRAMES for m in MODES];active=[];finished=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    while jobs or active:
        while jobs and len(active)<3:
            f,m=jobs.pop(0);out=ROOT/f/m;log=open(out/'worker.log','a')
            p=subprocess.Popen([sys.executable,__file__,'render','--frame',f,'--mode',m],stdout=log,stderr=subprocess.STDOUT,env=env)
            active.append((f,m,p,log))
        live=[]
        for f,m,p,log in active:
            code=p.poll()
            if code is None:live.append((f,m,p,log))
            else:finished.append(dict(frame=f,mode=m,pid=p.pid,exit_code=code));log.close()
        active=live
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        stages=[]
        for f,m,p,_ in active:
            file=ROOT/f/m/'progress.json'
            stages.append(dict(frame=f,mode=m,pid=p.pid,progress=read(file) if file.exists() else None))
        status=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),active=stages,pending=len(jobs),finished=finished,gpu=gpu,
            free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
        with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(status)+'\n')
        print('active',len(active),'pending',len(jobs),'finished',len(finished),flush=True)
        if any(r['exit_code'] for r in finished):jobs=[]
        if active:time.sleep(30)
    atomic_json(ROOT/'supervisor_result.json',dict(finished=finished,all_six_passed=len(finished)==6 and all(r['exit_code']==0 for r in finished)))


def review():
    import cv2
    from classify_midsequence_jaw_seam import POLYGON
    results=[]
    for frame in FRAMES:
        gt_path=Path('/mnt/data/dec5_midsequence_jaw_completion')/frame/'seam_diagnosis/train_gt.png'
        gt=np.array(Image.open(gt_path));old=baseline(frame,VIEW);a,ar=verified_image(old,frame)
        ad=np.load(old/'frames'/frame/'target_depth.npz')['depth']
        mask=np.zeros(gt.shape[:2],np.uint8);cv2.fillPoly(mask,[np.array(POLYGON,np.int32)],1);inside=mask.astype(bool)
        gl=gt.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
        ims=[gt,a];labels=['real train GT','production']
        for mode,out in [('production',old)]+[(m,ROOT/frame/m) for m in MODES]:
            b,br=verified_image(out,frame)
            for key in ['camera','mesh_sha256','fixed_exposure','source_cameras']:assert ar[key]==br[key]
            bd=np.load(out/'frames'/frame/'target_depth.npz')['depth'];np.testing.assert_array_equal(ad,bd)
            l=b.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
            dark=inside&(l<gl-20)&(l<cv2.medianBlur(l,5)-15)
            ids=np.rot90(np.array(Image.open(out/'frames'/frame/'source_ids.png')))
            results.append(dict(frame=frame,mode=mode,depth_byte_equal=True,roi_pixels=int(inside.sum()),
                dark_ridge_pixels=int(dark.sum()),roi_no_source=int((inside&(ids==255)).sum()),
                changed_rgb_pixels=int(np.any(a!=b,2).sum()),new_black_pixels=int(((a.max(2)>0)&(b.max(2)==0)).sum()),
                own_camera_roi_pixels=int((inside&(ids==br['source_cameras'].index(VIEW))).sum()),
                image_sha256=sha(out/'frames'/frame/'frame.png'),complete_sha256=sha(out/'frames'/frame/'complete.json'),
                gt_sha256=sha(gt_path),visual_status='pending'))
            if mode!='production':ims.append(b);labels.append(mode)
        out=ROOT/frame/'review';out.mkdir(exist_ok=True)
        panel(out/'jaw.png',ims,labels,(460,1060,700,1210))
        panel(out/'head.png',ims,labels,(180,450,950,1250))
        panel(out/'overview.png',[np.array(Image.fromarray(im).resize((270,480))) for im in ims],labels,(0,0,270,480))
    atomic_json(ROOT/'comparison.json',dict(records=results,diagnostic_counts_not_face_metrics=True,production_promoted=False,
        geometry_changed=False,script_sha256=sha(__file__)))
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','supervise','review'])
    p.add_argument('--frame',choices=FRAMES);p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.action=='render':render(a.frame,a.mode)
    else:globals()[a.action]()
