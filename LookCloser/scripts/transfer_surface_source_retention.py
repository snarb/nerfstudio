"""Transfer source-retention controls to moving views and a frozen face benchmark."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import subprocess
import sys
import os
import time
import json
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import study_surface_ray_source_prior as study
from joint_temporal_texture import read,sha,atomic_json
from render_midsequence_jaw_completion import baseline
from review_jaw_repair_transfer import verified_image,panel

ROOT=study.ROOT/'transfer'
MODES=['axis_relative','ray_relative']
CASES={f'{f}_moving':(f,baseline(f,'moving')) for f in ['000995','001193','001195']}
CASES['000995_native']=('000995',baseline('000995',study.VIEW))
CASES['heldout']=('001193',Path('/mnt/data/dec5_unwarped_head_texture/unwarped'))


def prepare():
    for case,(frame,old) in CASES.items():
        parent=study.engine.verify_request(old);verified_image(old,frame)
        for mode in MODES:
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame]
            q['ordered_frame_ids']=[frame];q['source_rows']=[r for r in q['source_rows'] if Path(r['source_dataset']).name==frame]
            q.update(retention_transfer=dict(case=case,mode=mode,baseline=str(old),baseline_request_sha256=sha(old/'request.json'),
                execution_sha256=hashlib.sha256(study.transformed(mode).encode()).hexdigest()),
                geometry_changed=False,partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False)
            for name in [Path(__file__).name,'study_surface_ray_source_prior.py','surface_ray_source_prior.py']:
                q['script_hashes'][name]=sha(Path(__file__).with_name(name))
            out=ROOT/case/mode;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
            if (out/'request.json').exists() and read(out/'request.json')!=q:raise ValueError('Transfer request mismatch')
            atomic_json(out/'request.json',q)


def render(case,mode):
    out=ROOT/case/mode;q=study.engine.verify_request(out)
    assert study.install(mode)==q['retention_transfer']['execution_sha256']
    study.engine.torch.set_num_threads(2);study.engine.render(out,[CASES[case][0]])


def supervise():
    jobs=[(c,m) for c in CASES for m in MODES];active=[];finished=[]
    while jobs or active:
        while jobs and len(active)<3:
            c,m=jobs.pop(0);log=(ROOT/c/m/'worker.log').open('a')
            p=subprocess.Popen([sys.executable,__file__,'render','--case',c,'--mode',m],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            active.append((c,m,p,log))
        live=[]
        for c,m,p,log in active:
            code=p.poll()
            if code is None:live.append((c,m,p,log))
            else:finished.append(dict(case=c,mode=m,pid=p.pid,exit_code=code));log.close()
        active=live
        stages=[]
        for c,m,p,_ in active:
            file=ROOT/c/m/'progress.json';stages.append(dict(case=c,mode=m,pid=p.pid,progress=read(file) if file.exists() else None))
        check=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),active=stages,pending=len(jobs),finished=finished,
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
            free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
        with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(check)+'\n')
        print('active',len(active),'pending',len(jobs),'finished',len(finished),flush=True)
        if any(r['exit_code'] for r in finished):jobs=[]
        if active:time.sleep(30)
    atomic_json(ROOT/'supervisor_result.json',dict(finished=finished,all_ten_passed=len(finished)==10 and all(r['exit_code']==0 for r in finished)))


def review():
    records=[]
    for case,(frame,old) in CASES.items():
        a,ar=verified_image(old,frame);ad=np.load(old/'frames'/frame/'target_depth.npz')['depth'];images=[a]
        for mode in MODES:
            out=ROOT/case/mode;b,br=verified_image(out,frame);images.append(b)
            for key in ['camera','mesh_sha256','source_cameras','fixed_exposure']:assert ar[key]==br[key]
            np.testing.assert_array_equal(ad,np.load(out/'frames'/frame/'target_depth.npz')['depth'])
            records.append(dict(case=case,mode=mode,frame=frame,depth_byte_equal=True,
                changed_rgb_pixels=int(np.any(a!=b,2).sum()),new_black_pixels=int(((a.max(2)>0)&(b.max(2)==0)).sum()),
                complete_sha256=sha(out/'frames'/frame/'complete.json'),baseline_complete_sha256=sha(old/'frames'/frame/'complete.json')))
        if 'moving' in case:
            box=(150,900,850,1600) if frame=='000995' else (180,750,800,1340)
        else:box=(150,450,1000,1300)
        panel(ROOT/case/'head.png',images,['production',*MODES],box)
        panel(ROOT/case/'overview.png',[np.array(Image.fromarray(im).resize((270,480))) for im in images],['production',*MODES],(0,0,270,480))
    atomic_json(ROOT/'comparison.json',dict(records=records,quality_metrics=False,visual_status='pending',production_promoted=False))
    # Reuse exactly the existing frozen held-out ROI scorer. This repeatedly
    # consulted benchmark is a regression check, not a fresh blind test.
    score_root=ROOT/'heldout';link=score_root/'production'
    if link.exists():assert link.is_symlink() and link.resolve()==CASES['heldout'][1].resolve()
    else:link.symlink_to(CASES['heldout'][1],target_is_directory=True)
    import evaluate_head_source_quality as evaluation
    evaluation.ROOT=score_root;evaluation.MODES=['production',*MODES];evaluation.score()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','supervise','review'])
    p.add_argument('--case',choices=list(CASES));p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.action=='render':render(a.case,a.mode)
    else:globals()[a.action]()
