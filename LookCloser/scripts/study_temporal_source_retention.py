"""Two contiguous dynamic-camera clips for the frozen relative-source control.

Native HD diagnostics only, not a replacement/upscale of the delivered 6K video.
Changes pixel source retention at ratio .5; geometry/graph/color stay frozen.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from joint_temporal_texture import read,sha,atomic_json

BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
ROOT=Path('/mnt/data/dec5_temporal_source_retention')
CLIPS={'lipstick':[f'{f:06d}' for f in range(973,1020,2)],
       'late_face':[f'{f:06d}' for f in range(1087,1134,2)]}
FRAMES=sum(CLIPS.values(),[])


def subset(parent,frame):
    q=deepcopy(parent)
    q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame]
    q['source_rows']=[r for r in q['source_rows'] if Path(r['source_dataset']).name==frame]
    assert len(q['inventory'])==len(q['source_rows'])==1
    assert q['inventory'][0]['index']<118,'Do not compare against dissolved or actual-train endings'
    q['ordered_frame_ids']=[frame]
    return q


def prepare():
    import study_surface_ray_source_prior as study
    from review_jaw_repair_transfer import verified_image
    parent=study.engine.verify_request(BASE)
    assert not ROOT.exists();ROOT.mkdir();(ROOT/'logs').mkdir()
    compiled=hashlib.sha256(study.transformed('axis_relative').encode()).hexdigest()
    bindings={}
    for f in FRAMES:
        verified_image(BASE,f)
        bindings[str(BASE/'frames'/f/'complete.json')]=sha(BASE/'frames'/f/'complete.json')
        q=subset(parent,f)
        q.update(temporal_retention=dict(mode='axis_relative',relative_weight=.5,
            execution_sha256=compiled,baseline=str(BASE),baseline_request_sha256=sha(BASE/'request.json')),
            geometry_changed=False,source_masks_unchanged=True,diagnostic_only=True,
            artifact_free_approval=False,full_video_candidate=False)
        for n in [Path(__file__).name,'study_surface_ray_source_prior.py','surface_ray_source_prior.py']:
            q['script_hashes'][n]=sha(Path(__file__).with_name(n))
        out=ROOT/f;out.mkdir();(out/'frames').mkdir();atomic_json(out/'request.json',q)
    atomic_json(ROOT/'request.json',dict(clips=CLIPS,frames=FRAMES,
        baseline=str(BASE),baseline_request_sha256=sha(BASE/'request.json'),
        baseline_receipts=bindings,script_sha256=sha(__file__),execution_sha256=compiled,
        one_source_rgb=True,geometry_unchanged=True,output_resolution=[1080,1920],
        delivered_6k_video_unchanged=True,relative_weight=.5))


def worker(frame):
    import study_surface_ray_source_prior as study
    assert frame in FRAMES
    root=ROOT/frame;q=study.engine.verify_request(root)
    assert study.install('axis_relative')==q['temporal_retention']['execution_sha256']
    study.engine.torch.set_num_threads(2);study.engine.render(root,[frame])


def supervise():
    q=read(ROOT/'request.json');assert q['script_sha256']==sha(__file__)
    assert q['frames']==FRAMES and q['clips']==CLIPS
    assert not (ROOT/'supervisor_result.json').exists()
    pending=FRAMES.copy();active=[];finished=[];failed=False
    while pending or active:
        while pending and len(active)<3 and not failed:
            f=pending.pop(0);log=(ROOT/'logs'/f'{f}.log').open('x')
            process=subprocess.Popen([sys.executable,__file__,'worker','--frame',f],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OPENCV_IO_ENABLE_OPENEXR='1'))
            active.append((f,process,log))
        live=[]
        for f,p,log in active:
            code=p.poll()
            if code is None:live.append((f,p,log))
            else:
                log.close();finished.append(dict(frame=f,pid=p.pid,exit_code=code))
                if code:failed=True
        active=live
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        stages=[]
        for f,p,_ in active:
            status=ROOT/f/'progress.json'
            stages.append(dict(frame=f,pid=p.pid,live=p.poll() is None,progress=read(status) if status.exists() else None))
        errors={}
        for f in FRAMES:
            p=ROOT/'logs'/f'{f}.log'
            if p.exists():
                hits=[line for line in p.read_text().splitlines() if any(s in line.lower() for s in ['out of memory','cuda error','traceback'])]
                if hits:errors[f]=hits[-3:]
        status=dict(utc=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),active=stages,
            pending=len(pending),finished=len(finished),gpu=gpu.stdout.strip(),gpu_probe_exit=gpu.returncode,
            errors=errors,free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
        with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(status)+'\n')
        print('finished',len(finished),'active',[(r['frame'],r['progress']['stage'] if r['progress'] else 'starting') for r in stages],
            'pending',len(pending),'errors',len(errors),flush=True)
        if failed and not active:break
        if active:time.sleep(10)
    ok=len(finished)==len(FRAMES) and not failed
    atomic_json(ROOT/'supervisor_result.json',dict(finished=finished,not_started=pending,all_passed=ok))
    if not ok:raise RuntimeError('Failed clip render; preserve logs/workspaces')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','supervise','worker'])
    p.add_argument('--frame');a=p.parse_args()
    if a.stage=='worker':worker(a.frame)
    else:globals()[a.stage]()
