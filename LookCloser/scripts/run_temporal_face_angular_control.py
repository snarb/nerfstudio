"""Supervised contiguous dynamic controls for the frozen face-angular policy.

Opt-in HD diagnostics. Reuse only audited immutable pilot times, stage new
train-only semantics in isolated paths, keep the actor/camera/mesh/color fixed
to each cinematic time. No production or 6K delivery changes.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
from build_train_hair_semantics import read,write,sha

ROOT=Path('/mnt/data/dec5_temporal_face_angular_control')
INPUTS=ROOT/'inputs'
OUTPUTS=ROOT/'renders'
BASE=Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
PILOT=Path('/mnt/data/dec5_face_angular_visibility')
MEDIA='/home/brans/lookcloser_temp/mediapipe_hand_env/bin/python'
CLIPS={'camera_tie':[f'{x:06d}' for x in range(1061,1108,2)],
       'nose_seam':[f'{x:06d}' for x in range(1087,1134,2)]}
FRAMES=sorted(set(sum(CLIPS.values(),[])))
REUSE={'001083','001119','001123','001127'}
SCRIPTS=['run_temporal_face_angular_control.py','stage_face_visibility_transfer.py',
    'transfer_face_angular_visibility.py','study_face_angular_visibility.py',
    'study_face_interior_visibility.py','joint_temporal_texture.py',
    'native_texture_footprint.py','diagnose_gap_texture_admission.py',
    'bake_joint_temporal_mesh.py','diffusion_mesh_repair.py',
    'calibrated_depth_witness.py','temporal_texture_view_prior.py']


def initialize():
    assert not ROOT.exists();ROOT.mkdir();(ROOT/'logs').mkdir();(ROOT/'states').mkdir()
    parent=read(BASE/'request.json');seal=read(PILOT/'final_manifest.json')
    for p,h in seal['hashes'].items():assert sha(p)==h,p
    inventory=[]
    for f in FRAMES:
        record=next(r for r in parent['inventory'] if r['frame_id']==f);assert record['index']<118
        base=BASE/'frames'/f;receipt=read(base/'complete.json')
        assert receipt['request_sha256']==sha(BASE/'request.json')
        for n,h in receipt['hashes'].items():assert sha(base/n)==h
        reused=f in REUSE;out=(PILOT if reused else OUTPUTS)/f/'consensus'
        inventory.append(dict(frame=f,output=str(out),reuse=reused,
            baseline_complete_sha256=sha(base/'complete.json'),
            reused_result_sha256=sha(out/'result.json') if reused else None))
    write(ROOT/'request.json',dict(clips=CLIPS,frames=FRAMES,inventory=inventory,
        scripts={n:sha(Path(__file__).with_name(n)) for n in SCRIPTS},
        baseline_request_sha256=sha(BASE/'request.json'),pilot_manifest_sha256=sha(PILOT/'final_manifest.json'),
        output_resolution=[1080,1920],fps=24,production_changed=False,geometry_changed=False,
        infer_python=MEDIA,one_source_rgb=True,face_policy_frozen=True))
    print('frames',len(FRAMES),'reuse',len(REUSE),'new',len(FRAMES)-len(REUSE),flush=True)


def verify():
    q=read(ROOT/'request.json');assert q['frames']==FRAMES and q['clips']==CLIPS
    for n,h in q['scripts'].items():assert sha(Path(__file__).with_name(n))==h,n
    assert sha(BASE/'request.json')==q['baseline_request_sha256']
    return q


def action(stage,frame):
    verify();assert frame in FRAMES and frame not in REUSE
    if stage in ['stage','infer']:
        import stage_face_visibility_transfer as semantic
        semantic.ROOT=INPUTS
        getattr(semantic,stage)(frame)
    elif stage=='prepare':
        import transfer_face_angular_visibility as transfer
        transfer.INPUTS=INPUTS
        source=inspect.getsource(transfer.prepare)
        old="out=pilot.ROOT/frame/'consensus'"
        assert source.count(old)==1;source=source.replace(old,"out=CLIP_OUTPUTS/frame/'consensus'")
        transfer.__dict__['CLIP_OUTPUTS']=OUTPUTS
        exec(compile(source,__file__+':prepare','exec'),transfer.__dict__);transfer.prepare(frame)
        path=OUTPUTS/frame/'consensus/request.json';q=read(path)
        q['input_hashes'][str(Path(__file__).resolve())]=sha(__file__)
        q['input_hashes'][str(ROOT/'request.json')]=sha(ROOT/'request.json')
        q['compiled_prepare_sha256']=hashlib.sha256(source.encode()).hexdigest();write(path,q)
    elif stage=='render':
        import study_face_angular_visibility as pilot
        pilot.ROOT=OUTPUTS;out,generated=pilot.configure('consensus',frame)
        assert read(out/'request.json')['generated_render_sha256']==generated
        pilot.base.torch.set_num_threads(2)
        with pilot.base.torch.inference_mode():pilot.base.render()
    else:raise ValueError(stage)


def worker(frame):
    verify();state=ROOT/'states'/(frame+'.json')
    for stage in ['stage','infer','prepare','render']:
        interpreter=MEDIA if stage=='infer' else sys.executable
        logpath=ROOT/'logs'/f'{frame}_{stage}.log'
        with logpath.open('x') as log:
            proc=subprocess.Popen([interpreter,__file__,'action','--frame',frame,'--stage',stage],stdout=log,stderr=subprocess.STDOUT)
            write(state,dict(frame=frame,stage=stage,frame_worker_pid=os.getpid(),stage_pid=proc.pid,terminal=False))
            code=proc.wait()
        if code:
            write(state,dict(frame=frame,stage=stage,exit_code=code,terminal=True));raise RuntimeError(str(logpath))
    out=OUTPUTS/frame/'consensus';r=read(out/'result.json')
    assert r['request_sha256']==sha(out/'request.json')
    for n,h in r['hashes'].items():assert sha(out/n)==h
    write(state,dict(frame=frame,stage='render_complete',terminal=True,exit_code=0,result_sha256=sha(out/'result.json')))


def supervise():
    q=verify();assert not (ROOT/'supervisor_result.json').exists()
    pending=[f for f in FRAMES if f not in REUSE];active=[];finished=[];failed=False
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OPENCV_IO_ENABLE_OPENEXR='1')
    while pending or active:
        while pending and len(active)<3 and not failed:
            f=pending.pop(0);log=(ROOT/'logs'/f'{f}_worker.log').open('x')
            p=subprocess.Popen([sys.executable,__file__,'worker','--frame',f],stdout=log,stderr=subprocess.STDOUT,env=env)
            active.append((f,p,log))
        live=[]
        for f,p,log in active:
            code=p.poll()
            if code is None:live.append((f,p,log))
            else:log.close();finished.append(dict(frame=f,pid=p.pid,exit_code=code));failed|=code!=0
        active=live
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
        statuses=[]
        for f,p,_ in active:
            path=ROOT/'states'/(f+'.json');statuses.append(dict(frame=f,pid=p.pid,live=p.poll() is None,state=read(path) if path.exists() else None))
        errors=[]
        for path in (ROOT/'logs').glob('*.log'):
            for line in path.read_text(errors='replace').splitlines():
                if any(x in line.lower() for x in ['out of memory','cuda error','traceback']):errors.append(dict(log=str(path),line=line[:240]))
        state=dict(utc=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),active=statuses,
            completed=len(finished),pending=len(pending),reused=len(REUSE),gpu=gpu.stdout.strip(),errors=errors,
            free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
        with (ROOT/'checks.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(state)+'\n')
        print('completed',len(finished),'active',[(x['frame'],x['state']['stage'] if x['state'] else 'starting') for x in statuses],'pending',len(pending),flush=True)
        if failed and not active:break
        if active:time.sleep(10)
    ok=not failed and len(finished)==len(FRAMES)-len(REUSE)
    write(ROOT/'supervisor_result.json',dict(finished=finished,reused=sorted(REUSE),not_started=pending,all_passed=ok))
    if not ok:raise RuntimeError('Preserve failed workspaces; no automatic restart')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','supervise','worker','action']);p.add_argument('--frame',choices=FRAMES);p.add_argument('--stage',choices=['stage','infer','prepare','render']);a=p.parse_args()
    if a.command=='init':initialize()
    elif a.command=='supervise':supervise()
    elif a.command=='worker':worker(a.frame)
    else:action(a.stage,a.frame)
