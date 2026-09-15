"""Freeze actually inspected central-pose diagnostics; no video/mesh approval."""
import argparse
from pathlib import Path
import subprocess
import shutil
import time
import numpy as np
from joint_temporal_texture import read,sha,atomic_json,cameras
from probe_central_train_camera_workaround import ROOT as FIRST,PARENT
from probe_central_train_pose_transfer import ROOT,verify
from review_jaw_repair_transfer import verified_image
from render_smooth_temporal_mesh_video import verify_request
from audit_central_train_camera_probe import validate_camera

MANIFEST=ROOT/'artifact_manifest.json'
NOTES={
    '001037':'Torn hand/wrist persists in exact train views. Native G..K clip the hand; wide-FOV controls expose it. Rough brown hair rim remains. No hand workaround approved.',
    '001123':'Left views expose jagged under-chin boundary; right views retain dark jaw/neck seam. Crown gaps/fragments remain. No clean combined waypoint established.',
    '001193':'Left views reduce isolated under-jaw fleck but expose sharp false nose contour; right views retain under-jaw fleck. Hair/neck boundaries remain rough. No uniformly improved waypoint established.',
}


def run(check=False):
    if check:
        files=read(MANIFEST)['files']
        for p,h in files.items():
            if sha(p)!=h:raise ValueError('Changed artifact: '+p)
        print('Verified',len(files),'hashes');return
    if MANIFEST.exists():raise ValueError('Already frozen')
    verify();ps=subprocess.check_output(['ps','-eo','pid,etime,pcpu,rss,args'],text=True)
    names=['probe_central_train_camera_workaround.py render','probe_central_train_pose_transfer.py render','run_central_train_probe_isolated.py --']
    active=[line for line in ps.splitlines() if 'python' in line and '/bin/bash' not in line and any(n in line for n in names)]
    if active:raise ValueError('Workers still alive')
    external={};summaries=[]
    for frame,root in [('001037',FIRST),('001123',ROOT/'001123'),('001193',ROOT/'001193')]:
        request=read(root/'probe_request.json');assert len(request['views'])==15
        rows,_,_=cameras(frame);lookup={r['physical_camera']:r for r in rows}
        parent=read(PARENT/'request.json');entry=next(r for r in parent['inventory'] if r['frame_id']==frame)
        assert sha(request['mesh'])==request['mesh_sha256'];external[request['mesh']]=request['mesh_sha256']
        inspected=[root/'gt_overview.png'];views=[]
        for r in request['views']:
            view=r['view'];location=root/view;q=verify_request(location)
            assert sha(location/'request.json')==r['request_sha256']
            assert len(q['inventory'])==1 and q['inventory'][0]['frame_id']==frame
            e=q['inventory'][0];assert e['camera']==r['camera'] and e['mesh_sha256']==request['mesh_sha256']
            if view!='moving':validate_camera(e['camera'],lookup[r['physical_camera']],entry['camera'],'native' if view.startswith('native_') else 'flight_intrinsics')
            else:assert e['camera']==entry['camera']
            image,result=verified_image(location,frame);assert image.shape==(1920,1080,3)
            for n,h in q['script_hashes'].items():external[str(Path(__file__).with_name(n))]=h
            if frame=='001037':
                if view.startswith('native_'):panels=[root/'review'/(view+'_hand.png'),root/'head_review'/(view+'.png')]
                else:
                    panels=[root/'review'/(view+'_overview.png')]
                    if any(view.startswith('flight_intrinsics_'+c) for c in ['E','H','K']):panels.append(root/'review'/(view+'_hand.png'))
            else:panels=[root/'head_review'/(view+'.png')]
            inspected.extend(panels)
            views.append(dict(view=view,status='fail_workaround_gate',notes=NOTES[frame],
                inspected={str(p):sha(p) for p in panels},prediction_sha256=sha(location/'frames'/frame/'frame.png')))
        external.update(request['rgb_receipt']['source_rgb_hashes'])
        atomic_json(root/'visual_review.json',dict(frame=frame,reviewer='main_agent',status='reviewed_no_clean_workaround',
            inspected={str(p):sha(p) for p in inspected},views=views,video_changed=False,mesh_changed=False,
            interpretation='A waypoint screening failure, not proof that every possible path fails.',notes=NOTES[frame]))
        for rel in ['review/result.json','head_review/audit.json']:
            p=root/rel;review=read(p);review['visual_status']='see_explicit_inspected_panels_in_visual_review'
            for r in review['records']:r['visual_status']='overall_view_reviewed_see_visual_review'
            atomic_json(p,review)
        summaries.append(dict(frame=frame,completed=15,video_waypoint_approved=False))
    for p,h in external.items():assert sha(p)==h
    tests=Path('/mnt/data/dec5_central_train_probe_tests.log');assert '10 passed' in tests.read_text()
    logs=[tests,Path('/mnt/data/dec5_central_train_probe_init.log'),Path('/mnt/data/dec5_central_train_transfer_init.log')]
    for prefix in ['dec5_central_train_probe_worker','dec5_central_train_transfer_worker','dec5_central_train_transfer_isolated_worker']:
        logs += [Path('/mnt/data')/(prefix+str(w)+'.log') for w in range(3)]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    atomic_json(ROOT/'supervision_final.json',dict(unix_time=time.time(),workers_alive=False,gpu=gpu,
        free_bytes=shutil.disk_usage(ROOT).free,frames=summaries,completed_views=45,tests_passed=10,
        initial_launcher_failure='Repeated in-process renderer installation; isolated subprocess resume reused first-time receipts.',
        logs={str(p):sha(p) for p in logs},video_rerendered=False))
    repo=Path(__file__).parents[1]
    paths=[Path(__file__),repo/'scripts/audit_central_train_camera_probe.py',repo/'scripts/run_central_train_probe_isolated.py',
        repo/'tests/test_central_train_camera_probe.py',repo/'experiments/dec5_central_train_pose_workaround.md',*logs]
    files={str(p):sha(p) for base in [FIRST,ROOT] for p in base.rglob('*') if p.is_file() and p!=MANIFEST}
    files.update(external);files.update({str(p):sha(p) for p in paths})
    atomic_json(MANIFEST,dict(files=files,completed_views=45,actual_instants=3,tests_passed=10,
        status='reviewed_no_clean_central_waypoint',full_video_changed=False,goal_complete=False))
    print('Frozen',len(files),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
