"""1685-only isolated native observer; identity reference is qualified1670.

No live native archive, DATA, source marker or NeRF state is written.
CPU packing/RAW transport precede native GPU demand; original qualified V6.2
streaming inference/capture/union runtime executes every camera unchanged.
"""
from pathlib import Path
import argparse, importlib, json, os, shlex, subprocess, sys, time
import numpy as np

V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
sys.path[:0]=[str(V/'operations/sam3_assist/code'),str(V/'code'),str(V/'code/LookCloser/scripts')]
import marshawn_sam3_assist as io
import marshawn_sam3_assist_v5 as assist
import marshawn_sam3_union_capture_v5 as capture
import marshawn_native_stream_v62 as streaming
import marshawn_tuning_controller as fleet

ROOT=V/'operations/sam3_assist/v5/frame001685'
RELEASE=ROOT/'observer_release.json'
REMOTE=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107')
SGPY='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python'
BATCH=V/'operations/sam3_assist/code/marshawn_sam3_union_batch_v62_runtime.py'


def guard():
    r=io.read(RELEASE)
    if r['frame']!=1685 or not r['observer_only']or r['reference_frame']!=1670:raise ValueError('Observer scope changed')
    for p,h in r['files_sha256'].items():
        if io.sha(p)!=h:raise ValueError('Observer frozen source/reference changed '+p)
    return r


def configure():
    ROOT.mkdir(parents=True,exist_ok=True)
    if RELEASE.exists():return guard()
    source=io.read(V/'sources/1685/source_ready.json')
    if source['frame']!=1685 or not source['source_verified_all66']or len(source['cameras'])!=66:raise ValueError('Need all66 current sources')
    files=[Path(__file__),BATCH,Path(assist.__file__),Path(io.__file__),Path(capture.__file__),Path(streaming.__file__),V/'operations/sam3_assist/code/marshawn_native_pool_v6.py',V/'operations/marshawn_union_resolution_v5.py',V/'sources/1685/source_ready.json',V/'frames/001670/native_semantic_ready.json']
    for c in source['cameras']:files += [V/'frames/001670/native_masks'/(c+'.npz'),V/'frames/001670/native_masks'/(c+'.json')]
    for p in files:
        if not p.exists():raise ValueError('Qualified identity/source file missing '+str(p))
    r=dict(kind='ROOT_AUTHORIZED_1685_ISOLATED_NATIVE_OBSERVER_V1',frame=1685,observer_only=True,reference_frame=1670,files_sha256={str(p):io.sha(p)for p in files},no_live_source_or_DATA_publication=True,time=time.time())
    io.write(RELEASE,r);return r


def pack():
    r=guard();config_path=ROOT/'all66_config.json'
    if config_path.exists():return config_path
    source=io.read(V/'sources/1685/source_ready.json');module=io.canonical(V/'code');old_seed=module.seed;old_policy=capture.seed_with_ownership_policy;cache={}
    def pinned_seed(campaign,frame,camera):
        if Path(campaign)!=V or frame!=1685:raise ValueError('Pinned reference escaped observer scope')
        p=V/'frames/001670/native_masks'/(camera+'.npz')
        with np.load(p,allow_pickle=False)as z:refs={a:module.small(z[a])for a in ['black','grey']}
        return refs,1670,io.sha(p)
    def cached_policy(mod,campaign,frame,camera,original_seed=None):
        if camera not in cache:cache[camera]=old_policy(mod,campaign,frame,camera,pinned_seed)
        result,receipt=cache[camera];refs,prior,h=result
        return ({a:v.copy()for a,v in refs.items()},prior,h),dict(receipt,observer_pinned_qualified_reference=True,policy='Explicit latest-qualified1670 identity pin; ownership-uncertain native pixels excluded before canonical small(); never finalforeground')
    module.seed=pinned_seed;capture.seed_with_ownership_policy=cached_policy;records=[]
    try:
        for camera in source['cameras']:
            variants={}
            for host in ['silver','olive']:
                task=ROOT/('tasks'if host=='silver'else'tasks_local')/camera
                destination=REMOTE/'operations/sam3_assist/v5/frame001685/tasks'/camera if host=='silver'else task
                spec=assist.pack(V,1685,camera,task,str(destination),str(REMOTE/'code'if host=='silver'else V/'code'),str(REMOTE/'tools/sam3_overlay')if host=='silver'else'/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay',V/'operations/marshawn_union_resolution_v5.py')
                if spec['reference']['frame']!=1670:raise ValueError('Unqualified mutable reference selected')
                variants[host]=dict(task=str(task/'task.json'),task_sha256=io.sha(task/'task.json'))
            records.append(dict(camera=camera,host='silver',**variants['silver'],variants=variants,original_native=None,original_native_sha256=None))
            io.write(ROOT/'pack_progress.json',dict(frame=1685,packed=len(records),total=66,time=time.time()))
    finally:module.seed=old_seed;capture.seed_with_ownership_policy=old_policy
    hosts=dict(silver=dict(address='ubuntu@silver-sable',python=SGPY,root=str(REMOTE/'operations/sam3_assist/v5/frame001685/all66_dispatch'),controller=str(REMOTE/'code/LookCloser/scripts/marshawn_tuning_controller.py'),adapter=str(REMOTE/'operations/sam3_assist/code/marshawn_sam3_assist_v5.py'),gpus=list(range(8))),olive=dict(python=sys.executable,root=str(ROOT/'all66_LOCAL_dispatch'),controller=str(V/'code/LookCloser/scripts/marshawn_tuning_controller.py'),adapter=str(Path(assist.__file__)),gpus=[0]))
    c=dict(version=5,campaign=str(V),frame=1685,root=str(ROOT/'all66_controller'),isolated_campaign=str(ROOT/'isolated_campaign'),records=records,hosts=hosts,files_sha256=r['files_sha256'],precollected={},source_ready_sha256=io.sha(V/'sources/1685/source_ready.json'),prepared_publication_allowed=False,stream_configuration=str(config_path),observer_only=True,pinned_identity_reference_frame=1670)
    io.write(config_path,c);return config_path


def stage():
    r=guard();import marshawn_mask_assist_full_v7 as old
    old.guard=lambda campaign,frame:dict(r,remote_root=str(REMOTE),remote_host='ubuntu@silver-sable',remote_python=SGPY)
    old.stage(V,1685)


def start(argv,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab')as stream:return subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def run():
    guard();held=fleet.lock(ROOT/'observer_controller.lock')
    if held is None:return
    io.write(ROOT/'observer_controller_identity.json',fleet.identity());p=pack();c=io.read(p);intent=V/'operations/native10_requests/1685_isolated_observer.json';control=Path(c['root']);nonce='1685_observer_'+io.sha(p)[:16]
    while True:
        guard();done=control/'all66_review_ready.json'
        if done.exists():
            io.write(intent,dict(status='complete',host_gpus=dict(silver=list(range(8)),olive=[0]),nonce=nonce,frame=1685,actual_state=str(control/'state.json'),completion=str(done),completion_sha256=io.sha(done),time=time.time()));return
        stage_id=ROOT/'stage_identity.json'
        if not(ROOT/'portable_inputs_staged.json').exists()and not fleet.alive(io.read(stage_id)if stage_id.exists()else None):start([sys.executable,__file__,'stage'],ROOT/'source_stage.log')
        ready=streaming.ready(c,control)
        if len(ready)>=8 or(ROOT/'portable_inputs_staged.json').exists():
            io.write(intent,dict(status='pending',host_gpus=dict(silver=list(range(8)),olive=[0]),nonce=nonce,frame=1685,source_bound_plan=str(p),source_bound_plan_sha256=io.sha(p),actual_state=str(control/'state.json'),source_ready_cameras=len(ready),observer_only=True,time=time.time()))
            old=io.read(control/'watchdog_identity.json')if(control/'watchdog_identity.json').exists()else None
            if not fleet.alive(old):start([sys.executable,str(BATCH),'watchdog','--config',str(p)],ROOT/'batch_watchdog.log')
        io.write(ROOT/'observer_progress.json',dict(frame=1685,staged=len(ready),total66=66,observer_only=True,time=time.time()));time.sleep(5)


def watchdog():
    guard();held=fleet.lock(ROOT/'observer_watchdog.lock')
    if held is None:return
    io.write(ROOT/'observer_watchdog_identity.json',fleet.identity())
    while not(ROOT/'all66_controller/all66_review_ready.json').exists():
        guard();p=ROOT/'observer_controller_identity.json';old=io.read(p)if p.exists()else None
        if not fleet.alive(old):start([sys.executable,__file__,'run'],ROOT/'observer_controller.log')
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['configure','run','stage','watchdog']);a=p.parse_args()
    if a.action=='configure':configure()
    elif a.action=='run':run()
    elif a.action=='stage':stage()
    else:watchdog()
