"""Repeat the unchanged central-pose experiment at late jaw/crown actor times."""
import argparse
from pathlib import Path
import subprocess
import time
import json
import probe_central_train_camera_workaround as probe
from joint_temporal_texture import read,sha,atomic_json

ROOT=Path('/mnt/data/dec5_central_train_pose_transfer')
FRAMES=('001123','001193')


def configure(frame):
    if frame not in FRAMES:raise ValueError('Unplanned actor frame')
    probe.ROOT=ROOT/frame;probe.FRAME=frame


def initialize():
    ROOT.mkdir(exist_ok=False)
    for frame in FRAMES:
        configure(frame);probe.initialize()
        request=read(probe.ROOT/'probe_request.json')
        request.update(transfer_script_sha256=sha(__file__),not_a_dynamic_video=True)
        for record in request['views']:
            path=probe.ROOT/record['view']/'request.json';q=read(path)
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            q.update(central_pose_temporal_transfer=True,actual_actor_frame=frame)
            atomic_json(path,q);record['request_sha256']=sha(path)
        atomic_json(probe.ROOT/'probe_request.json',request)
    atomic_json(ROOT/'request.json',dict(frames=list(FRAMES),script_sha256=sha(__file__),
        frame_requests={frame:sha(ROOT/frame/'probe_request.json') for frame in FRAMES},
        partial_diagnostic_only=True,video_changed=False))


def verify():
    request=read(ROOT/'request.json');assert sha(__file__)==request['script_sha256']
    for frame,h in request['frame_requests'].items():assert sha(ROOT/frame/'probe_request.json')==h


def render(worker,workers):
    verify()
    for frame in FRAMES:
        configure(frame);probe.render(worker,workers)


def review():
    import audit_central_train_camera_probe as audit
    verify()
    for frame in FRAMES:
        configure(frame);probe.review()
        audit.ROOT=probe.ROOT;audit.FRAME=frame;audit.run()


def check():
    ps=subprocess.check_output(['ps','-eo','pid,etime,pcpu,rss,args'],text=True)
    processes=[line.strip() for line in ps.splitlines() if 'python scripts/probe_central_train_pose_transfer.py render' in line and '/bin/bash' not in line]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    counts={frame:len(list((ROOT/frame).glob('*/frames/'+frame+'/complete.json'))) for frame in FRAMES}
    stat=dict(unix_time=time.time(),processes=processes,gpu=gpu,completed_views=counts)
    with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(stat)+'\n')
    print(counts,'workers',len(processes),'GPU',gpu,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render','review','check'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.action=='render':render(a.worker,a.workers)
    else:{'init':initialize,'review':review,'check':check}[a.action]()
