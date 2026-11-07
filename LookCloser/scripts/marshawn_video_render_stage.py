#!/usr/bin/env python3
"""Durable olive→silver native-texture stage, local-on-silver render, image gather.

No model/config relocation: silver checkpoints and YAMLs remain on their training
host. Only native source inputs, rendering requests and final images move.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
from marshawn_video_path import raw_file,sha,write


def read(p):return json.loads(Path(p).read_text())


def ssh(plan,argv):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',plan['remote'],shlex.join(argv)],
        capture_output=True,text=True,timeout=60)


def transfer(plan,source,destination,extra=()):
    return subprocess.run(['rsync','-a','--partial','--protect-args',*extra,str(source),plan['remote']+':'+str(destination)],
        capture_output=True,text=True)


def stage(plan,root):
    request=read(plan['path']);local_spec=read(plan['spec']);remote_root=Path(plan['remote_campaign'])
    check='import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==sys.argv[2]'
    checked=ssh(plan,[plan['remote_python'],'-c',check,str(remote_root/'frames/001660/data/transforms.json'),request['source_transforms_sha256']])
    if checked.returncode:raise RuntimeError('Silver camera-source transform binding differs or SSH unavailable: '+checked.stderr[-400:])
    # Wait for every native photograph actually used by the72 neural frames.
    source=read(request['source_transforms']);rows={r['physical_camera']:r for r in source['frames']}
    train={r['physical_camera'] for r in source['frames'] if r['file_path'] in source['train_filenames']}
    required=[];missing=[]
    for record in request['records']:
        if record['kind']=='actual_gt':names=[request['endpoint_train_camera']]
        else:
            center=np.asarray(record['camera']['transform_matrix'])[:3,3]
            names=sorted(train,key=lambda n:np.linalg.norm(np.asarray(rows[n]['transform_matrix'])[:3,3]-center))[:local_spec.get('source_count',16)]
        for name in names:
            path=raw_file(name,int(record['frame_id']),plan['raw_roots'])
            if path:required.append(dict(frame=record['frame_id'],camera=name,path=path,index=record['index']))
            else:missing.append(dict(frame=record['frame_id'],camera=name,index=record['index']))
    if any(m['index']<72 for m in missing):
        return dict(status='waiting_native_neural_sources',missing=missing)
    # Native RAWs are staged under one small canonical subtree per supplied root.
    raw_dest=[str(remote_root/'video/native_raw/shared')]
    for number,origin in enumerate(plan['raw_roots']):
        destination=Path(raw_dest[0])
        files=sorted({str(Path(r['path']).relative_to(origin)) for r in required if Path(origin) in Path(r['path']).parents})
        if not files:continue
        listing=root/f'raw_files_{number:02d}.txt';listing.write_text('\n'.join(files)+'\n')
        result=ssh(plan,['mkdir','-p',str(destination)])
        if result.returncode:raise RuntimeError('SSH unknown/unavailable: '+result.stderr[-400:])
        result=transfer(plan,str(Path(origin))+'/',str(destination)+'/',extra=('--files-from='+str(listing),))
        if result.returncode:raise RuntimeError('Native RAW staging incomplete: '+result.stderr[-400:])
    # Reuse immutable source algorithms; no scene-specific historical paths run.
    refs=root/'reference';(refs/'legacy/campaign').mkdir(parents=True,exist_ok=True)
    (refs/'legacy/hybrid_reference').mkdir(parents=True,exist_ok=True);(refs/'rig').mkdir(exist_ok=True)
    from shutil import copy2
    legacy=Path(local_spec['legacy_hybrid_root'])
    copy2(legacy/'campaign/misha50_hybrid.py',refs/'legacy/campaign/misha50_hybrid.py')
    for name in ['render_luster_hybrid.py','transfer_luster_hybrid.py','luster_hybrid_common.py']:
        copy2(legacy/'hybrid_reference'/name,refs/'legacy/hybrid_reference'/name)
    from prepare_marshawn_color_variants import PQ
    from marshawn_actor_coordinates import RIG
    copy2(PQ,refs/'pq.py');copy2(local_spec['photometry_fit'],refs/'photometry_fit.json')
    for name in ['cameras.txt','images.txt']:copy2(RIG/name,refs/'rig'/name)
    remote_video=remote_root/'video';result=ssh(plan,['mkdir','-p',str(remote_video)])
    if result.returncode:raise RuntimeError(result.stderr)
    result=transfer(plan,str(refs)+'/',str(remote_video/'reference')+'/')
    if result.returncode:raise RuntimeError(result.stderr)
    # Preserve actual silver absolute model paths; rebind ONLY the camera source
    # file location and renderer I/O explicitly, not serialized model objects.
    remote_path=dict(request,source_transforms=str(remote_root/'frames/001660/data/transforms.json'))
    write(root/'silver_path.json',remote_path)
    remote_spec=dict(local_spec,output=str(remote_video/'delivery_4K'),
        frame_root_template=str(remote_root/'frames/{frame_id}'),data_template=str(remote_root/'frames/{frame_id}/data'),
        legacy_hybrid_root=str(remote_video/'reference/legacy'),photometry_fit=str(remote_video/'reference/photometry_fit.json'),
        native_vendor_pq=str(remote_video/'reference/pq.py'),native_rig=str(remote_video/'reference/rig'),raw_roots=raw_dest)
    write(root/'silver_render_spec.json',remote_spec)
    for origin,destination in [(root/'silver_path.json',remote_video/'path.json'),(root/'silver_render_spec.json',remote_video/'render_spec.json')]:
        # These requests are immutable once frame output exists. Repeated stage
        # passes add actual GT source TIFFs without editing the request contents.
        result=transfer(plan,origin,destination)
        if result.returncode:raise RuntimeError(result.stderr)
    return dict(status='inputs_staged',native_source_files=len(required),missing_actual_GT=missing,
        remote_path=str(remote_video/'path.json'),remote_spec=str(remote_video/'render_spec.json'))


def run(a):
    plan=read(a.plan);root=Path(plan['stage_state']);root.mkdir(parents=True,exist_ok=True)
    held=(root/'controller.lock').open('a');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            result=stage(plan,root);result['time']=time.time()
            if result['status']=='inputs_staged':
                remote_root=Path(plan['remote_campaign']);script=remote_root/'code/LookCloser/scripts/marshawn_video_render_controller.py'
                # An existing durable campaign service can run this exact CLI.
                # A detached worker uses a lock and idempotent frame receipts.
                launcher='import subprocess,sys; p=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=open(sys.argv[1],"ab"),stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)'
                marker=ssh(plan,['test','-f',str(remote_root/'video/delivery_4K/controller.json')])
                alive=False
                if marker.returncode==0:
                    check='import json,os,sys; d=json.load(open(sys.argv[1])); os.kill(d["pid"],0); c=open("/proc/%s/cmdline"%d["pid"],"rb").read(); assert b"marshawn_video_render_controller.py" in c'
                    alive=ssh(plan,[plan['remote_python'],'-c',check,str(remote_root/'video/delivery_4K/controller.json')]).returncode==0
                if not alive:
                    command=[plan['remote_python'],str(script),'--spec',result['remote_spec'],'--path',result['remote_path'],
                        '--gpus',*[str(n) for n in plan['remote_gpus']]]
                    launched=ssh(plan,[plan['remote_python'],'-c',launcher,str(remote_root/'video/render_controller.log'),*command])
                    if launched.returncode:raise RuntimeError(launched.stderr)
                # Gather output images/receipts and MP4s, never checkpoints/YAMLs.
                destination=Path(plan['gather_output']);destination.mkdir(parents=True,exist_ok=True)
                gathered=subprocess.run(['rsync','-a','--partial','--protect-args','--exclude=*.npz','--exclude=worker.lock',
                    plan['remote']+':'+str(remote_root/'video/delivery_4K')+'/',str(destination)+'/'],capture_output=True,text=True)
                if gathered.returncode:raise RuntimeError(gathered.stderr[-400:])
                if (destination/'complete.json').exists():
                    for index in range(120):
                        unit=destination/'frames'/f'{index:04d}';receipt=read(unit/'complete.json')
                        for name,digest in receipt['images'].items():
                            if sha(unit/name)!=digest:raise ValueError('Gathered frame checksum differs')
                    write(root/'complete.json',dict(time=time.time(),frames=120,render_host=plan['remote'],no_model_relocation=True));return
            write(root/'status.json',result)
        except Exception as error:
            write(root/'status.json',dict(time=time.time(),status='retry_wait',error=str(error)))
        time.sleep(min(60,a.poll_seconds))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=60);run(p.parse_args())
