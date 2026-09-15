"""Opt-in TSDF scale control on cached, fixed-pose train depth; no new MVS."""
import argparse
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from colmap_patchmatch_tsdf_campaign_common import append_jsonl

ROOT=Path('/mnt/data/dec5_forearm_tsdf_scale')
CONTROL=Path('/mnt/data/dec5_forearm_temporal_transfer/controls')
SCALES={'fine':.0005,'medium':.00075,'coarse':.001,'coarsest':.0015}


def command_for(command, output, voxel):
    command=list(command)
    command[0]=sys.executable
    command[command.index('--output')+1]=str(output)
    command[command.index('--voxel-length')+1]=str(voxel)
    # Only voxel spacing changes; truncation, weight, support and poses do not.
    return command


def run(frame):
    source=CONTROL/frame;root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    control=read(source/'complete.json')
    for n,h in control['hashes'].items():assert sha(source/n)==h
    original=next(command for stage,command in read(source/'commands.json') if stage=='fuse-tsdf')
    dataset=Path(original[original.index('--data')+1]);transforms=read(dataset/'transforms.json')
    inputs={str(dataset/'transforms.json'):sha(dataset/'transforms.json'),
            str(source/'complete.json'):sha(source/'complete.json'),
            str(source/'commands.json'):sha(source/'commands.json')}
    for row in transforms['frames']:
        if 'frame_train_' not in row['file_path']:continue
        assert row['physical_camera'] not in {'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
        path=dataset/row['depth_file_path'];inputs[str(path)]=sha(path)
    commands={name:command_for(original,root/name/'mesh.ply',voxel) for name,voxel in SCALES.items()}
    request=dict(frame=frame,commands=commands,source_hashes=inputs,
        script_sha256=sha(__file__),fuser_sha256=sha(original[1]),scales=SCALES,
        fixed_truncation=.004,fixed_weight=2,heldout_used=False,source_rgb_changed=False,
        geometry_scale_control=True,whole_coarse_mesh_is_not_a_production_candidate=True,
        production_video_changed=False,normalization_reference=str(source/'fuse-original/mesh.json'))
    atomic_json(root/'request.json',request);results=[]
    for name,command in commands.items():
        dest=root/name;dest.mkdir();log=dest/'stage.log';started=time.time()
        with log.open('wb') as stream:
            worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            while True:
                code=worker.poll()
                atomic_json(root/'progress.json',dict(frame=frame,stage=name,controller_pid=os.getpid(),worker_pid=worker.pid,returncode=code))
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
                text=log.read_text(errors='replace')
                append_jsonl(root/'checks.jsonl',dict(unix_time=time.time(),stage=name,controller_pid=os.getpid(),
                    worker_pid=worker.pid,returncode=code,gpu=gpu,log_tail=text.splitlines()[-3:],
                    suspicious_error=any(t in text.lower() for t in ['out of memory','cuda error','traceback']),
                    free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize))
                if code is not None:break
                time.sleep(10)
        if code:raise RuntimeError('TSDF control failed; keep workspace: '+str(log))
        metadata=read(dest/'mesh.json');reference=read(source/'fuse-original/mesh.json')
        for key in ['dataparser_scale','dataparser_transform']:
            np.testing.assert_allclose(metadata[key],reference[key],atol=1e-7,rtol=0)
        assert metadata['train_image_count']==62 and metadata['triangles']>0
        result=dict(variant=name,elapsed_seconds=time.time()-started,mesh_sha256=sha(dest/'mesh.ply'),
            metadata_sha256=sha(dest/'mesh.json'),vertices=metadata['vertices'],triangles=metadata['triangles'],
            connected_components=metadata['connected_components'],normalization_matches=True)
        atomic_json(dest/'complete.json',result);results.append(result)
        print(frame,name,result['triangles'],'triangles',round(result['elapsed_seconds'],1),'seconds',flush=True)
    atomic_json(root/'complete.json',dict(request_sha256=sha(root/'request.json'),variants=results,visual_status='pending',production_updated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',default='001037',choices=['001029','001037'])
    run(p.parse_args().frame)
