#!/usr/bin/env python3
"""Fixed-calibration, mask-free, supervised PatchMatch for Luster opacity priors."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

from build_colmap_patch_match_config import explicit_source_views
from export_nerfstudio_colmap_model import export_model
from import_colmap_mvs_depth_dataset import read_colmap_dense_array


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(data,output,binary,canary):
    meta=json.loads((data/'transforms.json').read_text())
    train=set(meta['train_filenames'])
    frames=[r for r in meta['frames'] if r['file_path'] in train]
    if any('mask_path' in r for r in meta['frames']):
        raise ValueError('Source masks are forbidden')
    if canary:
        centers=np.array([r['transform_matrix'] for r in frames])[:,:3,3]
        selected=set()
        for cid in [97,151]:
            index=next(i for i,r in enumerate(frames) if r['camera_id']==cid)
            selected.update(np.argsort(np.linalg.norm(centers-centers[index],axis=-1),kind='stable')[:8].tolist())
        frames=[r for i,r in enumerate(frames) if i in selected]
    probe=subprocess.check_output([str(binary),'help'],stderr=subprocess.STDOUT,text=True)
    if not all(marker in probe for marker in ['3.13.0.dev0','5509fffe','with CUDA']):
        raise ValueError('Unverified CUDA COLMAP build')
    request=dict(data=str(data.resolve()),binary=str(binary.resolve()),binary_sha256=digest(binary),
                 transforms_sha256=digest(data/'transforms.json'),canary=canary,
                 references=[r['file_path'] for r in frames],source_count=min(12,len(frames)-1),
                 image_size=1920,depth_min=.01,depth_max=6.,iterations=3,
                 pixel_convention='COLMAP MVS integer samples: exported cx/cy = Nerfstudio cx/cy - 0.5',
                 source_rgb_sha256={r['file_path']:digest(data/r['file_path']) for r in frames})
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_text())!=request:
            raise ValueError('PatchMatch inputs/configuration changed; use a new output')
        if (output/'workspace_ready.json').exists():
            return request
    output.mkdir(parents=True,exist_ok=True)
    write(output/'request.json',request)
    dense=output/'dense'
    (dense/'images').mkdir(parents=True,exist_ok=True)
    if not (dense/'images/images').is_symlink():
        (dense/'images/images').symlink_to((data/'images').resolve(),target_is_directory=True)
    for sub in ['depth_maps','normal_maps','consistency_graphs']:
        (dense/'stereo'/sub).mkdir(parents=True,exist_ok=True)
    # PatchMatch uses integer sample locations in its native K, while NS rays
    # use pixel centers. This coordinate conversion makes their rays identical;
    # images are neither undistorted again nor resampled here.
    exported=json.loads(json.dumps(meta))
    exported['train_filenames']=[r['file_path'] for r in frames]
    for row in exported['frames']:
        row['cx']-=.5;row['cy']-=.5
    export_data=output/'calibration_for_mvs'
    write(export_data/'transforms.json',exported)
    if not (export_data/'images').is_symlink():
        (export_data/'images').symlink_to((data/'images').resolve(),target_is_directory=True)
    export_model(export_data,dense/'sparse',split='train',camera_model='PINHOLE')
    mapping=explicit_source_views(frames,source_count=request['source_count'])
    lines=[]
    for name,sources in mapping.items():
        lines.extend([name,', '.join(sources)])
    (dense/'stereo/patch-match.cfg').write_text('\n'.join(lines)+'\n')
    (dense/'stereo/fusion.cfg').write_text('\n'.join(mapping)+'\n')
    write(output/'workspace_ready.json',dict(request_sha256=digest(output/'request.json')))
    return request


def work(args):
    request=prepare(args.data,args.output,args.colmap_bin,args.canary)
    dense=args.output/'dense'
    logs=args.output/'logs';logs.mkdir(exist_ok=True)
    for name in request['references']:
        for kind in ['depth_maps','normal_maps','consistency_graphs']:
            (dense/'stereo'/kind/Path(name).parent).mkdir(parents=True,exist_ok=True)
    common=[str(args.colmap_bin),'patch_match_stereo','--workspace_path',str(dense),
            '--workspace_format','COLMAP','--PatchMatchStereo.max_image_size','1920',
            '--PatchMatchStereo.gpu_index','0','--PatchMatchStereo.depth_min',str(request['depth_min']),
            '--PatchMatchStereo.depth_max',str(request['depth_max']),
            '--PatchMatchStereo.num_iterations','3','--PatchMatchStereo.write_consistency_graph','1']
    for phase,flags in [('photometric',['--PatchMatchStereo.geom_consistency','0','--PatchMatchStereo.filter','0']),
                         ('geometric',['--PatchMatchStereo.geom_consistency','1','--PatchMatchStereo.geom_consistency_max_cost','3',
                            '--PatchMatchStereo.filter','1','--PatchMatchStereo.filter_min_ncc','0.1',
                            '--PatchMatchStereo.filter_min_triangulation_angle','1',
                            '--PatchMatchStereo.filter_min_num_consistent','3',
                            '--PatchMatchStereo.filter_geom_consistency_max_cost','1'])]:
        expected=[dense/'stereo/depth_maps'/(name+f'.{phase}.bin') for name in request['references']]
        if all(p.exists() for p in expected):
            continue
        write(args.output/'progress.json',dict(status='running',phase=phase,worker=os.getpid()))
        with (logs/(phase+'.log')).open('a') as log:
            subprocess.run(common+flags,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not all(p.exists() for p in expected):
            raise RuntimeError(f'COLMAP did not write all expected {phase} maps')
    rows=[]
    for name in request['references']:
        path=dense/'stereo/depth_maps'/(name+'.geometric.bin')
        depth=read_colmap_dense_array(path)[...,0]
        valid=np.isfinite(depth)&(depth>0)
        rows.append(dict(image=name,shape=list(depth.shape),sha256=digest(path),
                         coverage=float(valid.mean()),median=float(np.median(depth[valid])) if valid.any() else 0.,
                         spread=float(np.std(depth[valid])) if valid.any() else 0.))
    # A build canary is stricter than accepting arbitrary maps as measurements.
    if args.canary and (min(r['coverage'] for r in rows)<.01 or min(r['spread'] for r in rows)<1e-4
                        or len({r['sha256'] for r in rows})!=len(rows)):
        write(args.output/'canary_failure.json',rows)
        raise ValueError('Canary depths are empty, degenerate or duplicated')
    write(args.output/'depth_inventory.json',dict(maps=rows,train_only=True,mask_free=True,request=request))
    write(args.output/'progress.json',dict(status='review_required',maps=len(rows),worker=os.getpid()))


def supervise(args):
    args.output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,__file__,'--data',str(args.data),'--output',str(args.output),
             '--colmap-bin',str(args.colmap_bin),'--worker']+(['--canary'] if args.canary else [])
    with (args.output/'controller_stdout.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def stop(*_):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
            raise SystemExit(130)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while True:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
            processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
            progress=json.loads((args.output/'progress.json').read_text()) if (args.output/'progress.json').exists() else {}
            maps={kind:len(list((args.output/'dense/stereo/depth_maps').rglob(f'*.{kind}.bin')))
                  for kind in ['photometric','geometric']}
            errors=[p for p in (args.output/'logs').glob('*.log') if 'out of memory' in p.read_text(errors='replace')[-8192:].lower()]
            row=dict(time=time.time(),controller=os.getpid(),worker=child.pid,exit=child.poll(),
                     progress=progress,maps=maps,gpu=gpu.stdout.strip(),gpu_processes=processes.stdout.strip(),oom=bool(errors))
            with (args.output/'supervision.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(f"patchmatch={progress.get('phase','setup')} maps={maps} gpu={row['gpu']}",flush=True)
            if child.poll() is not None:
                if child.returncode:
                    print((args.output/'controller_stdout.log').read_text(errors='replace')[-2000:],flush=True)
                return child.returncode
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--colmap-bin',type=Path,required=True)
    p.add_argument('--canary',action='store_true')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:
        work(args)
    else:
        raise SystemExit(supervise(args))
