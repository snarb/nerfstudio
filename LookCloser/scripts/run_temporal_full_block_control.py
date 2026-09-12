"""Supervised single-time real-depth TSDF activation control, separate from video.

Reproduces the historical geometry JPEG ingest only. Video texture exposure stays
fixed and is not taken from these temporary JPEGs. No existing defaults change.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import fcntl
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from joint_temporal_texture import SOURCE, CALIBRATION, read, sha, atomic_json
from colmap_patchmatch_tsdf_campaign_common import stage_fixed_calibration_dataset, validate_source_frame, append_jsonl
from import_colmap_mvs_depth_dataset import read_colmap_dense_array
import run_colmap_patchmatch_tsdf as runner

COLMAP=Path('/home/brans/lookcloser_temp/colmap_5509fffe_dev3_bundle/colmap_pinned')


def supervised(command, output, stage):
    log=output/'logs'/f'{stage}.log'; log.parent.mkdir(exist_ok=True)
    receipt=output/'stages'/f'{stage}.json'
    if receipt.exists():
        old=read(receipt)
        if old['command']!=command or old['request_sha256']!=sha(output/'request.json'):
            raise ValueError('Stage resume provenance mismatch')
        for p,h in old['retained_hashes'].items():
            if sha(p)!=h:raise ValueError('Stage output changed')
        print(f'stage={stage} verified_resume=True',flush=True);return
    start=time.monotonic()
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        while True:
            row={'utc':datetime.now(timezone.utc).isoformat(),'frame':output.name,'stage':stage,
                 'controller_pid':os.getpid(),'worker_pid':process.pid,'worker_alive':process.poll() is None,
                 'elapsed_seconds':time.monotonic()-start,'disk_free_bytes':shutil.disk_usage(output).free,
                 'photometric_maps':len(list((output/'pipeline/dense/stereo/depth_maps').glob('**/*.photometric.bin'))),
                 'geometric_maps':len(list((output/'pipeline/dense/stereo/depth_maps').glob('**/*.geometric.bin'))),
                 'gpu':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True).stdout,
                 'log_tail':log.read_text(errors='replace')[-1200:]}
            append_jsonl(output/'checks.jsonl',row);atomic_json(output/'progress.json',row)
            if process.poll() is not None:break
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        if process.returncode:raise RuntimeError(f'{stage} failed: {log}')
    paths=[]
    if stage.startswith('patchmatch-'):
        kind=stage.split('-')[-1];paths=list((output/'pipeline/dense/stereo/depth_maps').glob(f'**/*.{kind}.bin'))
        if len(paths)!=62:raise ValueError('Incomplete depth inventory')
    elif stage.startswith('fuse-'):
        path=Path(command[command.index('--output')+1]);paths=[path,path.with_suffix('.json')]
    elif stage=='convert':paths=list((output/'jpeg65').rglob('*.jpg'))+[output/'jpeg65/transforms.json',output/'jpeg65/conversion_manifest.json']
    else:
        # Small exported model/config and imported depth artifacts are bound for resume.
        paths=[p for p in (output/'pipeline').rglob('*') if p.is_file() and p.suffix in {'.json','.txt','.cfg','.gz'}]
    atomic_json(receipt,{'command':command,'request_sha256':sha(output/'request.json'),
                'elapsed_seconds':time.monotonic()-start,'retained_hashes':{str(p):sha(p) for p in paths}})
    print(f'stage={stage} complete seconds={time.monotonic()-start:.1f}',flush=True)


def run(frame,output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],check=True,text=True,capture_output=True).stdout
        if gpu.strip():raise RuntimeError(f'GPU occupied: {gpu}')
        build=subprocess.run([str(COLMAP),'-h'],check=True,text=True,capture_output=True)
        runner.validate_colmap_build(build.stdout+build.stderr,allow_unverified=False)
        source=SOURCE/frame;validate_source_frame(source);scripts=Path(__file__).parent
        names=[Path(__file__).name,'run_colmap_patchmatch_tsdf.py','colmap_patchmatch_tsdf_campaign_common.py',
               'convert_exr_nerfstudio_to_jpeg.py','export_nerfstudio_colmap_model.py','build_colmap_patch_match_config.py',
               'import_colmap_mvs_depth_dataset.py','fuse_depth_tsdf_mesh.py']
        request={'frame':frame,'source_transforms_sha256':sha(source/'transforms.json'),
                 'source_images':{r['file_path']:sha(source/r['file_path']) for r in read(source/'transforms.json')['frames']},
                 'calibration_sha256':sha(CALIBRATION),'colmap_build':build.stdout+build.stderr,
                 'binary_sha256':sha(COLMAP.parent/'bin/colmap'),'wrapper_sha256':sha(COLMAP),
                 'scripts':{n:sha(scripts/n) for n in names},'geometry_ingest':'historical per-image exposure, middle-gray .18, JPEG98 4:4:4',
                 'video_texture_ingest':'unchanged frozen global exposure and fixed camera profiles',
                 'controlled_variable':'per-view block activation versus full bounded block-union integration',
                 'eval_rgb_used_for_geometry':False,'masks_used':False}
        if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable control mismatch')
        atomic_json(output/'request.json',request)
        config=output/'config';config.mkdir(exist_ok=True)
        shutil.copyfile(CALIBRATION,config/'transforms.json')
        for n in names:shutil.copyfile(scripts/n,config/n)
        supervised([sys.executable,str(scripts/'convert_exr_nerfstudio_to_jpeg.py'),'--input',str(source),'--output',str(output/'jpeg65'),
                    '--middle-gray','.18','--exposure-mode','per-image','--exposure-percentile','70','--quality','98','--resume'],output,'convert')
        data=output/'staged63'
        if not data.exists():stage_fixed_calibration_dataset(source,output/'jpeg65',CALIBRATION,data)
        commands=[];original=runner.run_stage
        def capture(name,command,**kwargs):commands.append((name,command));return {'name':name}
        runner.run_stage=capture
        try:
            with redirect_stdout(io.StringIO()):
                runner.main(['--data',str(data),'--output-dir',str(output/'pipeline'),'--colmap-bin',str(COLMAP),'--dry-run'])
        finally:runner.run_stage=original
        atomic_json(output/'commands.json',commands)
        for name,command in commands:
            if name in {'texture-subset','raycast-mesh','hard-texture-render','fuse-tsdf'}:continue
            supervised(command,output,name)
        coverage=[]
        for p in sorted((output/'pipeline/dense/stereo/depth_maps').glob('**/*.geometric.bin')):
            d=read_colmap_dense_array(p)
            if d.shape!=(1080,1920,1):raise ValueError('Expected scalar full-resolution COLMAP depth map')
            d=d[...,0]
            if not np.isfinite(d).all() or (d>0).mean()<=0:raise ValueError('Invalid real depth map')
            coverage.append(float((d>0).mean()))
        if len(coverage)!=62:raise ValueError('Incomplete geometric depth inventory')
        atomic_json(output/'depth_qc.json',{'maps':62,'shape':[1080,1920],'coverage_mean':float(np.mean(coverage)),'coverage_min':float(np.min(coverage))})
        fuse=next(c for n,c in commands if n=='fuse-tsdf')
        for name,extra in [('fuse-original',[]),('fuse-full-block',['--tensor-full-block-integration'])]:
            command=list(fuse);folder=output/name;folder.mkdir(exist_ok=True)
            command[command.index('--output')+1]=str(folder/'mesh.ply');command+=extra
            supervised(command,output,name)
        atomic_json(output/'complete.json',{'request_sha256':sha(output/'request.json'),'visual_status':'pending',
                    'hashes':{n:sha(output/n) for n in ['fuse-original/mesh.ply','fuse-original/mesh.json','fuse-full-block/mesh.ply','fuse-full-block/mesh.json','depth_qc.json']}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',default='000971');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.frame,a.output)
