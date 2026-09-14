#!/usr/bin/env python3
"""Isolated DEC5 source-neighbor ablation; frozen campaign code and RGB ingest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/data/dec5_patchmatch_source_count_ablation')
CAMPAIGN = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150')
REMOTE = Path('/fsx/oregon/dec5_patchmatch_source_count_ablation')
FRAMES = ('001083', '001123')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def run(command, log=None, **kwargs):
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            return subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    return subprocess.run(command, check=True, **kwargs)

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    code = ROOT/'frozen_code'
    code.mkdir(exist_ok=True)
    for source_code in (CAMPAIGN/'config/reconstruct_code').glob('*.py'):
        target_code = code/source_code.name
        if target_code.exists():
            assert sha(target_code) == sha(source_code)
        else:
            shutil.copyfile(source_code, target_code)
    sys.path.insert(0, str(code))
    from colmap_patchmatch_tsdf_campaign_common import stage_fixed_calibration_dataset
    calibration = CAMPAIGN/'config/calibration/transforms.json'
    assert sha(calibration) == '79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900'
    write(ROOT/'status.json', dict(status='preparing', frames=FRAMES, sources=[12,24,36], remote_root=str(REMOTE)))
    for frame in FRAMES:
        source = Path('/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080')/frame
        work = ROOT/'data'/frame
        if not (work/'jpeg65/conversion_manifest.json').exists():
            run([sys.executable,str(code/'convert_exr_nerfstudio_to_jpeg.py'),'--input',str(source),'--output',str(work/'jpeg65'),'--middle-gray','0.18','--exposure-mode','per-image','--exposure-percentile','70','--quality','98'], ROOT/'logs'/f'{frame}_convert.log', env=dict(os.environ,OPENCV_IO_ENABLE_OPENEXR='1'))
        if not (work/'staged63/staging_manifest.json').exists():
            stage_fixed_calibration_dataset(source, work/'jpeg65', calibration, work/'staged63')
    manifest = dict(frames=FRAMES, source_counts=[12,24,36], code={p.name:sha(p) for p in code.glob('*.py')},
                    inputs={f:{str(p.relative_to(ROOT/'data'/f/'staged63')):sha(p) for p in (ROOT/'data'/f/'staged63').rglob('*') if p.is_file()} for f in FRAMES},
                    old_baseline_reuse=False, reason='Original full depth maps and per-arm input/config receipts not retained; rerun12 on same host and frozen code.',
                    calibration_sha256=sha(calibration), rgb_ingest='per-image exposure percentile70 middlegray0.18 Reinhard sRGB JPEG98',
                    remote_root=str(REMOTE), no_source_masks=True)
    write(ROOT/'experiment_request.json',manifest)
    run(['ssh','ubuntu@dev3','mkdir','-p',str(REMOTE),str(REMOTE/'data')])
    run(['rsync','-a',str(code)+'/',f'ubuntu@dev3:{REMOTE}/frozen_code/'])
    run(['rsync','-a',__file__,f'ubuntu@dev3:{REMOTE}/run_patchmatch_source_count_ablation.py'])
    for frame in FRAMES:
        run(['rsync','-a',str(ROOT/'data'/frame/'staged63')+'/',f'ubuntu@dev3:{REMOTE}/data/{frame}/'])
    run(['rsync','-a',str(ROOT/'experiment_request.json'),f'ubuntu@dev3:{REMOTE}/'])
    write(ROOT/'status.json',dict(status='ready', frames=FRAMES, sources=[12,24,36], remote_root=str(REMOTE)))

def remote():
    root = REMOTE
    os.environ.update(PYTHONPATH=f'/home/ubuntu/repos/nerfstudio:{root}/frozen_code', OPENCV_IO_ENABLE_OPENEXR='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    import fcntl
    lock = (root/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    for count in (12,24,36):
        for frame in FRAMES:
            out = root/'arms'/frame/f'sources{count}'
            if (out/'pipeline_manifest.json').exists():
                continue
            log = root/'logs'/f'{frame}_{count}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            command = [sys.executable,str(root/'frozen_code/run_colmap_patchmatch_tsdf.py'),'--data',str(root/'data'/frame),'--output-dir',str(out),'--source-count',str(count),'--colmap-bin','/usr/local/bin/colmap']
            if out.exists():
                command.append('--resume')
            start = time.time()
            with log.open('a') as stream:
                process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                peak = 0
                while True:
                    gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
                    apps = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
                    try: peak = max(peak,int(gpu.split(',')[0]))
                    except ValueError: pass
                    maps = out/'dense/stereo/depth_maps'
                    record = dict(timestamp=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),worker_pid=process.pid,worker_alive=process.poll() is None,frame=frame,source_count=count,elapsed_seconds=time.time()-start,gpu=gpu,gpu_processes=apps,peak_gpu_mib=peak,photometric_maps=len(list(maps.glob('**/*.photometric.bin'))),geometric_maps=len(list(maps.glob('**/*.geometric.bin'))),free_bytes=shutil.disk_usage(root).free,log_tail=log.read_text(errors='replace')[-1800:])
                    with (root/'checks.jsonl').open('a') as check: check.write(json.dumps(record)+'\n')
                    write(root/'status.json',record)
                    if process.poll() is not None: break
                    time.sleep(60)
            write(root/'timings'/f'{frame}_{count}.json',dict(frame=frame,source_count=count,seconds=time.time()-start,peak_gpu_mib=peak,returncode=process.returncode))
            if process.returncode:
                raise RuntimeError(f'{frame}/{count} failed; inspect {log}')
    write(root/'status.json',dict(status='complete',controller_pid=os.getpid(),timestamp=datetime.now(timezone.utc).isoformat()))

def sample_memory():
    """Sample short TSDF/render peaks without launching another GPU workload."""
    while True:
        status=json.loads((REMOTE/'status.json').read_text())
        if status.get('status')=='complete':break
        try:os.kill(status['controller_pid'],0)
        except ProcessLookupError:break
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
        record=dict(timestamp=datetime.now(timezone.utc).isoformat(),frame=status.get('frame'),source_count=status.get('source_count'),gpu=gpu)
        with (REMOTE/'gpu_samples.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        time.sleep(2)

def checkpoint():
    """Explicit supervising-agent checkpoint in addition to automatic heartbeats."""
    status=json.loads((REMOTE/'status.json').read_text())
    alive={}
    for key in ['controller_pid','worker_pid']:
        pid=status.get(key)
        try:os.kill(pid,0);alive[key]=True
        except (ProcessLookupError,TypeError):alive[key]=False
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    failures=[]
    for log in (REMOTE/'arms').glob('*/*/logs/*.log'):
        tail=log.read_text(errors='replace')[-12000:].lower()
        if any(term in tail for term in ['out of memory','cuda_error','traceback']):failures.append(str(log))
    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),type='explicit_agent_supervision',alive=alive,gpu_processes=apps,failure_evidence=failures,free_bytes=shutil.disk_usage(REMOTE).free,progress=status)
    with (REMOTE/'manual_checks.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps(record))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','remote','sample_memory','checkpoint'])
    args=parser.parse_args()
    globals()[args.action]()
