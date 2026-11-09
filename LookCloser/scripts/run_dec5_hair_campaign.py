#!/usr/bin/env python3
"""Explicit one-frame DEC5 campaign stages; no production paths are modified."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dec5_hair import VALIDATION, TEST, camera_split, mvs_camera_model


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))


def prepare(source, root, frame_id='001083'):
    """Hash/decode all fitting inputs; reserve test RGB without decoding it."""
    import cv2
    import numpy as np
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    original = json.loads((source/'transforms.json').read_text())
    split = camera_split(original['frames'])
    if list(split.values()).count('train') != 59:
        raise ValueError('Expected exactly 65 physical cameras / 59 train')
    if (root/'prepare.json').exists():
        raise FileExistsError('Preparation is immutable; use a new campaign root')
    output = root/'dataset'
    (output/'images').mkdir(parents=True, exist_ok=True)
    payload = {k: deepcopy(v) for k, v in original.items()
               if k not in ('frames', 'ply_file_path', 'train_filenames', 'val_filenames', 'test_filenames')}
    payload.update(frames=[], train_filenames=[], val_filenames=[], test_filenames=[])
    records, tests = [], []
    for frame in original['frames']:
        name = frame['physical_camera']
        partition = split[name]
        path = source/frame['file_path']
        if partition == 'test':
            tests.append(dict(frame=deepcopy(frame), source_path=str(path.resolve())))
            continue
        x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if x is None or x.shape != (frame['h'], frame['w'], 3) or not np.isfinite(x).all():
            raise ValueError('Invalid fitting EXR: '+str(path))
        # Historical fixed display transform; never fitted to validation RGB.
        linear = np.maximum(x, 0)*10.570320292648677
        display = linear/(1+linear)
        display = np.where(display <= .0031308, 12.92*display,
                           1.055*np.power(display, 1/2.4)-.055)
        prefix = 'train' if partition == 'train' else 'eval'
        relative = 'images/frame_'+prefix+'_'+name+'.png'
        if not cv2.imwrite(str(output/relative), np.rint(np.clip(display, 0, 1)*255).astype(np.uint8)):
            raise IOError('PNG write failed')
        row = deepcopy(frame)
        row['file_path'] = relative
        for key in ('depth_file_path', 'mask_path'):
            row.pop(key, None)
        payload['frames'].append(row)
        payload['train_filenames' if partition == 'train' else 'val_filenames'].append(relative)
        if partition == 'validation':
            payload['test_filenames'].append(relative)
        records.append(dict(camera=name, split=partition, original_path=str(path.resolve()),
                            original_sha256=sha(path), image_sha256=sha(output/relative)))
    atomic_json(output/'transforms.json', payload)
    atomic_json(root/'reserved_test.json', dict(records=tests, decoded=False, use='final locked evaluation only'))
    receipt = dict(frame=frame_id, train_count=59, validation=list(VALIDATION), test=list(TEST),
                   units='calibration units; metric scale not independently certified',
                   depth_convention='OpenCV camera-z', source_transforms_sha256=sha(source/'transforms.json'),
                   dataset_transforms_sha256=sha(output/'transforms.json'), records=records,
                   exposure_gain=10.570320292648677, display='Reinhard then sRGB',
                   inherited_sparse_points=False, inherited_color_fit=False,
                   time=datetime.now(timezone.utc).isoformat())
    atomic_json(root/'prepare.json', receipt)
    print('prepared frame='+frame_id+' train=59 validation=3 reserved_test=3', flush=True)


def resources():
    """Query the assigned physical GPU; legacy single-GPU default is retained."""
    import csv
    selector=os.environ.get('LOOKCLOSER_GPU_INDEX')
    args=['nvidia-smi']+(['--id='+selector] if selector else [])
    result=subprocess.run(args+['--query-gpu=uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    row=next(csv.reader(result.stdout.splitlines(),skipinitialspace=True));uuid=row[0];values=[int(x) for x in row[1:]]
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    entries=list(csv.reader(processes.stdout.splitlines(),skipinitialspace=True))
    return dict(gpu_uuid=uuid,gpu_total_mib=values[0],gpu_used_mib=values[1],gpu_utilization=values[2],
                processes=[', '.join(r[1:]) for r in entries if not selector or r[0]==uuid])


def patchmatch(data, output, colmap, image_size=960):
    """Train-only verified fixed-pose MVS; preserve raw maps before any fusion."""
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    payload = json.loads((data/'transforms.json').read_text())
    frames = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):
        raise ValueError('MVS requires exactly the campaign train inventory')
    export_camera_model = mvs_camera_model(frames, payload)
    already_undistorted = export_camera_model == 'PINHOLE'
    colmap_env = os.environ.copy()
    private_lib = colmap.parent.parent/'lib'
    if private_lib.is_dir():
        colmap_env['LD_LIBRARY_PATH'] = str(private_lib)+(':'+colmap_env['LD_LIBRARY_PATH'] if colmap_env.get('LD_LIBRARY_PATH') else '')
    probe = subprocess.run([str(colmap), '-h'], capture_output=True, text=True, check=True, env=colmap_env)
    build = probe.stdout+probe.stderr
    if not all(s in build for s in ('3.13.0.dev0', '5509fffe', 'with CUDA')):
        raise ValueError('Unverified COLMAP build')
    model, dense = output/'fixed_model', output/'dense'
    stages = [
        ('export', [sys.executable, str(scripts/'export_nerfstudio_colmap_model.py'),
                    '--data', str(data), '--output-model', str(model), '--split', 'train',
                    '--camera-model', export_camera_model]),
        ('native-workspace', [sys.executable, str(scripts/'prepare_native_colmap_workspace.py'),
                      '--data', str(data), '--model', str(model), '--output', str(dense), '--colmap', str(colmap)])
        if already_undistorted else
        ('undistort', [str(colmap), 'image_undistorter', '--image_path', str(data),
                      '--input_path', str(model), '--output_path', str(dense),
                      '--output_type', 'COLMAP', '--max_image_size', str(image_size), '--copy_policy', 'copy']),
        ('sources', [sys.executable, str(scripts/'build_colmap_patch_match_config.py'), '--data', str(data),
                     '--output', str(dense/'stereo/patch-match.cfg'), '--source-count', '12', '--split', 'train']),
    ]
    common = [str(colmap), 'patch_match_stereo', '--workspace_path', str(dense),
              '--workspace_format', 'COLMAP', '--PatchMatchStereo.max_image_size', str(image_size),
              '--PatchMatchStereo.gpu_index', '0', '--PatchMatchStereo.depth_min', '4.5',
              '--PatchMatchStereo.depth_max', '20', '--PatchMatchStereo.num_iterations', '3',
              '--PatchMatchStereo.write_consistency_graph', '1']
    stages += [('photometric', common+['--PatchMatchStereo.geom_consistency', '0', '--PatchMatchStereo.filter', '0']),
               ('geometric', common+['--PatchMatchStereo.geom_consistency', '1', '--PatchMatchStereo.filter', '1',
                '--PatchMatchStereo.geom_consistency_max_cost', str(6*image_size/1920),
                '--PatchMatchStereo.filter_min_ncc', '.1',
                '--PatchMatchStereo.filter_min_triangulation_angle', '1',
                '--PatchMatchStereo.filter_min_num_consistent', '2',
                '--PatchMatchStereo.filter_geom_consistency_max_cost', str(2*image_size/1920)])]
    atomic_json(output/'request.json', dict(data_sha256=sha(data/'transforms.json'), image_size=image_size,
                train_count=59, source_count=12, build=build, units='calibration units',
                metric_scale_certified=False, depth_convention='OpenCV camera-z', stages=stages,
                export_camera_model=export_camera_model, already_undistorted=already_undistorted,
                implementation_sha256=sha(__file__)))
    for name, command in stages:
        atomic_json(output/'progress.json', dict(stage=name, status='running', pid=os.getpid(),
                    time=datetime.now(timezone.utc).isoformat()))
        started = time.time()
        with (output/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=colmap_env if command[0] == str(colmap) else None)
            fatal = None
            while child.poll() is None:
                time.sleep(2)
                with (output/(name+'.log')).open('rb') as current:
                    current.seek(max(0, current.seek(0, 2)-65536))
                    tail = current.read().decode(errors='replace')
                if 'Check failed:' in tail or 'out of memory' in tail.lower():
                    fatal = 'COLMAP logged a fatal error even if its eventual exit code is zero'
                    child.terminate()  # This exact Popen-owned child, never another worker.
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
            code = child.wait()
        atomic_json(output/(name+'.json'), dict(command=command, returncode=code, fatal_error=fatal,
                    seconds=time.time()-started))
        if code or fatal:
            raise RuntimeError('MVS '+name+' failed; see '+str(output/(name+'.log')))
        if name in ('photometric', 'geometric'):
            actual = len(list((dense/'stereo/depth_maps').rglob('*.'+name+'.bin')))
            if actual != len(frames):
                raise RuntimeError('MVS '+name+' produced '+str(actual)+'/'+str(len(frames))+' depth maps')
        if name in ('undistort', 'native-workspace'):
            from PIL import Image
            from import_colmap_mvs_depth_dataset import load_binary_pinhole_calibration
            from dec5_hair import resize_pinhole
            calibration = load_binary_pinhole_calibration(dense/'sparse')
            raster_receipts = []
            for frame in frames:
                with Image.open(dense/'images'/frame['file_path']) as image:
                    width, height = image.size
                calibrated = calibration[frame['file_path']]
                resize_pinhole(calibrated, width, height)  # Reject unexplained crop/anisotropic resize before GPU work.
                raster_receipts.append(dict(camera=frame['physical_camera'], raster=[width,height],
                    model=[calibrated['w'],calibrated['h']]))
            atomic_json(output/'raster_preflight.json', dict(cameras=raster_receipts, passed=True))
        print('mvs stage='+name+' complete seconds='+str(round(time.time()-started)), flush=True)
    expected = {f['file_path'] for f in frames}
    for kind in ('photometric', 'geometric'):
        actual = {str(p.relative_to(dense/'stereo/depth_maps'))[:-len('.'+kind+'.bin')]
                  for p in (dense/'stereo/depth_maps').rglob('*.'+kind+'.bin')}
        if actual != expected:
            raise ValueError('MVS map inventory mismatch: '+kind)
    atomic_json(output/'progress.json', dict(stage='complete', status='complete', train_count=59))


def supervise(command, root, hours, reserve_gib, disk_reserve_gib, task_memory_gib=24):
    """Bound one *owned* worker group. Other processes are never signalled."""
    import signal
    root.mkdir(parents=True, exist_ok=False)
    before = resources()
    if task_memory_gib<=0:raise ValueError('Task memory cap must be positive')
    required = (task_memory_gib+reserve_gib)*1024
    if before['gpu_total_mib']-before['gpu_used_mib'] < required:
        raise RuntimeError('GPU headroom is below task cap + service reserve')
    if shutil.disk_usage(str(root)).free < disk_reserve_gib*(1 << 30):
        raise RuntimeError('Insufficient free disk before launch')
    started = time.time()
    atomic_json(root/'request.json', dict(command=command, hours=hours, baseline_resources=before,
                reserve_gib=reserve_gib, disk_reserve_gib=disk_reserve_gib,
                task_memory_gib=task_memory_gib,pid=os.getpid()))
    with (root/'worker.log').open('w') as log, (root/'checks.jsonl').open('a', buffering=1) as checks:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        reason = None
        try:
            while worker.poll() is None:
                state = resources()
                state.update(time=datetime.now(timezone.utc).isoformat(), worker_pid=worker.pid,
                             worker_alive=worker.poll() is None, elapsed_seconds=time.time()-started,
                             free_disk_bytes=shutil.disk_usage(str(root)).free)
                checks.write(json.dumps(state)+'\n')
                if state['elapsed_seconds'] >= hours*3600:
                    reason = 'budget'
                elif state['gpu_used_mib']-before['gpu_used_mib'] > task_memory_gib*1024:
                    # Conservative: a new unrelated allocation may also stop OUR job.
                    reason = 'task_memory_cap'
                elif state['gpu_total_mib']-state['gpu_used_mib'] < reserve_gib*1024:
                    reason = 'gpu_reserve'
                elif state['free_disk_bytes'] < disk_reserve_gib*(1 << 30):
                    reason = 'disk_reserve'
                if reason:
                    break
                time.sleep(30)
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
        text = (root/'worker.log').read_text(errors='replace')
        atomic_json(root/'result.json', dict(returncode=worker.returncode, stop_reason=reason,
                    elapsed_seconds=time.time()-started,
                    oom_evidence=('out of memory' in text.lower()), final_resources=resources()))
        return worker.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('supervise')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--reserve-gib', type=float, default=8)
    p.add_argument('--disk-reserve-gib', type=float, default=8)
    p.add_argument('--task-memory-gib',type=float,default=24,
                   help='Explicit per-job GPU delta cap; large jobs need cap + service reserve free before launch.')
    p.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('patchmatch')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--colmap', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=960)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source, args.root)
        return 0
    if args.stage == 'patchmatch':
        patchmatch(args.data, args.output, args.colmap, args.image_size)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.hours <= 24:
        parser.error('Provide a worker command and a positive budget up to 24 hours')
    return supervise(command, args.root, args.hours, args.reserve_gib, args.disk_reserve_gib,args.task_memory_gib)


if __name__ == '__main__':
    raise SystemExit(main())
