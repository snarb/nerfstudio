#!/usr/bin/env python3
"""Fit genuine train-only LookCloser maps, with quiet, supervised GPU workers."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def worker(data, index, workers):
    import torch
    from nerfstudio.scripts.lookcloser_preprocess import LookCloserPreprocessConfig, load_image_as_tensor
    torch.set_num_threads(2)
    meta = json.loads((data / 'transforms.json').read_text())
    output = data / 'lookcloser_frequencies'
    output.mkdir(exist_ok=True)
    cfg = LookCloserPreprocessConfig(max_res=8192, train_steps_per_level=1000)
    for name in meta['train_filenames'][index::workers]:
        path = data / name
        target = output / (path.stem + '.pt')
        if target.exists() and target.with_suffix('.json').exists():
            continue
        torch.manual_seed(42 + int(path.stem.split('_')[1]))
        started = time.time()
        image = load_image_as_tensor(path, torch.device('cuda'))
        cfg._process_single_image(img_path=path, image_tensor=image, save_freq_path=target,
                                  debug_out_dir=None, scene_size=4.)
        sidecar = json.loads(target.with_suffix('.json').read_text())
        sidecar['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar['fit_steps_per_level'] = 1000
        target.with_suffix('.json').write_text(json.dumps(sidecar, indent=2)+'\n')
        print(json.dumps(dict(image=path.name, seconds=time.time()-started)), flush=True)
        del image
        torch.cuda.empty_cache()


def supervise(data, workers):
    logs = data.parent / 'logs'
    logs.mkdir(exist_ok=True)
    children = []
    handles = []
    for index in range(workers):
        handle = (logs / f'frequency_worker_{index}.log').open('a')
        handles.append(handle)
        children.append(subprocess.Popen([sys.executable, __file__, '--data', str(data), '--worker',
                                          str(index), '--workers', str(workers)], stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True))
    def stop(*_):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(130)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start = time.time()
    try:
        while True:
            gpu = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total',
                                  '--format=csv,noheader'], capture_output=True, text=True)
            states = [c.poll() for c in children]
            row = dict(time=time.time(), elapsed=time.time()-start, controller=os.getpid(),
                       workers=[dict(pid=c.pid, exit=s) for c,s in zip(children,states)],
                       maps=len(list((data/'lookcloser_frequencies').glob('*.pt'))),
                       gpu=gpu.stdout.strip(), gpu_error=gpu.stderr.strip())
            row['oom']=any('out of memory' in (logs/f'frequency_worker_{i}.log').read_text(errors='replace')[-8192:].lower()
                           for i in range(workers))
            with (logs/'supervision.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(f"frequency_maps={row['maps']}/162 elapsed={row['elapsed']:.0f}s gpu={row['gpu']}",flush=True)
            if any(s is not None and s != 0 for s in states):
                stop()
            if all(s is not None for s in states):
                break
            time.sleep(30)
        meta = json.loads((data/'transforms.json').read_text())
        expected = {Path(n).stem for n in meta['train_filenames']}
        actual = {p.stem for p in (data/'lookcloser_frequencies').glob('*.pt')}
        if actual != expected:
            raise RuntimeError('Train-only frequency inventory mismatch')
        (data/'frequency_ready.json').write_text(json.dumps(dict(maps=162,elapsed=time.time()-start))+'\n')
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--worker',type=int)
    args=p.parse_args()
    if args.workers <= 0 or (args.worker is not None and not 0 <= args.worker < args.workers):
        p.error('Invalid worker configuration')
    worker(args.data,args.worker,args.workers) if args.worker is not None else supervise(args.data,args.workers)
