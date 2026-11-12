"""Detached, restartable eight-GPU inference queue; never starts training."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main(args):
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((root/'settings.json').read_text())
    frames = list(range(908, 1268, 5))
    expected = {r['frame']:r['model_sha256'] for r in json.loads((root/'sparse24_storage_audit.json').read_text())['results']}
    waits = json.loads((out/'variant_launch.json').read_text()) if args.wait_variants else []

    def worker(gpu):
        for row in waits:
            if row['gpu'] != gpu:
                continue
            cmdline = Path(f"/proc/{row['pid']}/cmdline")
            while cmdline.exists():
                try:
                    if b'render_depth_cleanup.py' not in cmdline.read_bytes():
                        break
                except FileNotFoundError:
                    break
                time.sleep(5)
        rows = []
        for frame in frames[gpu::args.gpus]:
            destination = out/f'{frame:06}'
            name = 'raw' if args.stage == 'raw' else args.name
            receipt = destination/(name+'_receipt.json')
            if receipt.is_file():
                saved = json.loads(receipt.read_text())
                assert saved['checkpoint_sha256'] == expected[frame]
                assert (destination/(name+'.png')).is_file() and (destination/(name+'.npz')).is_file()
                if args.stage == 'clean':
                    cleanup = saved['cleanup']
                    assert cleanup['temporal_support'] == args.temporal
                    assert cleanup.get('support_name','temporal_support') == args.support_name
                    if args.temporal:
                        guide = destination/(args.support_name+'.npz')
                        assert hashlib.sha256(guide.read_bytes()).hexdigest() == cleanup['support_sha256'], 'Cached guide changed; use a new output name'
                    assert (cleanup['band_m'], cleanup['fade_m'], cleanup['radius_pixels'], cleanup['feather_pixels']) == (args.band,args.fade,args.radius,args.feather)
                rows.append(dict(frame=frame, status='cached'))
                continue
            cmd = [settings['python'], str(root/'tools/render_depth_cleanup.py'),
                   '--root', str(root), '--output', str(out), '--frame', str(frame),
                   '--stage', args.stage, '--name', args.name]
            if args.stage == 'clean':
                if args.temporal:
                    cmd += ['--temporal','--support-name',args.support_name]
                for key in ('band', 'fade', 'radius', 'feather'):
                    cmd += ['--'+key, str(getattr(args, key))]
            with (out/f'{name}_{frame}.log').open('ab') as log:
                result = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            rows.append(dict(frame=frame, status='done' if result.returncode == 0 else 'failed',
                             returncode=result.returncode))
            (out/f'{name}_gpu{gpu}_progress.json').write_text(json.dumps(rows, indent=2))
        return rows

    with ThreadPoolExecutor(args.gpus) as pool:
        results = [row for group in pool.map(worker, range(args.gpus)) for row in group]
    summary = dict(time=time.time(), stage=args.stage, name=args.name, results=results,
                   failed=[r for r in results if r['status']=='failed'])
    (out/f'{args.stage}_{args.name}_batch_complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['raw','clean'], required=True)
    p.add_argument('--name', default='depth_clean'); p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--wait-variants', action='store_true')
    p.add_argument('--temporal', action='store_true')
    p.add_argument('--support-name', default='temporal_support')
    p.add_argument('--band', type=float, default=.15); p.add_argument('--fade', type=float, default=.1)
    p.add_argument('--radius', type=float, default=8); p.add_argument('--feather', type=float, default=6)
    main(p.parse_args())
