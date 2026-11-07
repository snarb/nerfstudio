"""CPU-only corrected ARM render retention and fixed-ROI source comparisons."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from retain_marshawn_tau_diagnostic_gates import atomic, commit_gate, read, sha

ARMS = ('legacy1024', 'corrected1024')
EARLY = (126816, 127316, 128316, 130316)
PARENT = '340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
REGIONS_SHA = '603991a50762581ce1f76912c2f6d8a7d768e77f3679f74787662db2e777a309'


def policy_at(path):
    policy = read(path)
    release = read(Path(path).with_name('production_guard_release.json'))
    if release.get('reviewer') != 'root' or release.get('external_guard_authorized') is not True:
        raise ValueError('Root production release required')
    if release['policy_sha256'] != sha(path) or 'production' not in release['authorized_phases']:
        raise ValueError('Exact production policy binding required')
    return policy


def inventory(policy, completed):
    root = Path(policy['remote_root']); records = []; states = {}
    for arm in ARMS:
        run = root / 'runs' / ('production_' + arm)
        states[arm] = {p.name: read(p) for p in run.glob('*.json')
                       if p.name in ('progress.json', 'latest.json', 'complete.json', 'failure.json')}
        if not (run / 'latest.json').exists(): continue
        latest = read(run / 'latest.json')
        steps = set(EARLY)
        # Include every subsequent published validation gate and terminal selected renders.
        steps.update(int(p.name.split('-')[1]) for p in (run / 'renders').glob('step-*'))
        spec_file = root / 'specs' / ('production_' + arm) / 'spec.json'; spec = read(spec_file)
        if spec['mode'] != 'production' or spec['parent_checkpoint_sha256'] != PARENT or Path(spec['output']) != run:
            raise ValueError('Unexpected production source')
        for step in sorted(steps):
            if step > latest['step'] or f'{arm}/{step}' in completed: continue
            cp = run / 'checkpoints' / f'step-{step:09d}.ckpt'
            grid = cp.with_name(cp.stem + '_grid.json'); archive = grid.with_suffix('.npz')
            receipt = read(grid); cp_hash = sha(cp)
            if receipt['checkpoint_step'] != step or receipt['checkpoint_sha256'] != cp_hash:
                raise ValueError('Checkpoint step/hash mismatch')
            if Path(receipt['checkpoint']) != cp or receipt['grid_archive_sha256'] != sha(archive):
                raise ValueError('Grid binding mismatch')
            if latest['step'] == step and (latest['sha256'] != cp_hash or latest['checkpoint'] != str(cp.relative_to(run))):
                raise ValueError('Atomic latest mismatch')
            render = run / 'renders' / f'step-{step:09d}'
            if read(render / 'metrics.json')['step'] != step: raise ValueError('Render step mismatch')
            files = [render / n for n in ('DC.png', 'HC.png', 'KA.png', 'DC_alpha.png', 'HC_alpha.png', 'KA_alpha.png', 'metrics.json')]
            files += [grid, archive]
            records.append(dict(arm=arm, step=step, source_checkpoint_sha256=PARENT,
                                spec_sha256=sha(spec_file), worker_sha256=spec['worker_sha256'],
                                checkpoint=dict(path=str(cp), sha256=cp_hash, bytes=cp.stat().st_size,
                                                retained_remote=True, copied_locally=False),
                                observed_atomic_latest=latest, remote_run=str(run), remote_time=time.time(),
                                files_sha256={str(p.relative_to(run)): sha(p) for p in files}, no_pruning=True))
    return dict(records=records, states=states, GPU_used=False)


def review(root, step):
    from PIL import Image, ImageDraw, ImageFont
    base = root / 'external_guard_corrected_ARM_v1/visual_gates_v1'
    output = base / 'reviews' / f'step-{step:09d}'
    if (output / 'provenance.json').exists(): return
    output.mkdir(parents=True, exist_ok=True)
    ready = root.parent / 'data/ready.json'; rows = {r['camera']: r for r in read(ready)['records']}
    regions_file = root.parent / 'native_semantic_grid_20000/corrected_v2/review/region_crops.json'
    if sha(regions_file) != REGIONS_SHA: raise ValueError('Fixed original ROI changed')
    regions = read(regions_file); metrics = {}; bindings = {}; sources = {}; details = []; actors = []
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    for arm in ARMS:
        gate = base / 'retained_gates' / arm / f'step-{step:09d}'
        receipt = read(gate / 'retention.json')
        for rel, digest in receipt['files_sha256'].items():
            if sha(gate / rel) != digest: raise ValueError('Retained file changed')
        bindings[arm] = sha(gate / 'retention.json')
        metrics[arm] = read(gate / 'renders' / f'step-{step:09d}' / 'metrics.json')['metrics']
    for camera in ('DC', 'HC', 'KA'):
        paths = [root.parent / 'data' / rows[camera]['image'], root / 'anchor/renders' / (camera + '.png')]
        paths += [base / 'retained_gates' / arm / f'step-{step:09d}' / 'renders' / f'step-{step:09d}' / (camera + '.png') for arm in ARMS]
        images = [Image.open(p).convert('RGB') for p in paths]
        if any(im.size != (1920, 1012) for im in images): raise ValueError('Unexpected FHD raster')
        sources.update({str(p): sha(p) for p in paths})
        crops = list(rows[camera]['crops'].values())
        actor_crop = [max(0, min(v[0] for v in crops)-64), max(0, min(v[1] for v in crops)-64),
                      min(1920, max(v[2] for v in crops)+64), min(1012, max(v[3] for v in crops)+64)]
        for name, crop in [('actors_margin64', actor_crop)] + list(regions[camera].items()):
            w, h = crop[2]-crop[0], crop[3]-crop[1]
            panel = Image.new('RGB', (w*4, h+38), (20, 20, 20)); draw = ImageDraw.Draw(panel)
            labels = ['GT', 'source126316', 'legacy1024 control', 'corrected1024']
            for i, (im, label) in enumerate(zip(images, labels)):
                panel.paste(im.crop(crop), (i*w, 38))
                draw.text((i*w+3, 2), camera+' '+name, font=font, fill='white')
                draw.text((i*w+3, 19), label, font=font, fill='white')
            panel.save(output / (camera+'_'+name+'.png'))
            panel.thumbnail((2200, 950 if name == 'actors_margin64' else 500))
            (actors if name == 'actors_margin64' else details).append(panel)
    for name, panels in [('all_details', details), ('all_actors', actors)]:
        contact = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), (20,20,20))
        y = 0
        for panel in panels: contact.paste(panel, (0,y)); y += panel.height
        contact.save(output / (name+'.jpg'), quality=95)
    atomic(output / 'provenance.json', dict(step=step, source_checkpoint_sha256=PARENT, source_files_sha256=sources,
        retained_manifests_sha256=bindings, ready_sha256=sha(ready), fixed_detail_regions_sha256=sha(regions_file),
        helper_sha256=sha(__file__), primary_metrics={a:{k:metrics[a][k] for k in ('psnr','ssim','lpips')} for a in ARMS},
        crop_policy='Fixed original hair/hands_contact/feet ROI plus actor union with64px context; no hardmask clipping',
        GPU_used=False, metric_or_labels_changed=False, promotion=False))


def once(policy_path):
    policy = policy_at(policy_path); root = Path(policy['local_root'])
    base = root / 'external_guard_corrected_ARM_v1/visual_gates_v1'; out = base / 'retained_gates'; out.mkdir(parents=True, exist_ok=True)
    completed = {f"{read(p)['arm']}/{read(p)['step']}" for p in out.glob('*/step-*/retention.json') if '.partial-' not in str(p)}
    remote_helper = Path(policy['remote_root']) / 'external_guard_corrected_ARM_v1/visual_gates_v1' / Path(__file__).name
    remote_policy = remote_helper.parent.parent / 'policy.json'
    argv = ['python3', str(remote_helper), 'inventory', '--policy', str(remote_policy), '--completed', json.dumps(sorted(completed))]
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',policy['address'],shlex.join(argv)],capture_output=True,text=True,check=True,timeout=180)
    snapshot = json.loads(result.stdout)
    def copy_file(source, target):
        subprocess.run(['rsync','-az','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',policy['address']+':'+source,str(target)],check=True,timeout=120)
    for record in snapshot['records']: commit_gate(record, out, copy_file)
    sets = [{int(p.name.split('-')[1]) for p in (out / a).glob('step-*') if (p / 'retention.json').exists()} for a in ARMS]
    common = sorted(set.intersection(*sets))
    for step in common:
        review(root, step)
        summarize(root, step)
    status = dict(time=time.time(), common_review_steps=common, states=snapshot['states'], GPU_used=False, checkpoint_pruning=False)
    atomic(base / 'status.json', status)
    return all('complete.json' in snapshot['states'][a] for a in ARMS)


def summarize(root, step):
    """Use the existing fixed-V1 Euclidean exterior-opacity definition unchanged."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from inspect_marshawn_exterior_opacity import read_alpha
    base = root / 'external_guard_corrected_ARM_v1/visual_gates_v1'
    output = base / 'reviews' / f'step-{step:09d}' / 'summary.json'
    if output.exists(): return
    ready_path = root.parent / 'data/ready.json'; ready = {r['camera']:r for r in read(ready_path)['records']}
    paths = {'parent126316':root / 'anchor/renders'}
    paths.update({a:base / 'retained_gates' / a / f'step-{step:09d}' / 'renders' / f'step-{step:09d}' for a in ARMS})
    metrics = {a:{k:read(p/'metrics.json')['metrics'][k] for k in ('psnr','ssim','lpips')} for a,p in paths.items()}
    interior_opacity={a:read(p/'metrics.json')['opacity_diagnostics']for a,p in paths.items()}
    footprint = {a:{} for a in paths}; sources = {str(ready_path):sha(ready_path)}
    for camera in ('DC','HC','KA'):
        mask_path = root.parent / 'data/semantics' / (camera+'.npz')
        if sha(mask_path) != ready[camera]['semantic_sha256']: raise ValueError('Original V1 mask changed')
        with np.load(mask_path,allow_pickle=False) as z:
            if z['black'].dtype != np.bool_ or z['grey'].dtype != np.bool_: raise ValueError('Expected fixed V1 boolean masks')
            union=z['black']|z['grey']
        exterior=distance_transform_edt(~union)>16
        sources[str(mask_path)]=sha(mask_path)
        for arm,path in paths.items():
            alpha_path=path/(camera+'_alpha.png'); alpha=read_alpha(alpha_path)
            if alpha.shape != union.shape: raise ValueError('Wrong opacity raster')
            footprint[arm][camera]=int(((alpha>.1)&exterior).sum()); sources[str(alpha_path)]=sha(alpha_path)
    value=dict(step=step,metrics=metrics,existing_eroded_actor_opacity_diagnostics=interior_opacity,outside16_alpha_above01_pixels=footprint,input_sha256=sources,
               definition='Exact original V1 black|grey union; scipy Euclidean distance >16HD px; scalar PNG16 alpha >.1',
               limitation='Mask-exterior opacity is diagnostic, not nonactor/free-space truth. No new labels or metric changes.',GPU_used=False)
    atomic(output,value)
    lines=['| arm | PSNR | SSIM | LPIPS | exterior DC / HC / KA |','|---|---:|---:|---:|---|']
    for arm,m in metrics.items():
        counts=' / '.join(str(footprint[arm][c]) for c in ('DC','HC','KA'))
        lines.append(f"| {arm} | {m['psnr']:.6f} | {m['ssim']:.6f} | {m['lpips']:.8f} | {counts} |")
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def watch(path):
    policy = policy_at(path); base = Path(policy['local_root']) / 'external_guard_corrected_ARM_v1/visual_gates_v1'
    lock = (base / 'watch.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(base / 'process.json', dict(pid=os.getpid(), started=time.time(), CPU_only=True))
    errors = 0
    while True:
        try:
            done = once(path); errors = 0
            if done: atomic(base / 'complete.json', dict(time=time.time(), passed=True)); return
        except Exception as exc:
            errors += 1; atomic(base / 'last_error.json', dict(time=time.time(), error=repr(exc), consecutive=errors))
            if errors >= 3: raise
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['inventory','once','watch','launch'])
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--completed', default='[]'); a = p.parse_args()
    if a.command == 'inventory': print(json.dumps(inventory(policy_at(a.policy), set(json.loads(a.completed)))))
    elif a.command == 'once': once(a.policy)
    elif a.command == 'watch': watch(a.policy)
    else:
        policy = policy_at(a.policy); base = Path(policy['local_root']) / 'external_guard_corrected_ARM_v1/visual_gates_v1'
        with (base / 'watch.log').open('ab') as log:
            proc = subprocess.Popen([policy['local_python'],str(Path(__file__).resolve()),'watch','--policy',str(a.policy)],
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(CPU_retention_pid=proc.pid)))
