"""Dry-run by default; reviewed native capture or independent cached head fit.

No native pipeline optimizer is created. The only optional Adam owns the new
semantic-grid logits. A root-reviewed receipt is required for either execution.
GPU portability and native parity remain technical gates, not CPU-certified facts.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from prepare_marshawn_frozen_semantic_cpu import sha, write


def authorize(a, manifest):
    if a.release is None:
        raise ValueError('Root review receipt required before execution')
    release = json.loads(a.release.read_text())
    if release.get('reviewer') != 'root' or release.get(a.phase+'_authorized') is not True or release.get('manifest_sha256') != sha(a.manifest):
        raise ValueError('Phase/manifest is not root reviewed')
    for path, digest in release['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Reviewed code changed: '+path)
    for path, digest in release['native_source_code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen native source changed: '+path)
    for path, digest in manifest['bound_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Bound source changed: '+path)
    return release


def nested_digest(value):
    """State/RNG digest without serialization formats or CUDA allocations."""
    import hashlib
    import numpy as np
    import torch
    h = hashlib.sha256()
    def feed(v):
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); feed(v[k])
        elif isinstance(v, (tuple, list)):
            for row in v: feed(row)
        else:
            h.update(repr(v).encode())
    feed(value)
    return h.hexdigest()


def capture_health(torch, expected_uuid):
    """Runtime-only guard; never called by dry-run or CPU preparation."""
    import csv
    import subprocess
    mem = {k: int(v.split()[0])*1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    status = {k: v.strip() for k, v in (line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())}
    rss = int(status['VmRSS'].split()[0])*1024
    if mem['MemAvailable'] < 2*2**30 or rss > 6*2**30 or torch.cuda.memory_reserved() > 12*2**30:
        raise RuntimeError('Capture RAM/RSS/GPU memory guard exceeded')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    ours = [r for r in rows if r and int(r[1]) == os.getpid()]
    if len(ours) != 1 or ours[0][0].lower() != expected_uuid.lower():
        raise RuntimeError('Actual compute process UUID differs from exclusive controller lease')
    if any(r[0].lower() == expected_uuid.lower() and int(r[1]) != os.getpid() for r in rows if r):
        raise RuntimeError('Foreign compute process appeared on the leased device')
    return dict(time=time.time(), available_RAM_bytes=mem['MemAvailable'], worker_RSS_bytes=rss,
        GPU_allocated_bytes=torch.cuda.memory_allocated(), GPU_reserved_bytes=torch.cuda.memory_reserved(),
        actual_compute_UUID=ours[0][0], physical_device_exclusive=True, pid=os.getpid())


def capture(a, m, release):
    import shutil
    import signal
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    if a.device != 'cuda:0':
        raise ValueError('Exclusive physical UUID must map to logical cuda:0')
    mem = {k: int(v.split()[0])*1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] < 8*2**30 or shutil.disk_usage(a.output.parent).free < 20*2**30:
        raise RuntimeError('Require8GiB freeRAM and20GiB free disk before capture')
    signal.signal(signal.SIGALRM, lambda signum, stack: (_ for _ in ()).throw(TimeoutError('Read-only capture exceeded1800seconds')))
    signal.alarm(1800)
    import numpy as np
    for path in reversed(m['native_python_path']):
        sys.path.insert(0, path)
    import torch
    import yaml
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.cameras.cameras import Cameras
    from marshawn_frozen_semantic_head import (assert_exact_restoration, tensor_digest,
        capture_native_packed, FrozenActorGrid, ray_capacity)
    torch.set_num_threads(2)
    if torch.__version__ != release['torch_version'] or torch.version.cuda != release['cuda_version']:
        raise ValueError('Historical native runtime differs')
    saved = torch.load(m['native_source']['checkpoint'], map_location='cpu', weights_only=False)
    if int(saved['step']) != m['native_source']['step']:
        raise ValueError('Wrong selected source step')
    prefix = '_model.'
    state = {k[len(prefix):]: v for k, v in saved['pipeline'].items() if k.startswith(prefix)}
    if not state or 'scene_aabb' not in state:
        raise ValueError('Expected native _model checkpoint namespace')
    config = yaml.load(Path(m['native_config']).read_text(), Loader=yaml.Loader)
    model = config.pipeline.model.setup(scene_box=SceneBox(state['scene_aabb'].reshape(2, 3)),
        num_train_data=63, metadata={}, device=a.device, grad_scaler=None).to(a.device)
    model.load_state_dict(state, strict=True)
    assert_exact_restoration(model, state)
    model.eval(); model.requires_grad_(False)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one leased visible GPU required')
    initial_health = capture_health(torch, lease['expected_physical_uuid'])
    _restore_rng_state(saved['rng_state'])
    rng_before = nested_digest(_capture_rng_state())
    before = tensor_digest(model)
    bounds = torch.tensor(m['bounds'], device=a.device)
    frame = torch.tensor(m['native_world_to_box'], device=a.device)
    native = json.loads((Path(m['native_data'])/'transforms.json').read_text())
    frames = {f['physical_camera']: f for f in native['frames']}
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output/'initial_health.json', dict(**initial_health, lease=lease, native_optimizer_created=False))
    records = []
    # Heldout model-render parity only. These samples are never fit-cache rows,
    # never labels, and contain no EVAL source RGB or semantic masks.
    reference_path = Path(release['native_baseline_samples'])
    if sha(reference_path) != release['native_baseline_samples_sha256']:
        raise ValueError('Native baseline reference changed')
    with np.load(reference_path, allow_pickle=False) as z:
        reference = {k: z[k].copy() for k in z.files}
    import nerfacc
    baseline_rows = []
    for name in ['DC', 'HC', 'KA']:
        f = frames[name]
        camera = Cameras(camera_to_worlds=torch.tensor(f['transform_matrix'], dtype=torch.float32)[None, :3],
            fx=f['fl_x'], fy=f['fl_y'], cx=f['cx'], cy=f['cy'], width=f['w'], height=f['h']).to(a.device)
        pixels = reference[name+'_pixel_xy']
        rays = camera.generate_rays(0, coords=torch.tensor(pixels[:, [1, 0]], device=a.device, dtype=torch.float32)+.5, keep_shape=False)
        row = capture_native_packed(model, rays)
        box_xyz = row['xyz'] @ frame[:3, :3].T + frame[:3, 3]
        inside = ((box_xyz >= bounds[0]) & (box_xyz <= bounds[1])).all(-1)
        inside_weights = nerfacc.render_weight_from_density(t_starts=row['starts'], t_ends=row['ends'],
            sigmas=row['sigma']*inside, packed_info=nerfacc.pack_info(row['ray_indices'], len(rays)))[0]
        inside_rgb = nerfacc.accumulate_along_rays(inside_weights, values=row['rgb'], ray_indices=row['ray_indices'], n_rays=len(rays))
        full_error = float(np.abs(row['native_rgb'].cpu().numpy()-reference[name+'_full_rgb']).max())
        inside_error = float(np.abs(inside_rgb.cpu().numpy()-reference[name+'_inside_rgb']).max())
        if max(full_error, inside_error) > 2e-5:
            raise ValueError('Saved190k native model RGB parity failed: '+name+' '+str((full_error, inside_error)))
        baseline_rows.append(dict(camera=name, full_RGB_max_error=full_error, inside_RGB_max_error=inside_error, pixels=pixels.tolist()))
    write(a.output/'baseline_parity.json', dict(rows=baseline_rows, labels_or_eval_RGB_read=False, reference_source='Saved190k MODEL float renders only; never enters fit cache'))
    for r in m['records']:
        if sha(r['archive']) != r['archive_sha256']:
            raise ValueError('TRAIN queries changed')
        q = dict(np.load(r['archive']))
        n = len(q['pixel_xy'])
        if not n:
            records.append(dict(camera=r['camera'], rays=0, note='Entire source UNKNOWN; no labels invented'))
            continue
        f = frames[r['camera']]
        camera = Cameras(camera_to_worlds=torch.tensor(f['transform_matrix'], dtype=torch.float32)[None, :3],
            fx=f['fl_x'], fy=f['fl_y'], cx=f['cx'], cy=f['cy'], width=f['w'], height=f['h']).to(a.device)
        coords = torch.tensor(q['pixel_xy'][:, [1, 0]], device=a.device, dtype=torch.float32)+.5
        rays = camera.generate_rays(0, coords=coords, keep_shape=False)
        # Runtime proof that native-world rays match the sealed floor camera.
        pose = frame @ torch.tensor(f['transform_matrix'], device=a.device, dtype=torch.float32)
        transformed_origin = rays.origins @ frame[:3, :3].T + frame[:3, 3]
        if not torch.allclose(transformed_origin, pose[:3, 3].expand_as(transformed_origin), atol=2e-7):
            raise ValueError('Native-to-floor ray origin transform mismatch')
        floor_camera = Cameras(camera_to_worlds=pose[None, :3], fx=f['fl_x'], fy=f['fl_y'], cx=f['cx'], cy=f['cy'], width=f['w'], height=f['h']).to(a.device)
        floor_rays = floor_camera.generate_rays(0, coords=coords, keep_shape=False)
        direction_error = float((floor_rays.directions-rays.directions @ frame[:3, :3].T).abs().max())
        if direction_error > 5e-7:
            raise ValueError('Native-to-floor ray directions mismatch')
        chunks = []; parity = []; capacities = []; alphas = []
        for first in range(0, n, a.chunk):
            row = capture_native_packed(model, rays[first:first+a.chunk])
            xyz_box = row['xyz'] @ frame[:3, :3].T + frame[:3, 3]
            inside = ((xyz_box >= bounds[0]) & (xyz_box <= bounds[1])).all(-1)
            alpha, cap = ray_capacity(row['weight'], inside, row['ray_indices'], len(rays[first:first+a.chunk]))
            alphas.append(alpha.cpu().numpy()); capacities.append(cap.cpu().numpy()); parity.append(row['parity_error'])
            row['ray_indices'] += first
            chunks.append({k: row[k].cpu().numpy() for k in ['xyz', 'weight', 'ray_indices', 'sigma', 'starts', 'ends', 'rgb']})
        packed = {k: np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
        if not np.isfinite(packed['sigma']).all() or (packed['sigma'] < 0).any():
            raise ValueError('Native density is not finite/nonnegative')
        native_tau = packed['sigma']*(packed['ends']-packed['starts'])
        archive = a.output/(r['camera']+'.npz')
        np.savez_compressed(archive, **packed,
            camera=np.repeat(r['camera'], n), model_digest=np.array(before), pixel_xy=q['pixel_xy'],
            semantic_target=q['semantic_target'], label_weight=q['label_weight'], category=q['category'],
            native_alpha=np.concatenate(alphas), accepted_box_actor_cap=np.concatenate(capacities))
        positive = q['semantic_target'] > .5
        ratio = np.concatenate(capacities)/np.maximum(np.concatenate(alphas), 1e-12)
        records.append(dict(camera=r['camera'], rays=n, archive=str(archive), archive_sha256=sha(archive),
            samples=len(packed['sigma']), max_native_sigma=float(packed['sigma'].max(initial=0)),
            max_native_interval_optical_depth=float(native_tau.max(initial=0)),
            sigma_quantiles=np.quantile(packed['sigma'], [0, .5, .9, .99, 1]).tolist() if len(native_tau) else [],
            density_intervals_RGB_saved_for_physical_recomposition=True,
            native_to_floor_direction_max_error=direction_error,
            native_parity_max=max(parity), positive_box_capacity_ratio_quantiles=np.quantile(ratio[positive], [0, .01, .1, .5, 1]).tolist() if positive.any() else [],
            positive_capacity_below095=int((ratio[positive] < .95).sum()),
            positive_native_alpha_quantiles=np.quantile(np.concatenate(alphas)[positive], [0, .01, .1, .5, 1]).tolist() if positive.any() else []))
        health = capture_health(torch, lease['expected_physical_uuid'])
        write(a.output/'progress.json', dict(**health, completed_cameras=len(records), total_cameras=63, completed_rays=sum(r['rays'] for r in records), native_updates=0))
    assert_exact_restoration(model, state)
    if before != tensor_digest(model) or sha(m['native_source']['checkpoint']) != m['native_source']['checkpoint_sha256']:
        raise ValueError('Frozen source changed')
    if rng_before != nested_digest(_capture_rng_state()):
        raise ValueError('Native ray capture unexpectedly changed RNG')
    if any(sha(r['archive']) != r['archive_sha256'] for r in m['records']):
        raise ValueError('Original TRAIN query cache changed')
    signal.alarm(0)
    write(a.output/'capture.json', dict(model_digest=before, native_model_unchanged=True,
        checkpoint_unchanged=True, records=records, native_step=int(saved['step']),
        native_to_floor_applied_once=True, native_RNG_unchanged=True, original_checkpoint_RNG_restored=True,
        source_RGB_loaded=False, optimizer_created=False, manifest_sha256=sha(a.manifest)))


def fit(a, m, release):
    import numpy as np
    import torch
    from marshawn_frozen_semantic_head import (FrozenActorGrid, validate_cache,
        rendered_classes, conditional_class_objective)
    source = json.loads((a.cache/'capture.json').read_text())
    if source['manifest_sha256'] != sha(a.manifest) or not source['native_model_unchanged']:
        raise ValueError('Cache provenance mismatch')
    if any(r.get('positive_capacity_below095', 0) for r in source['records']):
        raise ValueError('Positive box capacity gate needs root inspection before fitting')
    arrays = []
    for r in source['records']:
        if r['rays']:
            if sha(r['archive']) != r['archive_sha256']:
                raise ValueError('Cached native sample archive changed')
            arrays.append(validate_cache(r['archive'], m['train_cameras'], source['model_digest']))
    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], resolution=128).to(a.device)
    optimizer = torch.optim.Adam([head.logits], lr=.03)
    generator = np.random.default_rng(42)
    start = 0
    if a.resume:
        saved = torch.load(a.resume, map_location=a.device, weights_only=False)
        if saved['native_model_digest'] != source['model_digest'] or saved['manifest_sha256'] != sha(a.manifest):
            raise ValueError('Semantic resume source differs')
        head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        generator.bit_generator.state = saved['sampler_rng']
        start = int(saved['iterations'])
    a.output.mkdir(parents=True, exist_ok=False)
    for iteration in range(start, start+a.iterations):
        row = arrays[int(generator.integers(len(arrays)))]
        # A full fixed camera query set is a bounded batch, retaining every
        # packed termination contribution rather than truncating depth tails.
        xyz = torch.as_tensor(row['xyz'], device=a.device)
        weights = torch.as_tensor(row['weight'], device=a.device)
        ids = torch.as_tensor(row['ray_indices'], device=a.device)
        prediction = rendered_classes(head(xyz), weights, ids, len(row['camera']))
        objective = conditional_class_objective(prediction, torch.as_tensor(row['semantic_target'], device=a.device),
            torch.as_tensor(row['label_weight'], device=a.device))
        optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
        if not torch.isfinite(head.logits).all():
            raise RuntimeError('Nonfinite semantic head')
    torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), iterations=start+a.iterations,
        sampler_rng=generator.bit_generator.state, native_model_digest=source['model_digest'], manifest_sha256=sha(a.manifest)), a.output/'semantic_head.pt')
    write(a.output/'complete.json', dict(iterations=start+a.iterations, native_model_loaded=False,
        native_optimizer_created=False, semantic_only=True, checkpoint_sha256=sha(a.output/'semantic_head.pt')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['capture', 'fit'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--release', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--iterations', type=int, default=2000)
    a = p.parse_args(); m = json.loads(a.manifest.read_text())
    if not a.execute:
        print(json.dumps(dict(phase=a.phase, manifest_sha256=sha(a.manifest),
            model=m['native_source'], train_cameras=len(m['train_cameras']),
            output=str(a.output), CUDA_started=False, optimizer_created=False)))
    else:
        release = authorize(a, m)
        (capture if a.phase == 'capture' else fit)(a, m, release)
