"""Dry by default; exact root release + exclusive controller lease required.

Best126316 LookCloser TRAIN-only visibility capture, never semantic fitting.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import threading

from prepare_marshawn_lc_semantic_capture import sha, write


def absolute(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authorize(plan, release_path, campaign):
    m = json.loads(plan.read_text())
    if release_path is None:
        raise ValueError('Exact root-reviewed capture release required before Torch/source loading')
    release = json.loads(release_path.read_text())
    if release.get('reviewer') != 'root' or release.get('capture_authorized') is not True or release.get('manifest_sha256') != sha(plan):
        raise ValueError('No exact root capture release')
    if release.get('code_sha256') != m['code_sha256']:
        raise ValueError('Release must bind every executable byte')
    if Path(__file__).resolve() != (plan.parent/'code'/Path(__file__).name).resolve():
        raise ValueError('Execute only the reviewed packaged runner')
    for name, digest in m['code_sha256'].items():
        if sha(plan.parent/'code'/name) != digest:
            raise ValueError('Reviewed executable changed')
    old = Path(m['campaign'])
    def relocate(path):
        p = Path(path)
        return campaign/p.relative_to(old) if p.is_relative_to(old) else p
    for path, digest in m['bound_inputs'].items():
        if sha(relocate(path)) != digest:
            raise ValueError('Sealed input changed: '+path)
    branch = relocate(m['branch'])
    for name, digest in m['scientific_files_sha256'].items():
        if sha(branch/'code'/name) != digest:
            raise ValueError('Scientific code changed: '+name)
    archive = relocate(m['source_archive'])
    for name, digest in m['source_archive_sha256'].items():
        actual = Path(m['local_checkpoint']) if name == 'checkpoint.ckpt' else archive/name
        if sha(actual) != digest:
            raise ValueError('Actual source archive/checkpoint changed: '+name)
    for row in m['records']:
        if sha(relocate(row['archive'])) != row['archive_sha256']:
            raise ValueError('Fixed TRAIN query changed')
    return m, release, relocate


def monitor_resources(policy, uuid, directory, done):
    """Continuous guard independent of camera progress and model/kernel work."""
    while not done.is_set():
        try:
            stat = {k:v.strip() for k,v in (s.split(':',1) for s in Path('/proc/self/status').read_text().splitlines())}
            rss = int(stat['VmRSS'].split()[0])*1024
            available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            free = shutil.disk_usage(directory).free
            result = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
            rows = [s.replace(' ','').split(',') for s in result.stdout.splitlines() if s.strip()]
            rows = [r for r in rows if r[0].lower() == uuid.lower()]
            row = dict(time=time.time(),pid=os.getpid(),worker_alive=True,RSS_bytes=rss,available_RAM_bytes=available,free_disk_bytes=free,compute_processes=rows)
            with (directory/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
            if (rss > policy['max_worker_RSS_gib']*2**30 or available < 2*2**30 or free < 5*2**30 or
                    any(int(r[1]) != os.getpid() or int(r[2]) > policy['max_process_mib'] for r in rows)):
                raise RuntimeError('Continuous resource/exclusivity guard exceeded')
        except BaseException as exc:
            write(directory/'resource_failure.json',dict(time=time.time(),error=repr(exc),pid=os.getpid()))
            os.kill(os.getpid(),signal.SIGTERM)
            return
        done.wait(5)


def project_import_bindings(m, package):
    """Bind actually imported project/helper modules, including lazy imports."""
    trusted={str(Path(p).resolve()):h for p,h in m['bound_inputs'].items()}
    trusted.update({str((package/'code'/name).resolve()):h for name,h in m['code_sha256'].items()})
    base=Path(m['branch'])/'code'
    trusted.update({str((base/name).resolve()):h for name,h in m['scientific_files_sha256'].items()})
    roots=[base.resolve(),Path(m['helpers']).resolve(),(package/'code').resolve()]
    result={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        is_project=name.startswith(('nerfstudio','marshawn_','train_marshawn','probe_marshawn','prepare_marshawn','_lc_','_reviewed_lc','_local_LC'))
        if not is_project and not any(path.is_relative_to(root)for root in roots):continue
        expected=trusted.get(str(path))
        if expected is None or sha(path)!=expected:
            raise ValueError('Unbound actual project/helper import: '+name+' '+str(path))
        result[name]=dict(path=str(path),sha256=expected)
    return result


def prove_source_FHD(pipe, reference, output, model, torch):
    """Three native RGB+alpha model renders; no heldout targets/masks read."""
    from PIL import Image
    import numpy as np
    proof = absolute('_lc_FHD_reference_comparison', Path(reference['helper']))
    if sha(reference['helper']) != reference['helper_sha256']:
        raise ValueError('Source comparison helper changed')
    ds = pipe.datamanager.eval_dataset
    names = [Path(p).stem.split('_')[-1]for p in ds.image_filenames]
    if set(names) != {'DC','HC','KA'} or len(names) != 3:
        raise ValueError('Source parity requires original three heldout model views')
    dest = output/'source_FHD_parity';dest.mkdir()
    results = {}
    with torch.inference_mode():
        for i,cam in enumerate(names):
            camera = ds.cameras[i:i+1].to(model.device)
            native = model.get_outputs_for_camera(camera)
            for suffix, array in [('.png',(native['rgb'].clamp(0,1).float().cpu().numpy()*255+.5).astype(np.uint8)),
                    ('_alpha.png',(native['accumulation'][...,0].clamp(0,1).float().cpu().numpy()*65535+.5).astype(np.uint16))]:
                file=cam+suffix;ref=Path(reference['renders'])/file
                if sha(ref) != reference['files_sha256'][file]:raise ValueError('Original source MODEL reference changed')
                Image.fromarray(array).save(dest/file)
                results[file] = proof.compare_code_images(array,np.asarray(Image.open(ref)))
    write(dest/'parity.json',dict(passed=True,source_checkpoint_sha256=reference['checkpoint_sha256'],
        rows=results,EVAL_target_RGB_or_masks_read=False,TRAIN_capture_started=False,
        tolerance='Existing campaign source replay: at most one saved code; changed fraction <=1e-5 RGB8 or .001 alpha16.'))
    return results


def health(torch, uuid, policy):
    mem = {k: int(v.split()[0])*1024 for k, v in (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    status = {k: v.strip() for k, v in (s.split(':', 1) for s in Path('/proc/self/status').read_text().splitlines())}
    rss = int(status['VmRSS'].split()[0])*1024
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = [s.replace(' ', '').split(',') for s in result.stdout.splitlines() if s.strip()]
    ours = [r for r in rows if int(r[1]) == os.getpid()]
    if len(ours) != 1 or ours[0][0].lower() != uuid.lower():
        raise RuntimeError('Worker compute process is not on exactly the leased UUID')
    if any(r[0].lower() == uuid.lower() and int(r[1]) != os.getpid() for r in rows):
        raise RuntimeError('Foreign process on exclusively leased UUID')
    if int(ours[0][2]) > policy['max_process_mib'] or rss > policy['max_worker_RSS_gib']*2**30 or mem['MemAvailable'] < 2*2**30:
        raise RuntimeError('Read-only capture resource cap exceeded')
    return dict(UTC_unix=time.time(), pid=os.getpid(), UUID=ours[0][0], GPU_MiB=int(ours[0][2]),
        RSS_bytes=rss, available_RAM_bytes=mem['MemAvailable'], GPU_reserved_bytes=torch.cuda.memory_reserved())


def run(a):
    if not a.execute:
        print(json.dumps(dict(execute=False, authorization=False, plan=str(a.plan), manifest_sha256=sha(a.plan), phase='native_LC_capture_only', GPU_started=False)))
        return
    campaign = a.campaign or Path(json.loads(a.plan.read_text())['campaign'])
    m, release, locate = authorize(a.plan, a.release, campaign)
    output = Path(m['output'])
    if a.output is not None and a.output.resolve() != output.resolve():
        raise ValueError('Use sealed writable local output only')
    if output.exists():
        raise FileExistsError('Never overwrite a previous capture')
    if shutil.disk_usage(output.parent).free < m['min_disk_gib']*2**30:
        raise RuntimeError('Insufficient capture disk space')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    if available < m['min_available_RAM_gib']*2**30:
        raise RuntimeError('Insufficient free host RAM')
    helpers, branch, archive = [locate(m[k]) for k in ['helpers', 'branch', 'source_archive']]
    isolation = absolute('_lc_capture_gpu_isolation', helpers/'marshawn_gpu_isolation.py')
    lease = isolation.require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    if lease['expected_physical_uuid'] != os.environ.get('EXPECTED_GPU_UUID'):
        raise ValueError('Controller expected UUID differs from lease')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Native LC capture exceeded reviewed wall time')))
    signal.alarm(m['max_seconds'])
    sys.path[:0] = [str(helpers), str(branch/'code'), str(branch/'code/LookCloser/scripts')]
    import numpy as np
    import torch
    adapter = absolute('_reviewed_lc_native_capture', a.plan.parent/'code/marshawn_lc_semantic_capture.py')
    capture_lc_native = adapter.capture_lc_native
    if torch.__version__ != m['torch_version'] or torch.version.cuda != m['cuda_version']:
        raise ValueError('Unreviewed Torch/CUDA runtime')
    identity = isolation.verify_cuda(lease['expected_physical_uuid'], torch)
    if identity['actual_physical_uuid'].lower() != m['expected_physical_UUID'].lower():
        raise ValueError('Lease must be exact reviewed olive GPU0 UUID')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(m['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    output.mkdir()
    guard_done = threading.Event()
    threading.Thread(target=monitor_resources,args=(m,lease['expected_physical_uuid'],output,guard_done),daemon=True).start()
    # Identical source eval loader/config callback to reviewed native_A, with
    # inference mode suppressing image/mask caching. No optimizer constructed.
    late = absolute('_lc_capture_tree_digest', helpers/'train_marshawn_semantic_late_distortion.py')
    import marshawn_semantic_occupancy as semantic
    import nerfstudio.utils.eval_utils as eu
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    write(output/'import_binding_before_model.json',project_import_bindings(m,a.plan.parent))
    cp = Path(m['local_checkpoint'])
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    keys = ['pipeline', 'optimizers', 'schedulers', 'scalers', 'rng_state', 'tuning_runtime', 'actual_occupancy_capture']
    serialized = {k: late.tree_digest(payload[k]) for k in keys}
    if serialized != m['expected_serialized_states_sha256']:
        raise ValueError('Selected seven source states differ from CPU-qualified checkpoint')
    if payload['step'] != m['source_step']:
        raise ValueError('Wrong exact source step')
    def configure(cfg):
        cfg.eval_checkpoint = cp
        for dotted,expected in m['expected_config_paths'].items():
            actual=cfg
            for key in dotted.split('.'):actual=getattr(actual,key)
            if str(actual) != expected:
                raise ValueError('Materialized config path differs: '+dotted)
        return cfg
    cfg, pipe, actual_checkpoint, step = eu.eval_setup(Path(m['local_config']), test_mode='inference', update_config_callback=configure)
    model = pipe.model
    pipe.eval(); pipe.requires_grad_(False); model.current_train_step = step
    if Path(actual_checkpoint).resolve() != cp.resolve() or step != m['source_step'] or late.tree_digest(pipe.state_dict()) != serialized['pipeline']:
        raise ValueError('Strict full native pipeline restoration failed')
    c = model.config
    if (c.adaptive_coarse_step_size != .0015625 or c.corrected_arm_allocator or c.max_steps_per_ray != 1024 or
            c.occupancy_dilation_radius != 0 or c.occupancy_eval_dilation_radius != 0 or
            c.transmittance_threshold != 0 or c.adaptive_fixed_fallback_samples_per_ray != 0 or
            c.background_color != 'black' or model.field.density_activation != 'trunc_exp' or
            not model.field.correct_sh_directions or not getattr(model.field, '_semantic_admission_installed', False)):
        raise ValueError('Selected native source recipe differs')
    core, active, ema, _, grid_receipt = semantic.load_snapshot(archive/'grid.json', cp)
    with np.load(archive/'grid.npz', allow_pickle=False) as z:
        admission = z['admission'].copy()
    for expected, actual in [(active, model.occupancy_grid.binaries), (ema, model.occupancy_grid.occs.reshape(ema.shape)),
                             (core, model.semantic_learned_core), (admission.reshape(model.semantic_admission.shape), model.semantic_admission)]:
        if not np.array_equal(expected, actual.detach().cpu().numpy()):
            raise ValueError('Source actual core/EMA/admission/active mismatch')
    if model.config.feature_reweighting_strength != m['source_FR'] or model.field.feature_reweighting_strength != m['source_FR']:
        raise ValueError('Loaded source FR switch state must already equal0.3')
    model.field._get_discrete_feature_weights(torch.arange(model.field.num_levels, device=model.device, dtype=torch.float32).unsqueeze(-1))
    if model.adaptive_sampler.get_sigma_fn(torch.empty(0, 3, device=model.device), torch.empty(0, 3, device=model.device)) is not None:
        raise ValueError('Unexpected native eval density/opacity pruning callback')
    before = late.tree_digest(pipe.state_dict())
    lut = late.tree_digest((model.field._feature_weight_lut, model.field._feature_weight_lut_strength))
    admission_digest = late.tree_digest((model.semantic_admission, model.semantic_lookup.state_dict()))
    normalize = model.field._normalize_positions
    _restore_rng_state(payload['rng_state'])
    rng_before = late.tree_digest(_capture_rng_state())
    frames = {f['physical_camera']: f for f in json.loads((campaign/'data/transforms.json').read_text())['frames']}
    ds = pipe.datamanager.train_dataset
    names = [Path(p).stem.split('_')[-1] for p in ds.image_filenames]
    if set(names) != set(m['train_cameras']) or len(names) != 63:
        raise ValueError('TRAIN camera membership differs')
    write(output/'initial.json', dict(identity=identity, lease=lease, health=health(torch, lease['expected_physical_uuid'], m),
        manifest_sha256=sha(a.plan), release_sha256=sha(a.release), pipeline_sha256=before,
        serialized_states=serialized, LUT_sha256=lut, RNG_sha256=rng_before, world_to_box=m['world_to_box']))
    FHD_proof = prove_source_FHD(pipe,m['original_source_FHD'],output,model,torch)
    write(output/'import_binding_after_FHD.json',project_import_bindings(m,a.plan.parent))
    if late.tree_digest(pipe.state_dict()) != before or late.tree_digest(_capture_rng_state()) != rng_before:
        raise ValueError('Source FHD parity mutated native state/RNG')
    if late.tree_digest((model.field._feature_weight_lut,model.field._feature_weight_lut_strength)) != lut:
        raise ValueError('Source FHD parity mutated primed LUT')
    records, seen, global_offset = [], [], 0
    fields = ['xyz', 'starts', 'ends', 'sigma', 'rgb', 'weight', 'spacing_starts', 'spacing_ends', 'ray_indices']
    ray_fields = ['native_rgb', 'native_alpha', 'num_samples_per_ray', 'cap_hit']
    bounds = torch.tensor(m['bounds'], device=model.device)
    for row in m['records']:
        cam, n = row['camera'], row['rays']
        if not n:
            records.append(dict(camera=cam, rays=0, UNKNOWN=True)); continue
        camera = ds.cameras[names.index(cam):names.index(cam)+1].to(model.device)
        f = frames[cam]
        if not torch.equal(camera.camera_to_worlds[0].cpu(), torch.tensor(f['transform_matrix'], dtype=torch.float32)[:3]):
            raise ValueError('Parsed LC camera is not already in source floor frame')
        for key in ['fx', 'fy', 'cx', 'cy', 'width', 'height']:
            raw_key = {'fx':'fl_x', 'fy':'fl_y', 'width':'w', 'height':'h'}.get(key, key)
            if float(getattr(camera, key).reshape(-1)[0]) != float(torch.tensor(f[raw_key], dtype=getattr(camera, key).dtype)):
                raise ValueError('Parsed native raster/intrinsics drift')
        with np.load(locate(row['archive']), allow_pickle=False) as z:
            q = {k:z[k].copy() for k in z.files}
        for unknown in m['unknown_queries']:
            if unknown['camera'] == cam:
                i = unknown['camera_ray_index']
                if global_offset+i != unknown['global_query_index'] or q['pixel_xy'][i].tolist() != unknown['pixel_xy'] or q['semantic_target'][i] != 1:
                    raise ValueError('Exact qualified UNKNOWN query mismatch')
                q['label_weight'][i] = 0; seen.append(unknown['global_query_index'])
        chunks, per_ray, proof, capacities = [], [], [], []
        for first in range(0, n, m['chunk']):
            uv = torch.as_tensor(q['pixel_xy'][first:first+m['chunk']], device=model.device, dtype=torch.float32)
            rays = camera.generate_rays(0, coords=uv[:, [1, 0]]+.5, keep_shape=False)
            # Independent pinhole OpenGL direction and pixel-center proof.
            local = torch.stack(((uv[:,0]+.5-camera.cx[0,0])/camera.fx[0,0],
                -(uv[:,1]+.5-camera.cy[0,0])/camera.fy[0,0], -torch.ones_like(uv[:,0])), -1)
            expected_direction = local@camera.camera_to_worlds[0,:3,:3].T
            expected_direction = torch.nn.functional.normalize(expected_direction, dim=-1)
            if float((rays.directions-expected_direction).abs().max()) > 5e-7 or float((rays.origins-camera.camera_to_worlds[0,:3,3]).abs().max()) > 2e-7:
                raise ValueError('LC floor ray/pixel-center chain differs')
            # Native unwrapped reference, using the same complete source path.
            reference_rays = camera.generate_rays(0, coords=uv[:, [1, 0]]+.5, keep_shape=False)
            native = model(reference_rays)
            result = capture_lc_native(model, rays)
            for key, actual in [('rgb', result['native_rgb']), ('accumulation', result['native_alpha'][:,None]), ('packed_weights', result['weight'][:,None])]:
                if native[key].shape != actual.shape or (actual.numel() and float((native[key]-actual).abs().max()) > 2e-5):
                    raise ValueError('Unwrapped source native parity failed')
            if not torch.equal(native['packed_ray_indices'], result['ray_indices']):
                raise ValueError('Unwrapped source packed ordering differs')
            inside = ((result['xyz'] >= bounds[0]) & (result['xyz'] <= bounds[1])).all(-1)
            cap = result['weight'].new_zeros(len(rays)).index_add(0, result['ray_indices'].long(), result['weight']*inside)
            capacities.append(cap.cpu().numpy())
            packed = {k:result[k].cpu().numpy().copy() for k in fields}
            packed['ray_indices'] += first
            chunks.append(packed)
            per_ray.append({k:result[k].cpu().numpy().copy() for k in ray_fields})
            proof.append(result['proof'])
        packed = {k:np.concatenate([p[k] for p in chunks]) for k in fields}
        ray = {k:np.concatenate([p[k] for p in per_ray]) for k in ray_fields}
        path = output/(cam+'.npz')
        np.savez_compressed(path, **packed, **ray, accepted_box_actor_cap=np.concatenate(capacities),
            camera=np.repeat(cam,n), pixel_xy=q['pixel_xy'], semantic_target=q['semantic_target'],
            label_weight=q['label_weight'], category=q['category'], model_digest=np.array(before), world_to_box=np.eye(4))
        records.append(dict(camera=cam, rays=n, samples=len(packed['weight']), cap_hit_rays=int(ray['cap_hit'].sum()),
            archive=str(path), archive_sha256=sha(path), proof_max={k:max(p[k] for p in proof) for k in proof[0]}))
        global_offset += n
        if late.tree_digest(pipe.state_dict()) != before or late.tree_digest(_capture_rng_state()) != rng_before:
            raise ValueError('Native pipeline/RNG changed during capture')
        write(output/'progress.json', dict(records=records, health=health(torch, lease['expected_physical_uuid'], m), updates=0))
    if sorted(seen) != sorted(u['global_query_index'] for u in m['unknown_queries']) or global_offset != 69888:
        raise ValueError('Query or qualified UNKNOWN accounting failed')
    checks = dict(pipeline=late.tree_digest(pipe.state_dict()) == before,
        FR=model.config.feature_reweighting_strength == m['source_FR'] and model.field.feature_reweighting_strength == m['source_FR'],
        RNG=late.tree_digest(_capture_rng_state()) == rng_before,
        LUT=late.tree_digest((model.field._feature_weight_lut,model.field._feature_weight_lut_strength)) == lut,
        admission=late.tree_digest((model.semantic_admission,model.semantic_lookup.state_dict())) == admission_digest,
        normalization_method=model.field._normalize_positions is normalize,
        serialized_seven_states={k:late.tree_digest(payload[k]) for k in keys} == serialized,
        source_checkpoint=sha(cp) == m['source_archive_sha256']['checkpoint.ckpt'],
        source_queries=all(sha(locate(r['archive'])) == r['archive_sha256'] for r in m['records']))
    if not all(checks.values()):
        raise ValueError('Frozen-state/source immutability checks failed: '+str(checks))
    write(output/'import_binding_complete.json',project_import_bindings(m,a.plan.parent))
    write(output/'complete.json', dict(passed=True, records=records, unchanged=checks, qualified_UNKNOWN_queries=seen,
        source_step=step, source_checkpoint_sha256=sha(cp), source_FR=m['source_FR'],
        native_optimizer_created=False, optimizer_updates=0, occupancy_updates=0, frequency_updates=0,
        source_original_FHD_parity=FHD_proof,
        training_authorized=False, physical_export_authorized=False, limitations=m['limitations'],
        health=health(torch, lease['expected_physical_uuid'], m)))
    signal.alarm(0)
    guard_done.set()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--release', type=Path)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    run(p.parse_args())
