"""Opt-in independent128/256 FOUR-UNKNOWN direct-LC head; dry-run by default.

Native RGB/density are not instantiated or optimized. CUDA grid_sample backward
uses atomics: restoration must be exact, subsequent independent trajectories
are measured numerically and are never claimed bitwise deterministic.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time
from prepare_marshawn_frozen_semantic_cpu import sha, write
from marshawn_LC_semantic_resolution_contract import (validate_spec, validate_qualification,
    checkpoint_metadata, validate_resume, validate_matched_baseline)


def run(path, execute):
    spec = json.loads(path.read_text())
    resolution = validate_spec(spec)
    baseline = json.loads(Path(spec['matched_four_UNKNOWN128_spec']).read_text())
    validate_matched_baseline(spec, baseline)
    if not execute:
        print(json.dumps(dict(spec_sha256=sha(path), phase=spec['phase'], updates=spec['updates'],
            native_updates=0, CUDA_started=False, optimizer_created=False,
            initial_probability=spec['initial_probability'], grid_resolution=resolution, qualified_UNKNOWN_count=4,
            fresh_head_and_Adam=not bool(spec.get('resume')),
            label_qualification='FOUR_ROOT_REVIEWED_UNKNOWN_REQUIRED', capacity_policy=spec['capacity_policy'])))
        return
    release = json.loads(Path(spec['root_release']).read_text())
    authorization_key = spec.get('authorization_key', 'technical_head_authorized')
    if authorization_key not in ['LC_resolution_technical_head_authorized', 'LC_resolution_first2064_head_authorized']:
        raise ValueError('Unrecognized bounded semantic phase')
    if release.get('reviewer') != 'root' or release.get(authorization_key) is not True or release.get('spec_sha256', {}).get(str(path)) != sha(path):
        raise ValueError('Exact root head-technical release required')
    if spec['capacity_policy'] != 'KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND':
        raise ValueError('Do not filter or silently normalize capacity violations')
    for p, digest in release['bound_files'].items():
        if sha(p) != digest:
            raise ValueError('Bound file changed: '+p)
    labels = json.loads(Path(spec['label_qualification']).read_text())
    if not labels.get('root_review_complete') or labels.get('reviewer') != 'root' or sha(spec['label_qualification']) != release['label_qualification_sha256']:
        raise ValueError('Source-label qualification must be explicitly reviewed first')
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    mem = {k: int(v.split()[0])*1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] < 8*2**30:
        raise RuntimeError('Head technical probe requires8GiB available RAM initially')
    import numpy as np
    import torch
    from marshawn_frozen_semantic_head import (FrozenActorGrid, tensor_digest, assert_exact_restoration,
        validate_cache, rendered_classes, conditional_class_objective)
    from run_marshawn_frozen_semantic_probe import nested_digest, capture_health
    if torch.__version__ != spec['torch_version'] or torch.version.cuda != spec['cuda_version']:
        raise ValueError('Use exact reviewed direct-LC Torch/CUDA runtime')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one leased visible physical device required')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    signal.signal(signal.SIGALRM, lambda signum, stack: (_ for _ in ()).throw(TimeoutError('Bounded head technical probe exceeded1800seconds')))
    signal.alarm(1800)
    m = json.loads(Path(spec['manifest']).read_text())
    if m['source_kind'] != 'DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or m['native_world_to_box'] != np.eye(4).tolist():
        raise ValueError('Direct LC semantics requires independent visibility and IDENTITY floor transform')
    capture = json.loads(Path(spec['capture_receipt']).read_text())
    if capture['manifest_sha256'] != sha(spec['manifest']) or not capture['native_model_unchanged']:
        raise ValueError('Frozen native capture provenance differs')
    if labels['capture_receipt_sha256'] != sha(spec['capture_receipt']):
        raise ValueError('Label qualification belongs to another native cache')
    validate_qualification(labels)
    overrides = {}
    for r in labels['unknown_queries']:
        if r['label_weight'] != 0 or sha(r['qualification']) != r['qualification_sha256']:
            raise ValueError('Only source-qualified UNKNOWN overrides are allowed')
        key = (r['camera'], *r['pixel_xy'])
        if key in overrides:
            raise ValueError('Duplicate qualified UNKNOWN query')
        overrides[key] = r
    caches = []
    observed_overrides = set()
    for row in capture['records']:
        if not row['rays']:
            continue
        if sha(row['archive']) != row['archive_sha256']:
            raise ValueError('Native cache changed')
        cache = validate_cache(row['archive'], m['train_cameras'], capture['model_digest'])
        if not np.array_equal(cache['world_to_box'], np.eye(4)):
            raise ValueError('Direct LC cache is already floor-coordinate; do not transform twice')
        for index, pixel in enumerate(cache['pixel_xy']):
            key = (row['camera'], *pixel.tolist())
            if key in overrides:
                if cache['semantic_target'][index] != 1:
                    raise ValueError('UNKNOWN override must identify an existing positive')
                cache['label_weight'][index] = 0
                observed_overrides.add(key)
        keep = ['xyz', 'weight', 'ray_indices', 'semantic_target', 'label_weight', 'camera', 'pixel_xy', 'native_alpha', 'accepted_box_actor_cap']
        caches.append({k: cache[k] for k in keep})
    if observed_overrides != overrides.keys():
        raise ValueError('Every qualification must match exactly one actual query')
    initial_probability = float(spec.get('initial_probability', .99))
    head = FrozenActorGrid(m['bounds'], m['native_world_to_box'], resolution=resolution, initial_probability=initial_probability).cuda()
    optimizer = torch.optim.Adam([head.logits], lr=.03)
    sampler = np.random.default_rng(42)
    def rng():
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all(), sampler=sampler.bit_generator.state)
    def restore_rng(state):
        random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'])
        torch.cuda.set_rng_state_all(state['torch_cuda']); sampler.bit_generator.state = state['sampler']
    def adam_count():
        states = optimizer.state_dict()['state']
        return int(next(iter(states.values()))['step']) if states else 0
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    initial_iteration = 0; history = []; restoration = None
    if spec.get('resume'):
        parent = torch.load(spec['resume'], map_location='cpu', weights_only=False)
        validate_resume(parent, spec, m['bounds'])
        if parent.get('source_kind') != m['source_kind'] or parent.get('LC_capture_receipt_sha256') != m['LC_capture_receipt_sha256'] or not parent.get('world_to_box_is_identity'):
            raise ValueError('Resume requires this direct-LC capture/frame; NGP heads cannot transfer')
        if parent['model_digest'] != capture['model_digest'] or parent['label_qualification_sha256'] != sha(spec['label_qualification']):
            raise ValueError('Head continuation provenance differs')
        if parent.get('initial_probability', .99) != initial_probability:
            raise ValueError('Do not resume a head initialized with another semantic prior')
        head.load_state_dict(parent['head'], strict=True); assert_exact_restoration(head, parent['head'])
        optimizer.load_state_dict(parent['optimizer']); restore_rng(parent['rng'])
        restoration = dict(head_exact=tensor_digest(head) == parent['head_digest'],
            Adam_exact=nested_digest(optimizer.state_dict()) == nested_digest(parent['optimizer']),
            RNG_and_sampler_exact=nested_digest(rng()) == nested_digest(parent['rng']))
        if not all(restoration.values()):
            raise ValueError('Semantic resume-start restoration is not exact')
        initial_iteration = parent['iteration']; history = list(parent['camera_history'])
        expected_iteration = int(spec.get('resume_initial_iteration', 32))
        if initial_iteration != expected_iteration or adam_count() != expected_iteration:
            raise ValueError('Resume must start from the exact reviewed parent iteration')
    write(output/'initial_state.json', dict(phase=spec['phase'], restoration=restoration, initial_iteration=initial_iteration,
        initial_Adam_count=adam_count(), native_optimizer_created=False, source_model_loaded=False,
        initial_probability=initial_probability, initial_seed=42, grid_resolution=resolution,
        qualified_unknown_queries=len(observed_overrides), resolution_experiment_family=spec['resolution_experiment_family'], health=capture_health(torch, lease['expected_physical_uuid'])))
    def save(iteration, name):
        state = dict(head=head.state_dict(), head_digest=tensor_digest(head), optimizer=optimizer.state_dict(),
            rng=rng(), iteration=iteration, model_digest=capture['model_digest'], camera_history=history,
            label_qualification_sha256=sha(spec['label_qualification']), spec_sha256=sha(path))
        state.update(checkpoint_metadata(spec))
        state['initial_probability'] = initial_probability
        state['source_kind'] = m['source_kind']
        state['LC_capture_receipt_sha256'] = m['LC_capture_receipt_sha256']
        state['world_to_box_is_identity'] = True
        destination = output/name; torch.save(state, destination)
        loaded = torch.load(destination, map_location='cpu', weights_only=False)
        for key in ['head', 'optimizer', 'rng']:
            if nested_digest(state[key]) != nested_digest(loaded[key]):
                raise ValueError('Exact semantic serialization roundtrip failed: '+key)
        return destination
    rows = []
    for index in range(spec['updates']):
        selected = int(sampler.integers(len(caches))); cache = caches[selected]
        xyz = torch.as_tensor(cache['xyz'], device='cuda')
        weights = torch.as_tensor(cache['weight'], device='cuda')
        ids = torch.as_tensor(cache['ray_indices'], device='cuda')
        target = torch.as_tensor(cache['semantic_target'], device='cuda')
        confidence = torch.as_tensor(cache['label_weight'], device='cuda')
        classes = rendered_classes(head(xyz), weights, ids, len(cache['camera']))
        objective = conditional_class_objective(classes, target, confidence)
        ratio = np.clip(cache['accepted_box_actor_cap']/np.maximum(cache['native_alpha'], 1e-12), 1e-30, 1)
        bound = float(np.mean(cache['label_weight']*cache['native_alpha']*cache['semantic_target']*(-np.log(ratio))))
        if not torch.isfinite(objective) or objective.item() < bound-1e-5:
            raise ValueError('Nonfinite objective or violation of fixed-geometry conditional lower bound')
        optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
        iteration = initial_iteration+index+1
        if not torch.isfinite(head.logits).all() or adam_count() != iteration:
            raise ValueError('Nonfinite head or missing actual Adam update')
        history.append(str(cache['camera'][0]))
        rows.append(dict(iteration=iteration, camera=str(cache['camera'][0]), conditional_objective=float(objective.detach()),
            fixed_geometry_lower_bound=bound, excess_above_bound=float(objective.detach())-bound))
        if iteration == 32 and spec['updates'] == 64:
            save(iteration, 'midpoint32.pt')
        if iteration % 8 == 0:
            write(output/'progress.json', dict(time=time.time(), iteration=iteration, actual_Adam_count=adam_count(),
                native_updates=0, health=capture_health(torch, lease['expected_physical_uuid'])))
    trajectory_reference_sha256 = None
    if spec.get('camera_history_reference'):
        reference = json.loads(Path(spec['camera_history_reference']).read_text())
        if history != reference['camera_history']:
            raise ValueError('Fresh prior experiment camera trajectory differs from paired original2064')
        trajectory_reference_sha256 = sha(spec['camera_history_reference'])
    checkpoint = save(initial_iteration+spec['updates'], 'head.pt')
    for r in capture['records']:
        if r['rays'] and sha(r['archive']) != r['archive_sha256']:
            raise ValueError('Frozen native cache changed during head probe')
    if sha(m['native_source']['checkpoint']) != m['native_source']['checkpoint_sha256']:
        raise ValueError('Original native checkpoint changed')
    signal.alarm(0)
    write(output/'complete.json', dict(phase=spec['phase'], actual_head_updates=spec['updates'], iteration=adam_count(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), restoration=restoration,
        native_model_not_loaded_or_optimized=True, native_checkpoint_and_caches_unchanged=True,
        source_RGB_targets_loaded=False, records=rows, camera_history=history,
        initial_probability=initial_probability, initial_seed=42, grid_resolution=resolution,
        qualified_UNKNOWN_count=len(observed_overrides),
        resolution_experiment_family=spec['resolution_experiment_family'], learning_rate=.03,
        source_manifest_sha256=sha(spec['manifest']), capture_adapter_sha256=sha(spec['capture_receipt']),
        packet_history_reference_sha256=sha(spec['packet_history_reference']),
        matched_four_UNKNOWN128_spec_sha256=sha(spec['matched_four_UNKNOWN128_spec']),
        camera_history_reference_sha256=trajectory_reference_sha256,
        source_kind=m['source_kind'], LC_capture_receipt_sha256=m['LC_capture_receipt_sha256'], world_to_box_is_identity=True,
        explicit_prior_caveat='Unobserved initialclass probability is an experimental prior, not semantic certainty or measured physicalemptiness.',
        determinism='Exact resume START and serialization; CUDA3Dgrid_sample backward uses atomics, so independently continued trajectory equality is measured numerically, never promised bitwise.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); run(a.spec, a.execute)
