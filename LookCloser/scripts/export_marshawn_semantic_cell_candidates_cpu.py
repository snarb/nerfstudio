"""Opt-in CPU candidate export from a COMPLETED, bound direct-LC head.

No native model is loaded, no source grid is modified, and no density selector
is installed. Exported arrays are hypotheses for separately qualified replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from marshawn_semantic_node_cell_bridge import LogitNodeCellBridge, fresh_dilation

SOURCE_CHECKPOINT = '340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
SOURCE_PIPELINE = 'c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636'
CAPTURE_SHA = 'fb8b875d8c7f3676c3dfbd187b4c76c3ee40ec8a3b62a940fdf33bb97d99073d'
QUERY_SHA = dict(clothing103='e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c',
                 shoe6='b8dc52856872bd3f9f26bc1eee3e94110e992bf69ebc60009b68718589f57dde')
SOURCE_KIND = 'DIRECT_FROZEN_LC_NATIVE_VISIBILITY'
SELECTORS = ('hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05')
HEAD_BOUNDS = np.array([[-.84, -.21, -.07], [.15, .42, .63]], dtype=np.float32)
COORDINATE_GUARD = 2e-7
LOGIT_GUARD = 1e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def tensor_head_digest(state):
    """Same sorted tensor-byte digest as the qualified FrozenActorGrid helper."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_head(payload, completion, spec, contract, hashes):
    """Pure metadata/tensor validation; never instantiates a head or optimizer."""
    import torch
    for value in (payload, completion, contract):
        if value.get('source_kind') != SOURCE_KIND or value.get('LC_capture_receipt_sha256') != CAPTURE_SHA:
            raise ValueError('Wrong direct-LC source/capture; no NGP or earlier capture transfer')
    if contract.get('source_pipeline_digest') != SOURCE_PIPELINE or contract.get('native_source', {}).get('checkpoint_sha256') != SOURCE_CHECKPOINT:
        raise ValueError('Wrong exact native source')
    if payload.get('model_digest') != SOURCE_PIPELINE:
        raise ValueError('Wrong full pipeline digest')
    if not payload.get('world_to_box_is_identity') or not completion.get('world_to_box_is_identity'):
        raise ValueError('Identity floor-coordinate frame required')
    if not np.array_equal(contract['native_world_to_box'], np.eye(4)) or not np.array_equal(spec['world_to_box'], np.eye(4)):
        raise ValueError('Do not infer or apply an NGP/world transform')
    if payload.get('iteration') != 2064 or completion.get('iteration') != 2064 or completion.get('actual_head_updates') != 2064:
        raise ValueError('Only completed fresh2064 heads are candidates, not technical weights')
    if spec.get('updates') != 2064 or spec.get('resume') or spec.get('grid_resolution') != 128:
        raise ValueError('Fresh128-node2064 specification required')
    if completion.get('phase') != spec.get('phase') or spec.get('authorization_key') != 'LC_first2064_head_authorized':
        raise ValueError('Wrong completed production-head phase')
    if (spec.get('seed'), spec.get('learning_rate'), spec.get('native_model_updates')) != (42, .03, 0):
        raise ValueError('Unqualified head recipe')
    prior = payload.get('initial_probability')
    if prior not in (.01, .99) or completion.get('initial_probability') != prior or spec.get('initial_probability') != prior:
        raise ValueError('Exact declared head prior required')
    if completion.get('checkpoint_sha256') != hashes['head'] or payload.get('spec_sha256') != hashes['spec']:
        raise ValueError('Completion/head/spec binding differs')
    if payload.get('label_qualification_sha256') != hashes['labels']:
        raise ValueError('Head belongs to different source labels')
    if not completion.get('native_checkpoint_and_caches_unchanged') or not completion.get('native_model_not_loaded_or_optimized'):
        raise ValueError('Missing frozen-native completion proof')
    if completion.get('source_RGB_targets_loaded') is not False:
        raise ValueError('Semantic completion must not fit source RGB')
    if payload.get('camera_history') != completion.get('camera_history') or len(payload.get('camera_history', [])) != 2064:
        raise ValueError('Completed camera trajectory is missing or inconsistent')
    train = contract.get('train_cameras', [])
    if len(train) != 63 or len(set(train)) != 63 or set(train) & {'DC', 'HC', 'KA'} or not set(payload['camera_history']) <= set(train):
        raise ValueError('Head must be fitted only on declared TRAIN63')
    state = payload.get('head', {})
    if set(state) != {'logits', 'bounds', 'world_to_box'}:
        raise ValueError('Unexpected semantic state schema')
    if state['bounds'].shape != (2, 3) or state['world_to_box'].shape != (4, 4) or any(x.dtype != torch.float32 or x.device.type != 'cpu' for x in state.values()):
        raise ValueError('Exact float32 CPU head buffers required')
    logits = state['logits']
    if logits.shape != (1, 1, 128, 128, 128) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError('Finite float32 XYZ128 logits required')
    if not np.array_equal(state['bounds'].numpy(), HEAD_BOUNDS) or not np.array_equal(np.asarray(contract['bounds'], dtype=np.float32), HEAD_BOUNDS):
        raise ValueError('Head bounds differ from reviewed floor box')
    if not torch.equal(state['world_to_box'], torch.eye(4, dtype=state['world_to_box'].dtype)):
        raise ValueError('Head frame buffer differs')
    if tensor_head_digest(state) != payload.get('head_digest'):
        raise ValueError('Head state digest differs')
    optimizer = payload.get('optimizer', {}).get('state', {})
    if len(optimizer) != 1 or any(int(x['step']) != 2064 for x in optimizer.values()):
        raise ValueError('Actual completed head Adam state required')
    return logits.numpy()[0, 0].copy()


def compose_candidates(core, original_active, admission, envelope):
    """No mutation. Primary order matches dilate(core,R)&admission&envelope."""
    arrays = [np.asarray(x) for x in (core, original_active, admission, envelope)]
    if any(x.dtype != np.bool_ for x in arrays) or len({x.shape for x in arrays}) != 1 or arrays[0].ndim != 4:
        raise ValueError('Matching boolean cascade,XYZ arrays required')
    if not np.array_equal(original_active, core & admission):
        raise ValueError('Source is not the specified actual R0 core/admission state')
    expanded = np.stack([fresh_dilation(x, 1) for x in core])
    r0 = core & admission & envelope
    r1 = expanded & admission & envelope
    alternative = np.stack([fresh_dilation(x, 1) for x in r0]) & admission & envelope
    if not (r0 <= r1).all() or not (alternative <= r1).all():
        raise AssertionError('Unexpected order/dilation containment')
    return dict(active_r0=r0, active_r1=r1,
        alternative_filter_then_dilate_r1=alternative,
        original_active=original_active.copy(), source_core=core.copy(),
        original_admission=admission.copy(), semantic_envelope=envelope.copy())


def reference_probabilities(logits, bounds, xyz):
    """FP64 CPU trilinear-logit reference, not CUDA deployment inference."""
    xyz = np.asarray(xyz, dtype=np.float64); bounds = np.asarray(bounds, dtype=np.float64)
    q = (xyz-bounds[0])/(bounds[1]-bounds[0])*(np.array(logits.shape)-1)
    inside = ((q >= 0) & (q <= np.array(logits.shape)-1)).all(1)
    q = np.clip(q, 0, np.array(logits.shape)-1)
    lo = np.floor(q).astype(int); hi = np.minimum(lo+1, np.array(logits.shape)-1); t = q-lo
    values = np.zeros(len(xyz), dtype=np.float64)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                choices = np.array([x, y, z]); ids = np.where(choices, hi, lo)
                weight = np.prod(np.where(choices, t, 1-t), 1)
                values += weight*logits[tuple(ids.T)]
    probability = np.empty_like(values)
    positive = values >= 0
    probability[positive] = 1/(1+np.exp(-values[positive]))
    exp = np.exp(values[~positive]); probability[~positive] = exp/(1+exp)
    probability *= inside
    return probability, values, inside


def source_point_report(records, logits, bounds, cascade_aabbs, arrays, selector):
    xyz = np.array([x['witness_xyz'] for x in records], dtype=np.float64)
    probability, logit, inside = reference_probabilities(logits, bounds, xyz)
    threshold = .5 if selector == 'hard_theta05' else .1
    selected = probability > threshold if selector.startswith('exactzero') else probability >= threshold
    margin = np.abs(logit-(np.log(threshold)-np.log1p(-threshold)))
    per_cascade = []
    for level, box in enumerate(np.asarray(cascade_aabbs).reshape(-1, 2, 3)):
        r = np.array(arrays['active_r0'].shape[1:])
        unit = (xyz-box[0])/(box[1]-box[0]); in_native = ((unit >= 0) & (unit <= 1)).all(1)
        ids = np.clip(np.floor(unit*r).astype(int), 0, r-1)
        checked = {key: (value[level][tuple(ids.T)] & in_native).tolist() for key, value in arrays.items() if key != 'source_core'}
        per_cascade.append(dict(level=level, native_cell_indices=ids.tolist(), inside_native_bounds=in_native.tolist(), classifications=checked))
    return dict(records=records, observations=len(records), unique_XYZ=len(np.unique(xyz, axis=0)),
        probability_reference_FP64=probability.tolist(), interpolated_logit_reference_FP64=logit.tolist(),
        inside_semantic_bounds=inside.tolist(), selected_by_reference=selected.tolist(),
        near_threshold_arithmetic_guard=(margin <= LOGIT_GUARD).tolist(), per_cascade=per_cascade,
        diagnostic_only=True, labels_or_targets_changed=False, physical_geometry_certified=False)


def validate_binding(binding_path):
    binding = read(binding_path)
    if binding.get('schema') != 'LC_SEMANTIC_CELL_CPU_V1' or binding.get('ready') is not True:
        raise ValueError('Explicit CPU candidate binding schema required')
    files = binding['files']; required = {'head', 'completion', 'spec', 'contract', 'capture', 'labels', 'adapter', 'alias_equivalence', 'grid_json', 'grid_npz', 'clothing103', 'shoe6'}
    if set(files) != required:
        raise ValueError('Missing or unexpected source binding roles')
    paths = {}; hashes = {}
    for key, item in files.items():
        p = Path(item['path'])
        if sha(p) != item['sha256']:
            raise ValueError('Bound source drift: '+key)
        paths[key] = p; hashes[key] = item['sha256']
    if hashes['capture'] != CAPTURE_SHA:
        raise ValueError('Wrong exact completed LC capture')
    if any(hashes[key] != value for key, value in QUERY_SHA.items()):
        raise ValueError('Sealed109 source observation records differ')
    capture, grid, contract = [read(paths[k]) for k in ('capture', 'grid_json', 'contract')]
    if capture.get('passed') is not True or capture.get('source_step') != 126316 or capture.get('source_FR') != .3 or capture.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT or not all(capture.get('unchanged', {}).values()):
        raise ValueError('Unqualified unchanged native capture')
    if grid.get('schema') != 2 or grid.get('radius') != 0 or grid.get('cpu_reconstruction_used') is not False or grid.get('checkpoint_sha256') != SOURCE_CHECKPOINT or grid.get('checkpoint_step') != 126316 or grid.get('grid_archive_sha256') != hashes['grid_npz']:
        raise ValueError('Wrong actual native occupancy source')
    spec, adapter, labels, alias = [read(paths[k]) for k in ('spec', 'adapter', 'labels', 'alias_equivalence')]
    if adapter.get('manifest_sha256') != hashes['contract'] or adapter.get('original_receipt_sha256') != CAPTURE_SHA or adapter.get('model_digest') != SOURCE_PIPELINE or not adapter.get('native_model_unchanged'):
        raise ValueError('Capture schema alias differs')
    if labels.get('capture_receipt_sha256') != hashes['adapter'] or labels.get('reviewer') != 'root' or not labels.get('root_review_complete'):
        raise ValueError('Wrong root-qualified semantic targets')
    if alias.get('alias_contract_sha256') != hashes['contract'] or alias.get('alias_label_qualification_sha256') != hashes['labels'] or alias.get('canonical_capture_receipt_sha256') != CAPTURE_SHA or alias.get('original_CP_hash') != SOURCE_CHECKPOINT or not alias.get('all_cache_arrays_and_source_metadata_byte_identical'):
        raise ValueError('Unverified remote/local alias provenance')
    if hashes['spec'] not in alias.get('alias_specs_sha256', {}).values():
        raise ValueError('Spec absent from explicit qualified alias')
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Run this preparation in a CPU-only fresh process')
    rng = torch.get_rng_state().clone()
    payload = torch.load(paths['head'], map_location='cpu', weights_only=False)
    logits = validate_head(payload, read(paths['completion']), spec, contract, hashes)
    with np.load(paths['grid_npz'], allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in ('core', 'active', 'admission', 'ema', 'cascade_aabbs', 'resolution')}
    if saved['core'].shape[1:] != (128, 128, 128) or not np.array_equal(saved['resolution'], [128]*3) or not np.isfinite(saved['ema']).all():
        raise ValueError('Wrong actual native cell schema')
    if any(saved[k].dtype != np.bool_ or saved[k].shape != saved['core'].shape for k in ('core', 'active', 'admission')) or saved['ema'].shape != saved['core'].shape:
        raise ValueError('Boolean actual source occupancy arrays required')
    if hashlib.sha256(saved['admission'].tobytes()).hexdigest() != grid['semantic_receipt']['mapped_bool_sha256']:
        raise ValueError('Original semantic admission bytes differ')
    if not np.array_equal(saved['active'], saved['core'] & saved['admission']):
        raise ValueError('Saved source R0 consistency failed')
    records = [p for key in ('clothing103', 'shoe6') for p in read(paths[key])['pixels']]
    if len(read(paths['clothing103'])['pixels']) != 103 or len(read(paths['shoe6'])['pixels']) != 6:
        raise ValueError('Exact sealed109 witness observations required')
    if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError('CPU validator changed Torch RNG or initialized CUDA')
    return binding, paths, hashes, payload, logits, saved, records


def seal_binding(source_paths, output):
    """Bind existing completed inputs, validate, then atomically publish metadata.

    Paths may be explicit local aliases; their bytes and the original alias
    equivalence are verified. Missing heads fail; this never waits or dispatches.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing binding; use a separate generation')
    value = dict(schema='LC_SEMANTIC_CELL_CPU_V1', ready=True,
        files={key: dict(path=str(Path(path).resolve()), sha256=sha(path)) for key, path in source_paths.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', dir=output.parent, delete=False) as stream:
        draft = Path(stream.name); json.dump(value, stream, indent=2)
    try:
        validate_binding(draft)
        draft.replace(output)
    except BaseException:
        draft.unlink(missing_ok=True)
        raise
    return value


def export(binding_path, output, selectors, execute=False):
    if not selectors or len(set(selectors)) != len(selectors) or any(x not in SELECTORS for x in selectors):
        raise ValueError('Explicit unique predeclared physical selectors required')
    binding, paths, hashes, payload, logits, source, records = validate_binding(binding_path)
    proof = dict(validated=True, CPU_only=True, native_model_loaded=False, CUDA_started=False,
        binding_sha256=sha(binding_path), head_sha256=hashes['head'], selectors=selectors,
        iteration=2064, initial_probability=payload['initial_probability'], exported=False)
    if not execute:
        return proof
    output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for selector in selectors:
        gate = LogitNodeCellBridge(logits, payload['head']['bounds'].numpy(), selector=selector,
            world_to_box=payload['head']['world_to_box'].numpy(), align_corners=True,
            coordinate_guard=COORDINATE_GUARD, numerical_logit_guard=LOGIT_GUARD)
        envelope = gate.map_cascades(source['cascade_aabbs'], source['resolution'])
        arrays = compose_candidates(source['core'], source['active'], source['admission'], envelope)
        destination = output/selector; destination.mkdir()
        target = destination/'candidate.npz'
        np.savez_compressed(target, **arrays, cascade_aabbs=source['cascade_aabbs'], resolution=source['resolution'])
        points = source_point_report(records, logits, payload['head']['bounds'].numpy(), source['cascade_aabbs'], arrays, selector)
        write(destination/'witnesses.json', points)
        metadata = dict(schema='LC_SEMANTIC_CELL_CANDIDATE_V1', candidate_sha256=sha(target),
            binding_sha256=sha(binding_path), inputs=hashes, source_checkpoint_sha256=SOURCE_CHECKPOINT,
            source_pipeline_digest=SOURCE_PIPELINE, head_sha256=hashes['head'], source_grid_sha256=hashes['grid_npz'],
            selector=selector, initial_probability=payload['initial_probability'], head_iteration=2064,
            head_digest=payload['head_digest'], coordinate_guard=COORDINATE_GUARD, numerical_logit_guard=LOGIT_GUARD,
            node_contract=dict(storage='XYZ', grid_sample_coordinates='ZYX', interpolation='trilinear_LOGIT_then_sigmoid',
                align_corners=True, padding_mode='border', resolution=[128]*3, bounds=payload['head']['bounds'].tolist(),
                world_to_box=np.eye(4).tolist(), outside_probability=0, threshold_equality='inactive' if selector.startswith('exactzero') else 'active'),
            composition=dict(primary='active_R = fresh_dilate(source_core,R) & original_admission & semantic_envelope',
                alternative='fresh_dilate(source_core & original_admission & semantic_envelope,1) & original_admission & semantic_envelope'),
            counts={k: int(v.sum()) for k, v in arrays.items()},
            r1_new_vs_original=int((arrays['active_r1'] & ~source['active']).sum()),
            alternative_r1_xor_primary=int((arrays['active_r1'] ^ arrays['alternative_filter_then_dilate_r1']).sum()),
            witnesses_sha256=sha(destination/'witnesses.json'), source_masks_and_records_unchanged=True,
            implementation_sha256={Path(__file__).name: sha(__file__),
                'marshawn_semantic_node_cell_bridge.py': sha(Path(__file__).with_name('marshawn_semantic_node_cell_bridge.py'))},
            CPU_only=True, phi_installed_or_changed=False, runtime_integration_authorized=False,
            prerequisites_for_future_replay=['Install exactly bound physical density selector separately; this array loader never changes phi.',
                'Actual CUDA interpolation/threshold support checks, all109 witness probabilities, native/dense/full-render coverage.',
                'Native source core may omit raw sigma; semantic support does not repair false base cells.',
                'FreshR1 exposes potentially untrained/skipped sigma and changes stepping phase; independently inspect it.',
                'Support certificate is for chosen semantic gate only, not physical free space or complete actor/hair/feet coverage.'])
        write(destination/'metadata.json', metadata)
        outputs[selector] = dict(candidate_sha256=sha(target), metadata_sha256=sha(destination/'metadata.json'), counts=metadata['counts'])
    for key, path in paths.items():
        if sha(path) != hashes[key]:
            raise ValueError('Source changed during CPU export: '+key)
    proof.update(exported=True, outputs=outputs, source_files_unchanged=True, quality_promoted=False)
    write(output/'complete.json', proof)
    return proof


def load_candidate(path, *, expected_metadata_sha256, expected_head_sha256, expected_source_grid_sha256, expected_selector, radius):
    """Future replay API: read-only arrays plus explicit phi contract; no hook."""
    path = Path(path); metadata = read(path/'metadata.json')
    if sha(path/'metadata.json') != expected_metadata_sha256:
        raise ValueError('Exact candidate metadata binding differs')
    if radius not in (0, 1) or expected_selector not in SELECTORS:
        raise ValueError('Explicit qualified selector and native radius required')
    if metadata.get('schema') != 'LC_SEMANTIC_CELL_CANDIDATE_V1' or metadata.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT or metadata.get('head_sha256') != expected_head_sha256 or metadata.get('source_grid_sha256') != expected_source_grid_sha256 or metadata.get('selector') != expected_selector:
        raise ValueError('Candidate belongs to a different field, head, source grid or phi')
    if metadata.get('phi_installed_or_changed') is not False or sha(path/'candidate.npz') != metadata['candidate_sha256']:
        raise ValueError('Unexpected mutation or candidate drift')
    with np.load(path/'candidate.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ('active_r'+str(radius), 'cascade_aabbs', 'resolution')}
    for value in arrays.values():
        value.flags.writeable = False
    return arrays, metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selector', action='append', choices=SELECTORS)
    p.add_argument('--seal-paths', type=Path, help='JSON role→existing path: seal/validate binding only, no candidate arrays')
    p.add_argument('--execute', action='store_true', help='Opt in to a separate CPU candidate export; never changes source state')
    a = p.parse_args()
    if a.seal_paths:
        if a.execute or a.selector:
            p.error('Sealing and candidate export are separate explicit steps')
        seal_binding(read(a.seal_paths), a.binding)
        print(json.dumps(dict(binding=str(a.binding), validated=True, CPU_only=True, candidates_exported=False)))
    else:
        print(json.dumps(export(a.binding, a.output, a.selector, a.execute), indent=2))
