"""Read-only capture before canonical actor-overlap deletion, then V5 resolution.

install_capture() leaves original model/refinement arrays and canonical outputs
unchanged. resolve_captured() is an explicit, separately recorded operation on
an isolated campaign output. Never call it on published training inputs.
"""
from pathlib import Path
import hashlib, importlib.util, inspect, json, shutil
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def install_capture(module, capture_root):
    """Return a restore callback; callers must retain per-view source bindings."""
    capture_root = Path(capture_root); capture_root.mkdir(parents=True, exist_ok=True)
    original_refine = module.refine; original_np = module.np; captured = {}
    module_source_sha = sha(module.__file__)

    def observe_refine(*args, **kwargs):
        result = original_refine(*args, **kwargs)
        caller = inspect.currentframe().f_back
        if caller.f_code is not module.main.__code__:
            raise ValueError('Unexpected canonical refine caller')
        camera = caller.f_locals['camera']; actor = caller.f_locals['actor']
        if actor not in {'black', 'grey'}:
            raise ValueError('Unexpected actor name')
        captured[(camera, actor)] = result[0].copy()
        return result

    class NumpyProxy:
        def __getattr__(self, name):
            return getattr(original_np, name)

        def savez_compressed(self, path, *args, **kwargs):
            path = Path(path)
            if path.parent.name != 'native_masks' or not {'black', 'grey', 'source_ineligible', 'unknown', 'crop_xyxy'}.issubset(kwargs):
                return original_np.savez_compressed(path, *args, **kwargs)
            if args:
                raise ValueError('Unexpected canonical native save arguments')
            camera = path.stem
            pre = {actor: captured.pop((camera, actor), kwargs[actor].copy()) for actor in ['black', 'grey']}
            overlap = pre['black'] & pre['grey']
            for actor in ['black', 'grey']:
                if not np.array_equal(kwargs[actor], pre[actor] & ~overlap):
                    raise AssertionError('Canonical post-clear output differs from exact pre-clear replay')
            archive = capture_root / (camera + '.npz')
            if archive.exists():
                raise ValueError('Pre-clear capture already exists; refuse overwrite')
            original_np.savez_compressed(archive, pre_black=pre['black'], pre_grey=pre['grey'],
                                        source_ineligible=kwargs['source_ineligible'], unknown=kwargs['unknown'], crop_xyxy=kwargs['crop_xyxy'])
            # Preserve the original save call and its exact canonical output.
            result = original_np.savez_compressed(path, **kwargs)
            write(capture_root / (camera + '.json'), dict(
                schema=5, camera=camera, frame=int(path.parent.parent.name),
                canonical_mask_sha256=module_source_sha, capture_code_sha256=sha(__file__),
                pre_clear_archive_sha256=sha(archive), canonical_post_clear_archive_sha256=sha(path),
                overlap_pixels=int(overlap.sum()), pre_arrays_sha256={k: array_sha(v) for k, v in pre.items()},
                canonical_post_clear_replay_exact=True,
                observation='Copies of original refine return before canonical destructive overlap clear; model/tiles/RNG unchanged'))
            return result

    module.refine = observe_refine; module.np = NumpyProxy()

    def restore():
        module.refine = original_refine; module.np = original_np
    return restore


def resolve_captured(native_root, capture_root, resolver_source, resolver_sha256):
    native_root = Path(native_root); capture_root = Path(capture_root)
    resolver_source = Path(resolver_source)
    if sha(resolver_source) != resolver_sha256:
        raise ValueError('V5 union resolver source changed')
    spec = importlib.util.spec_from_file_location('marshawn_v5_union_resolver', resolver_source)
    resolver = importlib.util.module_from_spec(spec); spec.loader.exec_module(resolver)
    results = {}
    for capture_receipt in sorted(capture_root.glob('[A-Z][A-Z].json')):
        camera = capture_receipt.stem; observed = json.loads(capture_receipt.read_text())
        captured_archive = capture_root / (camera + '.npz')
        path = native_root / (camera + '.npz'); receipt_path = native_root / (camera + '.json')
        record = json.loads(receipt_path.read_text())
        existing = record.get('union_resolution_v5')
        if existing:
            if existing['resolver_sha256'] != resolver_sha256 or sha(path) != record['archive_sha256']:
                raise ValueError('Existing V5 union result changed')
            results[camera] = existing
            continue
        if sha(captured_archive) != observed['pre_clear_archive_sha256'] or sha(path) != observed['canonical_post_clear_archive_sha256']:
            raise ValueError('Captured or canonical post-clear artifact changed')
        if record['archive_sha256'] != sha(path):
            raise ValueError('Canonical archive receipt differs')
        with np.load(captured_archive, allow_pickle=False) as z:
            pre = {key: z[key] for key in z.files}
        with np.load(path, allow_pickle=False) as z:
            old = {key: z[key] for key in z.files}
        for key in ['source_ineligible', 'unknown', 'crop_xyxy']:
            if not np.array_equal(pre[key], old[key]):
                raise ValueError('Independent veto/UNKNOWN or crop changed')
        # The canonical aggregate UNKNOWN may contain own-actor proposal
        # uncertainty. Only a separately qualified different-person UNKNOWN
        # belongs in the resolver's fourth argument; crew guard handles it later.
        arrays, union_receipt = resolver.resolve_actor_union(pre['pre_black'], pre['pre_grey'],
                                                           pre['source_ineligible'])
        audit = capture_root / 'canonical_post_clear'; audit.mkdir(exist_ok=True)
        shutil.copy2(path, audit / path.name); shutil.copy2(receipt_path, audit / receipt_path.name)
        output = dict(old, **arrays)
        temporary = path.with_name(camera + '.v5.tmp.npz')
        np.savez_compressed(temporary, **output); temporary.replace(path)
        union_receipt.update(resolver_sha256=resolver_sha256, capture_receipt_sha256=sha(capture_receipt),
                             pre_clear_archive_sha256=sha(captured_archive),
                             canonical_post_clear_archive_sha256=observed['canonical_post_clear_archive_sha256'],
                             canonical_post_clear_receipt_sha256=sha(audit / receipt_path.name),
                             capture_code_sha256=sha(__file__),
                             actor_ownership_uncertain_is_geometry_unknown=False,
                             uncertain_pixels_excluded_from_future_identity_anchors=True)
        record.update(archive_sha256=sha(path), union_resolution_v5=union_receipt)
        write(receipt_path, record); results[camera] = union_receipt
    write(capture_root / 'resolution_ready.json', dict(version=5, records=results,
          prepared_publication_allowed=False, isolated_result_requires_visual_review=True))
    return results


def seed_with_ownership_policy(module, campaign, frame, camera, original_seed=None):
    """Canonical prior identity selection, excluding uncertain ownership only."""
    campaign = Path(campaign)
    original_seed = original_seed or module.seed
    reference, previous, reference_sha = original_seed(campaign, frame, camera)
    reference = {actor: value.copy() for actor, value in reference.items()}
    path = campaign / 'frames' / ('%06d' % previous) / 'native_masks' / (camera + '.npz')
    receipt = dict(version=5, frame=frame, camera=camera, reference_frame=previous,
                   reference_archive_sha256=reference_sha,
                   input_arrays_sha256={k: array_sha(v) for k, v in reference.items()},
                   uncertainty_present=False, uncertain_native_pixels=0,
                   policy='Original prior-frame identity choice; remove ownership-uncertain native pixels before original small()')
    if path.exists():
        if sha(path) != reference_sha:
            raise ValueError('Canonical seed archive binding differs')
        with np.load(path, allow_pickle=False) as z:
            if 'actor_ownership_uncertain' in z.files:
                uncertain = z['actor_ownership_uncertain']
                if uncertain.dtype != np.bool_:
                    raise ValueError('Ownership uncertainty must be boolean')
                for actor in ['black', 'grey']:
                    if uncertain.shape != z[actor].shape:
                        raise ValueError('Ownership uncertainty and actor shapes differ')
                    reference[actor] = module.small(z[actor] & ~uncertain)
                receipt.update(uncertainty_present=True, uncertain_native_pixels=int(uncertain.sum()),
                               uncertainty_sha256=array_sha(uncertain))
    receipt['output_arrays_sha256'] = {k: array_sha(v) for k, v in reference.items()}
    receipt['legacy_reference_arrays_unchanged'] = receipt['input_arrays_sha256'] == receipt['output_arrays_sha256']
    return (reference, previous, reference_sha), receipt
