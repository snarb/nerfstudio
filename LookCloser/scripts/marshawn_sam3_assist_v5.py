"""Versioned isolated V5 union capture/transport over the sealed SAM3 worker."""
from pathlib import Path
import argparse, json, shutil
import marshawn_sam3_assist as base
import marshawn_sam3_union_capture_v5 as capture


def pack(campaign, frame, camera, task_root, remote_task, remote_code, remote_overlay, resolver):
    campaign = Path(campaign); task_root = Path(task_root); resolver = Path(resolver)
    if '/v5/' not in str(remote_task):
        raise ValueError('V5 requires an isolated remote task namespace')
    existing = task_root / 'task.json'
    if existing.exists():
        spec = base.read(existing)
        if spec['v5_adapter_sha256'] != base.sha(__file__) or spec['v5_capture_sha256'] != base.sha(capture.__file__) or spec['v5_resolver_sha256'] != base.sha(resolver):
            raise ValueError('Existing V5 task code changed')
        base.validate_task(spec, task_root)
        return spec
    module = base.canonical(campaign / 'code'); original_seed = module.seed; policy = {}

    def seed_io(root, current, cam):
        result, receipt = capture.seed_with_ownership_policy(module, root, current, cam, original_seed)
        policy.update(receipt)
        return result

    module.seed = seed_io
    try:
        spec = base.pack(campaign, frame, camera, task_root, remote_task, remote_code, remote_overlay)
    finally:
        module.seed = original_seed
    shutil.copy2(resolver, task_root / 'union_resolver.py')
    base.write(task_root / 'seed_policy_v5.json', policy)
    spec.update(v5_adapter_sha256=base.sha(__file__), v5_capture_sha256=base.sha(capture.__file__),
                v5_resolver_sha256=base.sha(resolver), v5_seed_policy_sha256=base.sha(task_root / 'seed_policy_v5.json'),
                v5_operation='ISOLATED_PRE_CLEAR_CAPTURE_AND_UNION_RESOLUTION', prepared_publication_allowed=False)
    base.write(existing, spec)
    return spec


def remote_run(spec_path):
    spec_path = Path(spec_path); spec = base.read(spec_path); root = Path(spec['remote_task'])
    for path, expected in [(Path(__file__), spec['v5_adapter_sha256']),
                           (Path(capture.__file__), spec['v5_capture_sha256']),
                           (root / 'union_resolver.py', spec['v5_resolver_sha256']),
                           (root / 'seed_policy_v5.json', spec['v5_seed_policy_sha256'])]:
        if base.sha(path) != expected:
            raise ValueError('V5 operation source or seed policy changed')
    if '/v5/' not in str(root):
        raise ValueError('V5 result must remain isolated')
    original_canonical = base.canonical; restore = []

    def canonical_with_capture(code):
        module = original_canonical(code)
        restore.append(capture.install_capture(module, root / 'pre_clear'))
        return module

    base.canonical = canonical_with_capture
    try:
        canonical_proof = base.remote_run(spec_path)
    finally:
        base.canonical = original_canonical
        for callback in reversed(restore):
            callback()
    base.write(root / 'canonical_operation_receipt.json', canonical_proof)
    result = root / 'campaign/frames' / ('%06d' % spec['frame']) / 'native_masks'
    resolved = capture.resolve_captured(result, root / 'pre_clear', root / 'union_resolver.py', spec['v5_resolver_sha256'])
    record = base.read(result / (spec['camera'] + '.json'))
    import numpy as np
    with np.load(result / (spec['camera'] + '.npz'), allow_pickle=False) as z:
        arrays = {key: base.array_sha(z[key]) for key in z.files}
    proof = dict(canonical_proof,
                 archive_sha256=base.sha(result / (spec['camera'] + '.npz')),
                 remote_mask_receipt_sha256=base.sha(result / (spec['camera'] + '.json')),
                 arrays_sha256=arrays,
                 v5_adapter_sha256=base.sha(__file__), v5_capture_sha256=base.sha(capture.__file__),
                 v5_resolver_sha256=spec['v5_resolver_sha256'], v5_seed_policy_sha256=spec['v5_seed_policy_sha256'],
                 canonical_operation_receipt_sha256=base.sha(root / 'canonical_operation_receipt.json'),
                 pre_clear_archive_sha256=base.sha(root / 'pre_clear' / (spec['camera'] + '.npz')),
                 pre_clear_receipt_sha256=base.sha(root / 'pre_clear' / (spec['camera'] + '.json')),
                 union_resolution=resolved[spec['camera']],
                 operation='Unchanged canonical SAM3/refine; readonly pre-clear capture; explicit separately recorded V5 actor-union resolution',
                 prepared_publication_allowed=False)
    if not record['union_resolution_v5']['union_conserved']:
        raise AssertionError('V5 actor union not conserved')
    base.write(root / 'operation_receipt.json', proof)
    return proof


def collect(campaign, task_root, isolated_campaign):
    campaign = Path(campaign).resolve(); isolated_campaign = Path(isolated_campaign).resolve()
    if campaign == isolated_campaign or '/operations/sam3_assist/v5/' not in str(isolated_campaign):
        raise ValueError('V5 collector only publishes isolated review outputs')
    task_root = Path(task_root); spec = base.read(task_root / 'task.json')
    proof = base.read(task_root / 'returned/operation_receipt.json')
    for key in ['v5_adapter_sha256', 'v5_capture_sha256', 'v5_resolver_sha256', 'v5_seed_policy_sha256']:
        if proof[key] != spec[key]:
            raise ValueError('V5 returned code/seed binding differs: ' + key)
    if proof['canonical_operation_receipt_sha256'] != base.sha(task_root / 'returned/canonical_operation_receipt.json'):
        raise ValueError('Original canonical operation receipt changed')
    c = spec['camera']; pre = task_root / 'returned/pre_clear'
    if base.sha(pre / (c + '.npz')) != proof['pre_clear_archive_sha256'] or base.sha(pre / (c + '.json')) != proof['pre_clear_receipt_sha256']:
        raise ValueError('Pre-clear observation changed')
    observed = base.read(pre / (c + '.json'))
    if not observed['canonical_post_clear_replay_exact'] or not proof['union_resolution']['union_conserved']:
        raise ValueError('Canonical replay or actor union verification failed')
    import numpy as np
    native = task_root / 'returned/native_masks' / (c + '.npz')
    with np.load(native, allow_pickle=False) as z:
        expected = {'black', 'grey', 'source_ineligible', 'unknown', 'crop_xyxy', 'actor_ownership_uncertain'}
        if set(z.files) != expected:
            raise ValueError('V5 native array schema differs')
        for key in z.files:
            if base.array_sha(z[key]) != proof['arrays_sha256'][key]:
                raise ValueError('V5 returned array changed: ' + key)
            if key != 'crop_xyxy' and (z[key].shape != (4320, 8192) or z[key].dtype != np.bool_):
                raise ValueError('V5 native raster/dtype differs')
        with np.load(pre / (c + '.npz'), allow_pickle=False) as before:
            for key in ['source_ineligible', 'unknown', 'crop_xyxy']:
                if not np.array_equal(z[key], before[key]):
                    raise ValueError('Independent veto/UNKNOWN/crop changed in transport')
            union_before = (before['pre_black'] | before['pre_grey']) & ~before['source_ineligible']
            if not np.array_equal((z['black'] | z['grey']) & ~z['source_ineligible'], union_before):
                raise ValueError('Transported V5 actor union differs')
            if not np.array_equal(z['actor_ownership_uncertain'], before['pre_black'] & before['pre_grey'] & ~before['source_ineligible']):
                raise ValueError('Transported actor ownership uncertainty differs')
        if (z['black'] & z['grey']).any():
            raise ValueError('V5 ownership labels overlap')
    gpu = proof['gpu']
    if gpu['actual_physical_uuid'] != gpu['expected_physical_uuid'] or gpu['visible_device_count'] != 1:
        raise ValueError('V5 GPU identity proof differs')
    isolated_campaign.mkdir(parents=True, exist_ok=True)
    for name in ['sources', 'code']:
        destination = isolated_campaign / name
        if not destination.exists():
            destination.symlink_to(campaign / name, target_is_directory=True)
        if destination.resolve() != (campaign / name).resolve():
            raise ValueError('Isolated source/code alias changed')
    result = base.collect(isolated_campaign, task_root)
    retained = isolated_campaign / 'frames' / ('%06d' % spec['frame']) / 'pre_clear' / c
    if retained.exists():
        raise ValueError('Isolated pre-clear evidence already exists')
    shutil.copytree(pre, retained)
    shutil.copy2(task_root / 'returned/canonical_operation_receipt.json', retained / 'canonical_operation_receipt.json')
    return dict(result, prepared_publication_allowed=False, visual_review_required=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['pack', 'run', 'collect'])
    p.add_argument('--campaign', type=Path); p.add_argument('--frame', type=int); p.add_argument('--camera')
    p.add_argument('--task-root', type=Path); p.add_argument('--remote-task'); p.add_argument('--remote-code'); p.add_argument('--remote-overlay')
    p.add_argument('--resolver', type=Path); p.add_argument('--spec', type=Path); p.add_argument('--isolated-campaign', type=Path)
    a = p.parse_args()
    if a.action == 'pack':
        result = pack(a.campaign, a.frame, a.camera, a.task_root, a.remote_task, a.remote_code, a.remote_overlay, a.resolver)
    elif a.action == 'run':
        result = remote_run(a.spec)
    else:
        result = collect(a.campaign, a.task_root, a.isolated_campaign)
    print(json.dumps(result), flush=True)
