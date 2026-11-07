"""Compare immutable V1 with a separately reviewed source-only bag exclusion.

This CPU diagnostic changes no dataset, training input, crop, or leaderboard.
Every compared render receives exactly the same exclusion. Historical V1 is
always reported alongside the explicitly named annotation diagnostic.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(root, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import lpips
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(4)
    proposal_root = root / 'eval_actor_mask_source_audit_CPU_v1'
    proposal_path = proposal_root / 'proposal_manifest.json'
    proposal = json.loads(proposal_path.read_text())
    approval = json.loads((proposal_root / 'root_manual_source_review.json').read_text())
    if (approval['manifest_sha256'] != sha(proposal_path)
            or approval.get('source_only_diagnostic_exclusion_approved') is not True
            or approval.get('historical_V1_changed') is not False):
        raise ValueError('Exact source-only manual review is required')
    ready = json.loads((root / 'data/ready.json').read_text())
    records = {r['camera']: r for r in ready['records']}
    records_proposal = {r['camera']: r for r in proposal['records']}
    helper_path = root / 'frozen_LC_semantic_physical_plan_v3/code/marshawn_tuning_metrics.py'
    spec = importlib.util.spec_from_file_location('_fixed_metrics', helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    metric = lpips.LPIPS(net='alex').eval().cpu().requires_grad_(False)
    p3 = root / 'frozen_LC_semantic_physical_plan_v3'
    variants = [('native_source', p3 / 'prior001/renders', 'native_full')]
    for prior in ['prior001', 'prior099']:
        for selector in ['soft_probability', 'hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']:
            variants.append((prior + '_' + selector, p3 / prior / 'renders', selector))
    # Add the already completed four-UNKNOWN candidate symmetrically, if present.
    new = root / 'frozen_LC_semantic_LA_two_UNKNOWN_physical_plan_v1/prior001/renders'
    if (new / 'complete.json').is_file():
        for selector in ['soft_probability', 'hard_theta01', 'hard_theta05', 'exactzero_smoothstep01_05']:
            variants.append(('four_UNKNOWN_' + selector, new, selector))
    exclusions = {}
    inputs = {str(proposal_path): sha(proposal_path), str(helper_path): sha(helper_path)}
    for camera in ['DC', 'HC', 'KA']:
        item = records_proposal[camera]
        path = Path(item['proposal_NPZ'])
        if sha(path) != item['proposal_NPZ_sha256']:
            raise ValueError('Source proposal bytes changed')
        with np.load(path, allow_pickle=False) as arrays:
            exclusions[camera] = {actor: arrays[actor + '_exclusion'].copy() for actor in ['black', 'grey']}
        inputs[str(path)] = sha(path)
    counts = {(camera, actor): int(exclusions[camera][actor].sum())
              for camera in exclusions for actor in exclusions[camera]}
    if counts[('KA', 'grey')] != 28 or sum(counts.values()) != 28:
        raise ValueError('Only the exact28 source-qualified bag pixels are allowed')
    results = {}
    for name, render_root, key in variants:
        values = {'historical_V1': [], 'source_bag28_exclusion_diagnostic': []}
        for camera in ['DC', 'HC', 'KA']:
            rec = records[camera]
            gt = np.asarray(Image.open(root / 'data' / rec['image']).convert('RGB')).copy()
            render_path = render_root / camera / 'physical_and_partition.npz'
            with np.load(render_path, allow_pickle=False) as arrays:
                rgb = np.clip(arrays[key][..., :3], 0, 1).copy()
            inputs[str(render_path)] = sha(render_path)
            target = torch.from_numpy(gt).permute(2, 0, 1)[None].float() / 255
            prediction = torch.from_numpy(rgb).permute(2, 0, 1)[None]
            mask_path = root / 'data/semantics' / (camera + '.npz')
            inputs[str(mask_path)] = sha(mask_path)
            with np.load(mask_path, allow_pickle=False) as arrays:
                for actor in ['black', 'grey']:
                    original = arrays[actor].copy().astype(bool)
                    modified = original & ~exclusions[camera][actor]
                    for domain, mask in [('historical_V1', original), ('source_bag28_exclusion_diagnostic', modified)]:
                        score = helper.masked_actor_metrics(prediction, target,
                            torch.from_numpy(mask)[None, None], metric,
                            crop_xyxy=rec['crops'][actor])
                        values[domain].append(dict(view=camera, actor=actor, **score))
        results[name] = {domain: helper.aggregate_actor_metrics(rows) for domain, rows in values.items()}
        print(json.dumps({'variant_completed': name}), flush=True)
    if torch.cuda.is_initialized():
        raise ValueError('CPU diagnostic initialized CUDA')
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Input changed during diagnostic: ' + path)
    result = dict(CPU_only=True, CUDA_initialized=False, training_changed=False,
        historical_V1_unchanged=True, crops_unchanged=True, leaderboard_changed=False,
        all_variants_receive_identical_source_exclusion=True,
        annotation_is_not_optical_alpha=True, excluded_pixels=28,
        source_manifest_sha256=sha(proposal_path), inputs=inputs, variants=results)
    # Metric helper internals may contain a selection objective; reports contain
    # only the requested image metrics and their fixed-domain bookkeeping.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if 'loss' not in k}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing diagnostic')
    output.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
