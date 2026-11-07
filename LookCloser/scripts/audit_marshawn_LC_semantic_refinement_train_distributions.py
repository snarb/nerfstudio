"""TRAIN-only post-fit class distributions; fixed thresholds, no model updates."""
import argparse
import json
import os
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(spec_path, execute):
    s = json.loads(spec_path.read_text())
    if not execute:
        print(json.dumps(dict(phase='TRAIN_only_class_distributions', spec_sha256=sha(spec_path),
            optimizer_created=False, CUDA_started=False, threshold_selection=False)))
        return
    release = json.loads(Path(s['root_release']).read_text())
    if release.get('reviewer') != 'root' or release.get('LC_train_distribution_audit_authorized') is not True or release['spec_sha256'].get(str(spec_path)) != sha(spec_path):
        raise ValueError('Exact root TRAIN-distribution audit release required')
    for path, digest in release['bound_files'].items():
        if sha(path) != digest: raise ValueError('Bound source differs: '+path)
    from marshawn_gpu_isolation import require_controller_lease
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    from marshawn_frozen_semantic_head import FrozenActorGrid, rendered_classes, tensor_digest
    from run_marshawn_frozen_semantic_probe import capture_health
    torch.set_num_threads(2); torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1: raise ValueError('Exactly one leased device required')
    m = json.loads(Path(s['manifest']).read_text()); capture = json.loads(Path(s['capture_receipt']).read_text())
    if m['source_kind'] != 'DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or m['native_world_to_box'] != np.eye(4).tolist(): raise ValueError('Independent LC identity-frame visibility required')
    if torch.__version__ != s['torch_version'] or torch.version.cuda != s['cuda_version']: raise ValueError('Exact LC runtime required')
    import csv
    with Path(s['original_native_capacity136_csv']).open() as f: original136 = {(r['camera'], int(r['x']), int(r['y'])) for r in csv.DictReader(f)}
    if len(original136) != 136: raise ValueError('Keep all original136 source-query identities')
    labels = json.loads(Path(s['label_qualification']).read_text()); completion = json.loads(Path(s['head_completion']).read_text())
    if not labels['root_review_complete'] or sha(completion['checkpoint']) != completion['checkpoint_sha256']:
        raise ValueError('Qualified labels/head provenance differs')
    trained = torch.load(completion['checkpoint'], map_location='cpu', weights_only=False)
    if trained['iteration'] != s['expected_head_iteration'] or trained['spec_sha256'] != s['head_training_spec_sha256'] or trained['label_qualification_sha256'] != sha(s['label_qualification']):
        raise ValueError('Head is not the reviewed first-physical result')
    initial_probability = float(trained.get('initial_probability', .99))
    if initial_probability != float(s.get('expected_initial_probability', .99)):
        raise ValueError('Semantic initial prior differs')
    from marshawn_LC_semantic_refinement_physical import physical_refinement_head
    head = physical_refinement_head(trained, completion, s, m['bounds']).cuda()
    digest = tensor_digest(head)
    overrides = {(r['camera'], *r['pixel_xy']) for r in labels['unknown_queries']}
    out = Path(s['output']); out.mkdir(parents=True, exist_ok=False)
    group_names = ['positive', 'negative', 'qualified_bag_negative', 'near_clear_negative', 'far_clear_negative', 'LC_capacity_limited_positive', 'original_NGP_capacity136_positive']
    groups = {k: dict(rays=0, no_native_hit=0, conditional=[], inside_conditional=[],
        unweighted_sample_p_hist20=np.zeros(20), native_weighted_sample_p_hist20=np.zeros(20), native_weight_sum=0.) for k in group_names}
    records = []
    with torch.inference_mode():
        for row in capture['records']:
            if not row['rays']: continue
            if sha(row['archive']) != row['archive_sha256']: raise ValueError('Native cache differs')
            with np.load(row['archive']) as z:
                cache = {k: z[k] for k in ['xyz', 'weight', 'ray_indices', 'semantic_target', 'label_weight', 'pixel_xy', 'category', 'native_alpha', 'accepted_box_actor_cap']}
            n = len(cache['semantic_target']); confidence = cache['label_weight'].copy()
            for index, pixel in enumerate(cache['pixel_xy']):
                if (row['camera'], *pixel.tolist()) in overrides: confidence[index] = 0
            p = head(torch.tensor(cache['xyz'], device='cuda'))
            q = rendered_classes(p, torch.tensor(cache['weight'], device='cuda'), torch.tensor(cache['ray_indices'], device='cuda'), n)
            actor_mass = q[:, 0].cpu().numpy(); sample_p = p.cpu().numpy()
            alpha, cap = cache['native_alpha'], cache['accepted_box_actor_cap']
            posterior = actor_mass/np.maximum(alpha, 1e-12)
            inside_posterior = actor_mass/np.maximum(cap, 1e-12)
            eligible = confidence > 0; target = cache['semantic_target']; category = cache['category']
            selections = dict(positive=eligible & (target > .5), negative=eligible & (target < .5),
                qualified_bag_negative=eligible & (category == 'qualified_bag_interior'),
                near_clear_negative=eligible & (category == 'near_qualified_clear'), far_clear_negative=eligible & (category == 'far_qualified_clear'),
                LC_capacity_limited_positive=eligible & (target > .5) & (cap/np.maximum(alpha, 1e-12) < .95),
                original_NGP_capacity136_positive=eligible & (target > .5) & np.array([(row['camera'], *uv.tolist()) in original136 for uv in cache['pixel_xy']]))
            for name, selected in selections.items():
                g = groups[name]; g['rays'] += int(selected.sum()); hit = selected & (alpha > 1e-8)
                g['no_native_hit'] += int((selected & ~hit).sum()); g['conditional'].extend(posterior[hit].tolist())
                inside_hit = selected & (cap > 1e-8); g['inside_conditional'].extend(inside_posterior[inside_hit].tolist())
                chosen_samples = selected[cache['ray_indices']]
                g['unweighted_sample_p_hist20'] += np.histogram(sample_p[chosen_samples], bins=20, range=(0, 1))[0]
                g['native_weighted_sample_p_hist20'] += np.histogram(sample_p[chosen_samples], bins=20, range=(0, 1), weights=cache['weight'][chosen_samples].astype(np.float64))[0]
                g['native_weight_sum'] += float(cache['weight'][chosen_samples].sum(dtype=np.float64))
            for index in np.flatnonzero(eligible):
                records.append(dict(camera=row['camera'], camera_ray_index=int(index), pixel_xy=cache['pixel_xy'][index].tolist(),
                    target=float(target[index]), category=str(category[index]), native_alpha=float(alpha[index]), insidecap=float(cap[index]),
                    Qactor_over_native_alpha=float(posterior[index]), Qactor_over_insidecap_DIAGNOSTIC_ONLY=float(inside_posterior[index])))
            write(out/'progress.json', dict(completed_cameras=len({r['camera'] for r in records}), health=capture_health(torch, lease['expected_physical_uuid'])))
    for name, g in groups.items():
        for key in ['conditional', 'inside_conditional']:
            values = np.asarray(g.pop(key), np.float64)
            g[key+'_quantiles'] = np.quantile(values, [0, .01, .1, .5, .9, .99, 1]).tolist() if len(values) else []
            g[key+'_fraction_ge_fixed_threshold'] = {str(t): float((values >= t).mean()) for t in [.1, .5]} if len(values) else {}
        for key in ['unweighted_sample_p_hist20', 'native_weighted_sample_p_hist20']: g[key] = g[key].tolist()
    if groups['original_NGP_capacity136_positive']['rays'] != 136: raise ValueError('Original136 must remain genuine fit queries without capacity filtering')
    if tensor_digest(head) != digest: raise ValueError('Frozen head changed during TRAIN audit')
    write(out/'distributions.json', dict(groups=groups, records=records, excluded_UNKNOWN_queries=len(overrides),
        head_checkpoint_sha256=completion['checkpoint_sha256'], head_type=trained['head_type'], head_schema_version=1, refinement_family=trained['refinement_family'], head_unchanged=True, optimizer_created=False, source_EVAL_inputs_read=False,
        initial_semantic_probability=initial_probability, source_kind=m['source_kind'],LC_capture_receipt_sha256=m['LC_capture_receipt_sha256'],
        capacity_groups='Original NGP136 identities retained separately from current LC cap/A<.95; neither group renormalizes targets or excludes rays.',
        threshold_selection=False, fixed_thresholds=[.1, .5],
        definitions='Qactor/nativealpha is the actual native-visibility conditional class render. Qactor/insidecap is a separately labeled diagnostic of classifier convergence on in-box native mass ONLY, never a training-target normalization. Samplep histograms are separately unweighted and nativevisibilityweighted; hiddenp is not semantic evidence.',
        no_hit_policy='No-native-hit rays are counted separately; conditionalclass is undefined there and their originalalpha-weighted fitting objective is0. No query is removed from fitting.',
        next_step='Root may judge convergence from TRAIN distributions and separate physicalvisual gate; no threshold is chosen automatically or solely from EVAL.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec', type=Path, required=True); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); run(a.spec, a.execute)
