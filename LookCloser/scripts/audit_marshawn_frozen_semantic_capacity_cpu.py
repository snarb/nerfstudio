"""CPU audit of saved native visibility capacity and dense-background export risk."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from prepare_marshawn_frozen_semantic_cpu import sha, write


def audit(manifest_path, capture_root, output):
    m = json.loads(manifest_path.read_text())
    capture = json.loads((capture_root/'capture.json').read_text())
    assert capture['manifest_sha256'] == sha(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    frame = np.asarray(m['native_world_to_box'], np.float32)
    bounds = np.asarray(m['bounds'], np.float32)
    positives, negatives, maxima = [], [], []
    samples = compressed = uncompressed = 0
    for record in capture['records']:
        if not record['rays']:
            continue
        path = Path(record['archive'])
        assert sha(path) == record['archive_sha256']
        compressed += path.stat().st_size
        with zipfile.ZipFile(path) as archive:
            uncompressed += sum(row.file_size for row in archive.infolist())
        with np.load(path) as z:
            alpha, cap, target, xy, category = [z[k] for k in ['native_alpha', 'accepted_box_actor_cap', 'semantic_target', 'pixel_xy', 'category']]
            ids, sigma, starts, ends, xyz, weight = [z[k] for k in ['ray_indices', 'sigma', 'starts', 'ends', 'xyz', 'weight']]
        ratio = cap/np.maximum(alpha, 1e-12)
        positive = target > .5
        for pixel, a, c, r in zip(xy[positive], alpha[positive], cap[positive], ratio[positive]):
            positives.append(dict(camera=record['camera'], pixel=pixel.tolist(), native_alpha=float(a), insideboxcap=float(c), ratio=float(r)))
        box_xyz = xyz @ frame[:3, :3].T + frame[:3, 3]
        inside = ((box_xyz >= bounds[0]) & (box_xyz <= bounds[1])).all(1)
        tau = sigma*(ends-starts)
        tau_inside = np.bincount(ids, weights=tau*inside, minlength=record['rays'])
        hypothetical = -np.expm1(-.001*tau_inside)
        negative = target < .5
        for pixel, cat, t, a in zip(xy[negative], category[negative], tau_inside[negative], hypothetical[negative]):
            negatives.append(dict(camera=record['camera'], pixel=pixel.tolist(), category=str(cat), inside_optical_depth=float(t), hypothetical_soft_p001_actor_alpha=float(a)))
        global_id = int(np.argmax(sigma))
        inside_ids = np.flatnonzero(inside)
        chosen = int(inside_ids[np.argmax(sigma[inside_ids])]) if len(inside_ids) else global_id
        maxima.append(dict(camera=record['camera'], global_max=float(sigma[global_id]), global_max_inside=bool(inside[global_id]),
            inside_max=float(sigma[chosen]) if len(inside_ids) else 0, inside_max_ray_pixel=xy[ids[chosen]].tolist(),
            inside_max_ray_category=str(category[ids[chosen]]), inside_max_native_weight=float(weight[chosen]),
            inside_max_interval_tau=float(tau[chosen])))
        samples += len(ids)
    bad = sorted([r for r in positives if r['ratio'] < .95], key=lambda r: r['ratio'])
    ratios = np.array([r['ratio'] for r in positives])
    alphas = np.array([r['native_alpha'] for r in positives])
    hypothetical = np.array([r['hypothetical_soft_p001_actor_alpha'] for r in negatives])
    report = dict(native_source_unchanged=capture['native_model_unchanged'], native_RNG_unchanged=capture['native_RNG_unchanged'],
        rays=len(positives)+len(negatives), positives=len(positives), negatives=len(negatives), samples=samples,
        positive_capacity_below095=len(bad), positive_native_alpha_quantiles=np.quantile(alphas, [0, .001, .01, .1, .5, 1]).tolist(),
        positive_insidecap_ratio_quantiles=np.quantile(ratios, [0, .001, .01, .1, .5, 1]).tolist(),
        bad_positive_counts_by_camera={name: sum(r['camera'] == name for r in bad) for name in sorted({r['camera'] for r in bad})},
        bad_positives=bad, max_sigma_by_camera=maxima,
        hypothetical_p001_uniform_inside_box_on_negative_rays=dict(
            interpretation='Counterfactual only: p.001 at EVERY inside-box sample,0 outside; recomputealpha from actual sigma/intervals. Not a fitted head, alphaGT, or clean-asset claim. CPU float32 box membership may differ at numerical boundaries; saved native cap values above remain authoritative.',
            alpha_gt001=int((hypothetical > .01).sum()), alpha_gt01=int((hypothetical > .1).sum()),
            alpha_gt05=int((hypothetical > .5).sum()), alpha_gt099=int((hypothetical > .99).sum()),
            alpha_quantiles=np.quantile(hypothetical, [0, .5, .9, .99, 1]).tolist(),
            top30=sorted(negatives, key=lambda r: r['hypothetical_soft_p001_actor_alpha'], reverse=True)[:30]),
        archive_compressed_bytes=compressed, archive_uncompressed_bytes=uncompressed,
        capture_receipt_sha256=sha(capture_root/'capture.json'), helper_sha256=sha(__file__), CPU_only=True,
        no_fit_or_new_labels=True)
    write(output/'capacity_and_density.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['bad_positives', 'max_sigma_by_camera', 'hypothetical_p001_uniform_inside_box_on_negative_rays']}))
    print(json.dumps({k: v for k, v in report['hypothetical_p001_uniform_inside_box_on_negative_rays'].items() if k != 'top30'}))
    print(json.dumps(max(maxima, key=lambda r: r['inside_max'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.manifest, a.capture, a.output)
