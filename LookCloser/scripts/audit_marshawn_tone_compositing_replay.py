"""CPU-only same-packet tone-compositing diagnostic; never edits model/data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def encode(x):
    t = x / (1 + x)
    return np.where(t <= .0031308, 12.92*t, 1.055*t**(1/2.4)-.055)


def replay(rgb, weights, ray_indices, num_rays):
    rgb = np.asarray(rgb, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(ray_indices, dtype=np.int64)
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError('Packed RGB must be finite in [0,1]')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Invalid weights')
    if rgb.shape != (len(weights), 3) or indices.shape != weights.shape:
        raise ValueError('Packed shapes differ')
    if np.any((indices < 0) | (indices >= num_rays)):
        raise ValueError('Ray index outside declared packet')
    saturated = rgb == 1
    display = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    # No epsilon clipping. Saturated positive-weight channels are censored below.
    linear = np.divide(display, 1-display, out=np.zeros_like(display), where=~saturated)
    def scatter(v):
        return np.stack([np.bincount(indices, weights=v[:,j], minlength=num_rays) for j in range(3)], 1)
    native = scatter(weights[:,None]*rgb)
    censored = scatter(((weights > 0)[:,None] & saturated).astype(float)) > 0
    finite_component_lower_bound = encode(scatter(weights[:,None]*linear))
    physical = np.where(censored, np.nan, finite_component_lower_bound)
    return dict(native=native, physical=physical, valid=~censored,
                finite_component_lower_bound=finite_component_lower_bound,
                saturated_weight=scatter(weights[:,None]*saturated),
                alpha=np.bincount(indices, weights=weights, minlength=num_rays))


def scores(pred, gt, valid):
    if not valid.any():
        return None
    err = (pred-gt)[valid]
    return dict(channels=int(valid.sum()), MAE=float(np.mean(abs(err))),
                RMSE=float(np.sqrt(np.mean(err**2))), signed_mean=float(err.mean()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--source-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    x = np.load(a.cache, allow_pickle=False)
    gt = x['GT_5x5'].astype(float)
    r = replay(x['rgb_samples'], x['weights'], x['ray_indices'], len(gt))
    assert np.max(abs(r['native']-x['native_RGB'])) < 1e-6
    valid, complete = r['valid'], r['valid'].all(1)
    delta = r['physical']-r['native']
    assert np.min(delta[valid]) > -1e-7
    source = json.loads(a.source_audit.read_text())
    rows = []
    for i in range(len(gt)):
        rows.append(dict(index=i, camera=str(x['camera_names'][i]), candidate_id=int(x['candidate_ids'][i]),
                         native=r['native'][i].tolist(), GT_5x5=gt[i].tolist(),
                         physical=[float(v) if valid[i,j] else None for j,v in enumerate(r['physical'][i])],
                         invertible_channels=valid[i].tolist(), saturated_weight=r['saturated_weight'][i].tolist()))
    report = dict(kind='SAME_PACKET_CPU_TONE_COMPOSITING_NO_APPLY', checkpoint_sha256=source['checkpoint_sha256'],
        source_audit_sha256=sha(a.source_audit), cache_sha256=sha(a.cache), script_sha256=sha(__file__),
        rays=len(gt), packed_samples=len(x['weights']), fully_invertible_rays=int(complete.sum()),
        invertible_channels=int(valid.sum()), saturated_sample_channels=int((x['rgb_samples']==1).sum()),
        positive_weight_saturated_samples=int(np.any(x['rgb_samples']==1,1)[x['weights']>0].sum()),
        saturated_weight_channel_max=r['saturated_weight'].max(0).tolist(),
        native_reconstruction_max_error=float(np.max(abs(r['native']-x['native_RGB']))),
        valid_channels=dict(native=scores(r['native'],gt,valid), physical=scores(r['physical'],gt,valid)),
        complete_rays=dict(native=scores(r['native'],gt,np.repeat(complete[:,None],3,1)),
                           physical=scores(r['physical'],gt,np.repeat(complete[:,None],3,1))),
        physical_minus_native_quantiles=dict(zip(['min','median','p90','p99','max'],np.percentile(delta[valid],[0,50,90,99,100]).tolist())),
        improved_channels=int(((abs(r['physical']-gt)<abs(r['native']-gt))&valid).sum()),
        worsened_channels=int(((abs(r['physical']-gt)>abs(r['native']-gt))&valid).sum()),
        per_camera={}, rows=rows, optimizer_updates=0, GPU_calls=0, source_unchanged=True,
        limitations=['RGB==1 with any positive weight is noninvertible; no epsilon clamp or claimed full-packet metric',
                     'Saturated tiny-weight tails can carry arbitrarily large decoded radiance; excluded subsets are biased',
                     'Counterfactual interpretation of already-trained encoded samples, not physical-model retraining',
                     'TRAIN soft MVS candidates and 5x5 source footprints; no inference about held-out KA from this subset'])
    for camera in np.unique(x['camera_names']):
        mask = valid & (x['camera_names']==camera)[:,None]
        report['per_camera'][str(camera)] = dict(native=scores(r['native'],gt,mask),physical=scores(r['physical'],gt,mask))
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output/'replay.npz', **r, GT_5x5=gt, camera_names=x['camera_names'],candidate_ids=x['candidate_ids'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','per_camera']},indent=2))


if __name__ == '__main__':
    main()
