"""CPU-only geometric-buffer counterfactual from already observed field samples.

No model, CUDA, optimizer, occupancy update, or new source annotation. Margins
are L-infinity distances to the ORIGINAL source-cell union, not grid dilation.
Numerical epsilon and geometric margin are separate, explicit quantities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def admitted(points, mask, bounds, *, margin, epsilon=2e-7):
    points, bounds = np.asarray(points, dtype=np.float64), np.asarray(bounds, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 3 or bounds.shape != (2, 3):
        raise ValueError('Boolean XYZ source cells and 2x3 bounds required')
    if points.shape[-1] != 3 or not np.isfinite(points).all() or not np.isfinite(bounds).all():
        raise ValueError('Finite XYZ required')
    pitch = (bounds[1]-bounds[0])/mask.shape
    if not np.all(pitch > 0) or not np.isfinite([margin, epsilon]).all() or min(margin, epsilon) < 0:
        raise ValueError('Nonnegative explicit distances and increasing bounds required')
    if 2*(margin+epsilon) >= pitch.min():
        raise ValueError('Buffer must remain below half the minimum source pitch')
    shape = points.shape[:-1]
    p = points.reshape(-1, 3)
    # UNKNOWN outside the source domain is inherited, and depends only on eps.
    outside = ((p < bounds[0]-epsilon) | (p > bounds[1]+epsilon)).any(-1)
    radius = margin+epsilon
    first = np.floor(np.nextafter((p-bounds[0]-radius)/pitch, -np.inf)).astype(np.int64)
    last = np.floor(np.nextafter((p-bounds[0]+radius)/pitch, np.inf)).astype(np.int64)
    first = np.clip(first, 0, np.asarray(mask.shape)-1)
    last = np.clip(last, 0, np.asarray(mask.shape)-1)
    result = outside.copy()
    for x in (first[:, 0], last[:, 0]):
        for y in (first[:, 1], last[:, 1]):
            for z in (first[:, 2], last[:, 2]):
                result |= mask[x, y, z]
    return result.reshape(shape)


def integrate(start, end, sigma, rgb, keep):
    start, end, sigma = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (start, end, sigma)]
    rgb, keep = np.asarray(rgb, dtype=np.float64), np.asarray(keep)
    if not (start.shape == end.shape == sigma.shape == keep.shape) or rgb.shape != (len(start), 3) or keep.dtype != bool:
        raise ValueError('Matching scalar samples and RGB required')
    if not all(np.isfinite(x).all() for x in (start, end, sigma, rgb)) or np.any(end < start) or np.any(sigma < 0):
        raise ValueError('Finite nonnegative intervals/density required')
    tau = (end-start)*np.where(keep, sigma, 0.)
    trans = np.exp(-np.r_[0., np.cumsum(tau[:-1])])
    weights = -np.expm1(-tau)*trans
    return weights, (weights[:, None]*rgb).sum(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, output):
    root, output = Path(root), Path(output)
    source = root/'source_grid.npz'
    z = np.load(source)
    mask, bounds = z['semantic_gate_pad1'], z['aabb']
    margins = [0., .0005, .001, .002]
    model = root/'quarter_distortion_raw126316'
    records, hashes = [], {str(source): sha(source)}
    witness_points = []
    for group in ['clothing103', 'DC12', 'shoe6', 'foot27', 'artifact12']:
        directory = model/'native_A'/f'{group}_trace/rays'
        audit_path = directory/'audit.json'
        hashes[str(audit_path)] = sha(audit_path)
        rows = json.loads(audit_path.read_text())['rows']
        for index, row in enumerate(rows):
            q = row['pixel']
            if 'witness_xyz' in q:
                witness_points.append(q['witness_xyz'])
            for profile in ['native', 'uniform2x']:
                path = directory/row['profiles'][profile]['trace']
                sample = np.load(path)
                hashes[str(path)] = sha(path)
                w_A, rgb_A = integrate(sample['t_start'], sample['t_end'], sample['sigma'], sample['rgb'], np.ones(len(sample['sigma']), bool))
                # Analytic consistency only. Never substitute it for actual
                # renderer weights in a claim about native rendering parity.
                np.testing.assert_allclose(w_A, sample['weights'], atol=1e-12, rtol=1e-10)
                record = {'group': group, 'index': index, 'query': q, 'profile': profile,
                          'baseline_analytic_alpha': float(w_A.sum()), 'baseline_analytic_rgb': rgb_A.tolist(),
                          'sampled_native_actual_weights_max_error': float(np.max(np.abs(w_A-sample['actual_weights']), initial=0.)),
                          'margins': []}
                prev = np.zeros(len(w_A), bool)
                for margin in margins:
                    keep = admitted(sample['xyz'], mask, bounds, margin=margin)
                    assert np.all(keep[prev])
                    prev = keep
                    weights, rgb = integrate(sample['t_start'], sample['t_end'], sample['sigma'], sample['rgb'], keep)
                    r = {'margin': margin, 'admitted_samples': int(keep.sum()),
                         'removed_original_analytic_weight': float(w_A[~keep].sum()),
                         'alpha': float(weights.sum()), 'rgb': rgb.tolist(),
                         'RGB_max_abs_delta_A': float(np.max(np.abs(rgb-rgb_A)))}
                    if 'query_witness_depth' in q:
                        mid = (sample['t_start'].astype(float)+sample['t_end'])/2
                        r['band_mass_pm02'] = float(weights[np.abs(mid-q['query_witness_depth']) <= .02].sum())
                    if margin == 0 and profile == 'uniform2x':
                        bpath = model/'source_point_B'/f'{group}_trace/rays'/path.name
                        b = np.load(bpath)
                        hashes[str(bpath)] = sha(bpath)
                        np.testing.assert_array_equal(np.where(keep, sample['sigma'], 0), b['sigma'])
                        np.testing.assert_allclose(weights, b['weights'], atol=1e-12, rtol=1e-10)
                    record['margins'].append(r)
                records.append(record)
    points = np.asarray(witness_points)
    retained = {str(m): int(admitted(points, mask, bounds, margin=m).sum()) for m in margins}
    assert all(n == len(points) for n in retained.values())
    result = {'kind': 'CPU_SAVED_SAMPLE_DENSITY_INTEGRAL_COUNTERFACTUAL', 'epsilon': 2e-7,
              'geometric_margins_Linf': margins, 'source_pitch': ((bounds[1]-bounds[0])/mask.shape).tolist(),
              'query_count': len(records)//2, 'witness_occurrences': len(points), 'witness_retained': retained,
              'source_hashes': hashes, 'helper_sha256': sha(__file__), 'records': records,
              'limitation': 'No new field evaluations or native resampling. Dense integrals and fixed native-sample counterfactuals are not native performance or full-image metrics. Original source support is not physical occupancy truth.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': True, 'rays': len(records)//2, 'witness_retained': retained, 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
