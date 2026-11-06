"""CPU postprocessor of existing immutable native ray traces, preserving actual weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from marshawn_signed_height_bins import partition_weights


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    report = json.loads(a.audit.read_text())
    assert report['frozen_state_unchanged']
    ready = json.loads(a.ready.read_text())
    if report['transforms_sha256'] != ready['transforms_sha256']:
        raise ValueError('Trace and accepted floor-frame data differ')
    if any(np.max(np.abs(np.asarray(bounds) - np.asarray(ready['aabb']))) > 1e-6 for bounds in report['live_bounds'].values()):
        raise ValueError('This postprocessor requires the accepted floor-frame LC field')
    if report['normalization']['scale'] != 1. or not np.array_equal(report['normalization']['transform'], np.eye(4)[:3]):
        raise ValueError('Unexpected additional parser coordinate normalization')
    a.output.mkdir(parents=True, exist_ok=False)
    cuts = [-.02, -.015, -.01, -.004636666777014908, 0., .004636666777014908, .01, .015, .02]
    records = []
    for index, row in enumerate(report['rows']):
        path = a.audit.parent / row['profiles']['native']['trace']
        with np.load(path, allow_pickle=False) as profile:
            starts, ends, sigma, weights = [profile[k].reshape(-1) for k in ['t_start', 't_end', 'sigma', 'actual_weights']]
            rgb = profile['rgb'].reshape(-1, 3)
            parts, numerical = partition_weights(starts, ends, sigma, weights,
                np.full(len(weights), row['origin'][2]), np.full(len(weights), row['direction'][2]), cuts,
                transmittance=profile['transmittance'])
            if len(weights):
                expected_z = row['origin'][2] + row['direction'][2] * (starts.astype(float) + ends.astype(float)) / 2
                xyz_error = float(np.abs(profile['xyz'][:, 2] - expected_z).max())
                if xyz_error > 2e-6:
                    raise ValueError('Trace XYZ is not in the stated floor ray frame')
                order = np.argsort(expected_z)
                cdf = np.cumsum(weights[order].astype(float))
                quantiles = [float(expected_z[order[min(np.searchsorted(cdf, q * cdf[-1]), len(order) - 1)]]) for q in [.1,.5,.9]] if cdf[-1] > 0 else None
            else:
                xyz_error, quantiles = 0., None
        mass = parts.sum(axis=0)
        color = parts.T @ rgb.astype(float)
        original_rgb = weights.astype(float) @ rgb.astype(float)
        rendered = np.asarray(row['profiles']['native']['model_rgb'], dtype=float)
        # This residual is not assumed to be background: it also includes any
        # original renderer/clamp/FP32 reduction discrepancy, retained verbatim.
        residual = rendered - original_rgb
        record = dict(pixel=row['pixel'], profile=str(path), profile_sha256=sha(path),
            mass_per_bin=mass.tolist(), RGB_contribution_per_bin=color.tolist(),
            original_native_weight_sum=float(weights.astype(float).sum()),
            original_native_foreground_RGB=original_rgb.tolist(), model_RGB=rendered.tolist(),
            original_renderer_minus_foreground_RGB=residual.tolist(),
            bin_RGB_sum_max_abs_error=float(np.abs(color.sum(axis=0) - original_rgb).max()),
            model_RGB_reconstruction_error=float(np.abs(color.sum(axis=0) + residual - rendered).max()),
            XYZ_midpoint_chain_error=xyz_error,
            signed_Z_weighted_midpoint_quantiles_10_50_90=quantiles,
            quantile_limitation='Midpoint diagnostic, unlike interval-split bin masses; not physical depth truth.',
            numerical=numerical)
        if record['bin_RGB_sum_max_abs_error'] > 1e-12:
            raise ValueError('Binned native RGB contribution did not conserve original weights')
        records.append(record)
    output = dict(kind='ACTUAL_NATIVE_WEIGHT_SIGNED_HEIGHT_PARTITION', audit=str(a.audit), audit_sha256=sha(a.audit),
        checkpoint_sha256=report['checkpoint_sha256'], ready_sha256=sha(a.ready), cuts=cuts, rows=records,
        helper_sha256=sha(Path(__file__)), bin_math_sha256=sha(Path(__file__).with_name('marshawn_signed_height_bins.py')),
        reference_transmittance_note='NPZ transmittance is the existing FP64 reference integration; actual_weights are the renderer output and remain authoritative.',
        masks_or_density_or_grid_changed=False, no_GPU=True,
        coordinate_note='Input must be the existing accepted floor-frame LC trace; original-world nativeNGP traces require an explicit transform first.')
    (a.output / 'partition.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(rays=len(records), max_RGB_error=max((r['bin_RGB_sum_max_abs_error'] for r in records), default=0))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--ready', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
