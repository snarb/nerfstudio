"""Partition existing renderer weights by fitted height; never infer empty space."""
import numpy as np


def partition_weights(starts, ends, sigma, actual_weights, origin_z, direction_z, cuts,
                      transmittance=None):
    """Return [samples,bins] weights using normalized constant-density fractions.

    All vectors describe the same packed samples, with per-ray Z origin/direction
    already gathered by ray index. Inputs are copied to FP64; actual_weights are
    authoritative, including any approximation made by the original renderer.
    Bins are left-closed/right-open; an exactly parallel sample on a cut goes to
    the bin above it. A zero-length sample uses that same point convention.
    """
    vectors = [np.asarray(x, dtype=np.float64).reshape(-1) for x in
               [starts, ends, sigma, actual_weights, origin_z, direction_z]]
    t0, t1, density, weights, oz, dz = vectors
    if len({len(x) for x in vectors}) != 1 or any(not np.isfinite(x).all() for x in vectors):
        raise ValueError('Packed vectors must have equal length and finite values')
    if (t1 < t0).any() or (density < 0).any() or (weights < 0).any():
        raise ValueError('Negative length, density or original weight')
    cuts = np.asarray(cuts, dtype=np.float64).reshape(-1)
    if not np.isfinite(cuts).all() or (np.diff(cuts) <= 0).any():
        raise ValueError('Cuts must be finite and strictly increasing')
    length = t1 - t0
    with np.errstate(over='ignore', invalid='ignore'):
        tau = density * length
        z0 = oz + dz * t0
        delta_z = dz * length
    if not all(np.isfinite(x).all() for x in [tau, z0, delta_z]):
        raise ValueError('Nonfinite optical thickness or sample coordinates')
    edges = np.r_[-np.inf, cuts, np.inf]
    fractions = np.zeros((len(t0), len(edges) - 1), dtype=np.float64)
    parallel = delta_z == 0
    if parallel.any():
        fractions[np.flatnonzero(parallel), np.searchsorted(cuts, z0[parallel], side='right')] = 1.
    moving = ~parallel
    denominator = -np.expm1(-tau)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        left = np.where(delta_z > 0, low, high)
        right = np.where(delta_z > 0, high, low)
        u0, u1 = np.zeros_like(t0), np.zeros_like(t0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            np.divide(left - z0, delta_z, out=u0, where=moving)
            np.divide(right - z0, delta_z, out=u1, where=moving)
        u0 = np.clip(u0, 0., 1.)
        u1 = np.clip(u1, 0., 1.)
        portion = np.maximum(0., u1 - u0)
        fraction = portion.copy()  # Continuous zero-density limit.
        nonzero = moving & (denominator > 0)
        fraction[nonzero] = (np.exp(-tau[nonzero] * u0[nonzero]) *
                             -np.expm1(-tau[nonzero] * portion[nonzero]) / denominator[nonzero])
        fractions[moving, index] = fraction[moving]
    sums = fractions.sum(axis=1)
    if (sums <= 0).any() or not np.isfinite(fractions).all():
        raise ValueError('Failed to cover an original interval')
    if len(sums):
        fractions /= sums[:, None]
    parts = weights[:, None] * fractions
    # Correct only FP64 roundoff, never substitute analytic opacity for weights.
    if len(weights):
        major = fractions.argmax(axis=1)
        parts[np.arange(len(weights)), major] += weights - parts.sum(axis=1)
    reconstruction = parts.sum(axis=1) - weights
    tolerance = 8 * np.spacing(weights)
    if (np.abs(reconstruction) > tolerance).any() or (parts < 0).any():
        raise ValueError('Original weight conservation failed')
    diagnostic = dict(samples=len(weights), cuts=cuts.tolist(),
        original_weight_sum=float(weights.sum()), partition_weight_sum=float(parts.sum()),
        max_sample_weight_reconstruction_error=float(np.abs(reconstruction).max()) if len(weights) else 0.,
        max_fraction_sum_error_before_normalization=float(np.abs(sums - 1).max()) if len(sums) else 0.,
        parallel_or_zero_length_samples=int(parallel.sum()), zero_density_nonzero_weight_samples=int(((density == 0) & (weights > 0)).sum()),
        zero_length_nonzero_weight_samples=int(((length == 0) & (weights > 0)).sum()),
        samples_crossing_cut=int((np.count_nonzero(fractions, axis=1) > 1).sum()),
        original_weight_in_crossing_samples=float(weights[np.count_nonzero(fractions, axis=1) > 1].sum()),
        per_cut_original_weight_in_straddling_intervals=[dict(cut=float(cut), weight=float(weights[(np.minimum(z0, z0 + delta_z) < cut) & (np.maximum(z0, z0 + delta_z) > cut)].sum())) for cut in cuts],
        original_weights_authoritative=True, density_or_transmittance_changed=False)
    if transmittance is not None:
        trans = np.asarray(transmittance, dtype=np.float64).reshape(-1)
        if trans.shape != weights.shape or not np.isfinite(trans).all() or (trans < 0).any():
            raise ValueError('Invalid reference incoming transmittance')
        discrepancy = trans * denominator - weights
        diagnostic['analytic_minus_original_weight'] = dict(
            sum=float(discrepancy.sum()), absolute_sum=float(np.abs(discrepancy).sum()),
            max_abs=float(np.abs(discrepancy).max()) if len(weights) else 0.)
    return parts, diagnostic


def accumulate_parts(parts, rgb, ray_indices, num_rays):
    """Reduce one bounded packed chunk; caller streams chunks into image maps."""
    parts = np.asarray(parts, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    indices = np.asarray(ray_indices)
    if parts.ndim != 2 or rgb.shape != (len(parts), 3) or indices.shape != (len(parts),):
        raise ValueError('Packed RGB/weight/index shapes disagree')
    if indices.dtype.kind not in 'iu' or (indices < 0).any() or (indices >= num_rays).any():
        raise ValueError('Invalid packed ray index')
    if not np.isfinite(parts).all() or not np.isfinite(rgb).all():
        raise ValueError('Nonfinite contributions')
    mass = np.zeros((num_rays, parts.shape[1]), dtype=np.float64)
    color = np.zeros((num_rays, parts.shape[1], 3), dtype=np.float64)
    for index in range(parts.shape[1]):
        mass[:, index] = np.bincount(indices, weights=parts[:, index], minlength=num_rays)
        for channel in range(3):
            color[:, index, channel] = np.bincount(indices, weights=parts[:, index] * rgb[:, channel], minlength=num_rays)
    return mass, color
