"""Bounded disparity-offset diagnosis with spatially held-out depth anchors."""
import numpy as np


def spatial_folds(uv, block=128, margin=8):
    uv = np.asarray(uv)
    cell = np.floor(uv / block).astype(int)
    local = uv - cell * block
    inside = ((local >= margin) & (local < block - margin)).all(1)
    return (cell[:, 0] + 2 * cell[:, 1]) % 4, inside


def error_stats(predicted, measured):
    error = np.abs(np.asarray(predicted) - measured)
    if not len(error): return dict(count=0, median=None, p90=None)
    return dict(count=len(error), median=float(np.median(error)), p90=float(np.percentile(error, 90)))


def fit_and_validate(uv, disparity, expected, fb, offset, min_anchors=100, bound=4.):
    disparity, expected = np.asarray(disparity), np.asarray(expected)
    if not np.isfinite(disparity).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite anchors')
    fold, interior = spatial_folds(uv)
    records = []; corrections = []; before = []; after = []; target = []
    for index in range(4):
        train = interior & (fold != index); test = interior & (fold == index)
        if min(train.sum(), test.sum()) < min_anchors:
            return dict(status='insufficient_spatial_anchors', folds=records, correction=None)
        correction = float(np.median(expected[train] - disparity[train]))
        if (disparity[test] + correction - offset <= 0).any():
            raise ValueError('Invalid corrected metric disparity')
        measured_z = fb / (expected[test] - offset)
        original_z = fb / (disparity[test] - offset)
        corrected_z = fb / (disparity[test] + correction - offset)
        records.append(dict(fold=index, train=int(train.sum()), test=int(test.sum()),
            correction=correction, original_depth_error=error_stats(original_z, measured_z),
            corrected_depth_error=error_stats(corrected_z, measured_z)))
        corrections.append(correction); before.extend(original_z); after.extend(corrected_z); target.extend(measured_z)
    original = error_stats(before, np.array(target)); corrected = error_stats(after, np.array(target))
    correction = float(np.median((expected - disparity)[interior]))
    stable = np.ptp(corrections) <= 1.5 and max(abs(np.array(corrections))) <= bound and abs(correction) <= bound
    improves = corrected['median'] <= original['median'] * .9 and corrected['p90'] <= original['p90'] * 1.05
    return dict(status='passes_anchor_bias_gate' if stable and improves else 'reject_bias_only_model',
        correction=correction, folds=records, original_depth_error=original,
        corrected_depth_error=corrected, fold_correction_range=float(np.ptp(corrections)),
        stable_bounded=bool(stable), improves_heldout_anchors=bool(improves),
        note='Passing anchors is not permission to fill missing surfaces or proof of independent ground truth')
