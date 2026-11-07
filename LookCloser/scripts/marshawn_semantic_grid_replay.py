"""Helpers for an isolated frozen-field occupancy counterfactual.

No helper installs a density selector or reconstructs core from EMA. Changing
traversal may change ARM quadrature/phase/cap; identical RGB is not expected.
"""
from contextlib import contextmanager
import hashlib

import numpy as np


@contextmanager
def temporary_binaries(pipeline, replacement, digest, source_assert):
    """Only replace active binaries; restore them before full source assertion."""
    import torch
    model = pipeline.model
    if model.training or model.config.occupancy_eval_dilation_radius != 0 or getattr(model, '_eval_occupancy_backup', None) is not None:
        raise ValueError('Frozen eval radius0 with no evaluation backup required')
    original_tensor = model.occupancy_grid.binaries
    original = original_tensor.detach().clone()
    if replacement.dtype != torch.bool or replacement.shape != original.shape or replacement.device != original.device:
        raise ValueError('Exact active-grid shape, bool dtype and device required')
    if bool((replacement & ~model.semantic_admission.reshape(replacement.shape)).any()):
        raise ValueError('Candidate exceeds unchanged original admission')
    state = pipeline.state_dict()
    keys = [k for k in state if k.endswith('occupancy_grid.binaries')]
    expected_keys = {'_model.occupancy_grid.binaries', '_model.adaptive_sampler.occupancy_grid.binaries'}
    if set(keys) != expected_keys:
        raise ValueError('Exact two registered source active-grid alias keys required')
    def storage_identity(tensor):
        return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.untyped_storage().data_ptr())
    expected_storage = storage_identity(original_tensor)
    def assert_aliases(current):
        if model.adaptive_sampler.occupancy_grid is not model.occupancy_grid:
            raise ValueError('Native sampler grid must be the same registered grid object')
        if any(storage_identity(current[k]) != expected_storage for k in expected_keys):
            raise ValueError('Registered active-grid aliases must share exact source storage/layout')
        if any(not torch.equal(current[k], original_tensor) for k in expected_keys):
            raise ValueError('Registered active-grid alias values differ')
    assert_aliases(state)
    before = digest({k: v for k, v in state.items() if k not in expected_keys})
    source_assert()
    def assert_inside():
        if model.occupancy_grid.binaries is not original_tensor or not torch.equal(model.occupancy_grid.binaries, replacement):
            raise ValueError('Counterfactual binaries unexpectedly rebuilt/changed')
        if model.config.occupancy_eval_dilation_radius != 0 or getattr(model, '_eval_occupancy_backup', None) is not None:
            raise ValueError('Unexpected evaluation dilation/backup')
        current = pipeline.state_dict()
        assert_aliases(current)
        if digest({k: v for k, v in current.items() if k not in expected_keys}) != before:
            raise ValueError('Non-binary pipeline state changed')
        return dict(registered_keys=sorted(expected_keys), same_grid_object=True, same_storage=True,
            storage_pointer=original_tensor.untyped_storage().data_ptr(), storage_offset=original_tensor.storage_offset(),
            shape=list(original_tensor.shape), stride=list(original_tensor.stride()), only_exact_alias_keys_excluded_from_state_digest=True)
    try:
        original_tensor.copy_(replacement)
        assert_inside()
        yield assert_inside
        assert_inside()
    finally:
        model.occupancy_grid.binaries = original_tensor
        original_tensor.copy_(original)
        # Never run the original full-state assertion while a candidate grid
        # remains installed, even after a proof/render exception.
        source_assert()


def packed_selected(row, probability, selector, membership_fn, nerfacc, stable_weights, tolerance=2e-5):
    """Same P3 physical field: original sigma times the fixed head selector."""
    import torch
    if selector != 'hard_theta01':
        raise ValueError('This one-factor grid plan fixes physical theta.1')
    membership = membership_fn(probability)[selector]
    sigma = row['sigma']*membership
    ids = row['ray_indices'].long(); count = len(row['num_samples_per_ray'])
    weights = nerfacc.render_weight_from_density(t_starts=row['starts'], t_ends=row['ends'],
        sigmas=sigma, packed_info=nerfacc.pack_info(ids, count))[0]
    reference = stable_weights(row['starts'], row['ends'], sigma, ids, count)
    error = float((weights.double()-reference).abs().max()) if len(weights) else 0.
    if error > tolerance:
        raise ValueError('Selected-density independent reference failed at unchanged2e-5: '+str(error))
    rgb = nerfacc.accumulate_along_rays(weights, values=row['rgb'], ray_indices=ids, n_rays=count)
    alpha = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ids, n_rays=count).reshape(-1)
    if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all() or not torch.isfinite(sigma).all():
        raise ValueError('Nonfinite physical composition')
    return dict(membership=membership, selected_sigma=sigma, selected_weight=weights,
        selected_RGBA=torch.cat((rgb, alpha[:, None]), -1), independent_weight_error=error)


def grid_membership(points, grid, boxes):
    """Actual XYZ cell membership (union of explicitly supplied cascades)."""
    import torch
    boxes = boxes.reshape(-1, 2, 3).to(device=points.device, dtype=points.dtype)
    if len(boxes) != len(grid):
        raise ValueError('Explicit matching cascade bounds required')
    result = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    resolution = torch.tensor(grid.shape[1:], device=points.device)
    for level, box in enumerate(boxes):
        unit = (points-box[0])/(box[1]-box[0])
        inside = ((unit >= 0) & (unit <= 1)).all(-1)
        ids = torch.minimum(torch.maximum(torch.floor(unit*resolution).long(), torch.zeros_like(resolution)), resolution-1)
        result |= inside & grid[level, ids[:, 0], ids[:, 1], ids[:, 2]]
    return result


def adversarial_positions(bounds, native_bounds, witnesses, seed=20261107):
    """Private NumPy draws only: fixed witnesses, knot/cell planes and +/-ULP."""
    b = np.asarray(bounds, np.float32); nb = np.asarray(native_bounds, np.float32).reshape(-1, 2, 3)
    rng = np.random.default_rng(seed)
    rows = [np.asarray(witnesses, np.float32), rng.uniform(b[0], b[1], (8192, 3)).astype(np.float32)]
    for frame, divisions in [(b, 127), *[(x, 128) for x in nb]]:
        for axis in range(3):
            pts = rng.uniform(frame[0], frame[1], (1024, 3)).astype(np.float32)
            indices = np.arange(1024) % (divisions+1)
            pts[:, axis] = frame[0, axis]+indices/divisions*(frame[1, axis]-frame[0, axis])
            rows.append(pts)
            for target in (np.float32(-np.inf), np.float32(np.inf)):
                shift = pts.copy(); shift[:, axis] = np.nextafter(shift[:, axis], target); rows.append(shift)
    points = np.concatenate(rows)
    return points, dict(seed=seed, count=len(points), observations=len(witnesses),
        sha256=hashlib.sha256(points.tobytes()).hexdigest(), private_numpy_rng=True,
        coverage='Witnesses,8192uniform points,semantic127/native128 planes and +/-one FP32ULP; sampled proof only.')


def missed_reference_mass(dense, native, masks):
    """Finite dense-reference mass outside grid/actual intervals; not empty GT."""
    starts = np.asarray(native['starts']).reshape(-1); ends = np.asarray(native['ends']).reshape(-1)
    mid = (np.asarray(dense['starts'])+np.asarray(dense['ends']))*.5
    weights = np.asarray(dense['selected_weight'], dtype=np.float64)
    covered = np.zeros(len(mid), bool)
    if len(starts):
        index = np.searchsorted(starts, mid, side='right')-1
        valid = index >= 0
        covered[valid] = mid[valid] <= ends[index[valid]]
    return dict(dense_opacity=float(weights.sum()), native_interval_uncovered_selected_mass=float(weights[~covered].sum()),
        selected_mass_beyond_native_last_end=float(weights[mid > ends[-1]].sum()) if len(ends) else float(weights.sum()),
        rejected_grid_selected_mass={k: float(weights[~np.asarray(v, bool)].sum()) for k, v in masks.items()},
        finite_midpoint_quadrature=True, physical_free_space_claim=False)


def numeric_rgba_parity(actual, expected):
    """Existing saved-code tolerances, plus the unchanged2e-5 float guard."""
    if actual.shape != expected.shape or actual.shape[-1] != 4 or not np.isfinite(actual).all():
        raise ValueError('Finite matching RGBA required')
    report = dict(float_max_error=float(np.max(np.abs(actual-expected))))
    for name, columns, scale, fraction in [('RGB', slice(0, 3), 255, 1e-5), ('alpha', 3, 65535, .001)]:
        a = np.rint(np.clip(actual[..., columns], 0, 1)*scale).astype(np.int64)
        b = np.rint(np.clip(expected[..., columns], 0, 1)*scale).astype(np.int64)
        report[name] = dict(max_code_error=int(np.abs(a-b).max()), changed_fraction=float(np.mean(a != b)))
        if report[name]['max_code_error'] > 1 or report[name]['changed_fraction'] > fraction:
            raise ValueError('Existing P3 '+name+' numeric parity failed: '+str(report))
    if report['float_max_error'] > 2e-5:
        raise ValueError('Existing P3 float2e-5 parity failed: '+str(report))
    return report
