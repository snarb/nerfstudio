"""Same-model occupancy recovery restricted to exactly empty primary rays.

The caller must derive the secondary support from this model's own current
source evidence, validate it inside unchanged admission, and restore model
buffers after secondary queries. This module does no training, image editing,
target-RGB matching, or cross-frame transfer.
"""
from __future__ import annotations

import numpy as np


def recover_geometry(primary, secondary, *, primary_binding, secondary_binding):
    for key in ['checkpoint_sha256', 'row']:
        if not primary_binding.get(key) or primary_binding[key] != secondary_binding.get(key):
            raise ValueError('Recovery requires exactly the same checkpoint and camera rays')
    required = {'rgb', 'alpha'}
    allowed = required | {'q10', 'q50', 'q90'}
    if not required <= primary.keys() or set(primary) != set(secondary) or not set(primary) <= allowed:
        raise ValueError('Matched RGB/alpha/optional depth-quantile geometry caches required')
    shape = np.asarray(primary['alpha']).shape
    if len(shape) != 2:
        raise ValueError('Two-dimensional target raster required')
    for values in [primary, secondary]:
        if values['rgb'].shape != (*shape, 3) or values['alpha'].shape != shape:
            raise ValueError('Mismatched ray raster')
        if not np.isfinite(values['rgb']).all() or not np.isfinite(values['alpha']).all():
            raise ValueError('Nonfinite RGB/alpha')
        # Float32 transmittance reductions can exceed one by a few ULPs
        # (observed 1.00000143 on the native 1024-sample render). Validate
        # this rounding tolerance without clipping or changing any values.
        if np.any((values['alpha'] < 0) | (values['alpha'] > 1 + 1e-5)):
            raise ValueError('Invalid accumulation')
        if any(values[k].shape != shape for k in set(values) - {'rgb'}):
            raise ValueError('Mismatched depth-quantile raster')
    recover = (primary['alpha'] == 0) & (secondary['alpha'] > 0)
    result = {}
    for key, original in primary.items():
        output = np.array(original, copy=True)
        output[recover] = secondary[key][recover]
        if not np.array_equal(output[~recover], original[~recover], equal_nan=True):
            raise AssertionError('An existing primary ray changed')
        result[key] = output
    return result, dict(kind='SAME_MODEL_EXACT_EMPTY_RAY_RECOVERY_V1',
        checkpoint_sha256=primary_binding['checkpoint_sha256'],
        primary_empty_rays=int((primary['alpha'] == 0).sum()), recovered_rays=int(recover.sum()),
        existing_nonempty_primary_rays_bitwise_unchanged=True,
        RGB_GT_or_actor_mask_used_for_recovery_decision=False, model_weights_changed=False,
        surface_depth_certified=False)
