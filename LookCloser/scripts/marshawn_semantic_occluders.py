"""Conservative source occlusion handling for a separate semantic gate.

A known non-actor surface can hide an actor. Its pixels must not become
unbounded free-space evidence when no independent surface depth is available.
This module changes neither actor labels nor the original source arrays.
"""
import numpy as np


def merge_occluder_unknown(support, unknown, valid, occluders):
    """Union qualified opaque-source regions into UNKNOWN without depth claims.

    `occluders` maps source identifiers to boolean masks in the exact same
    calibrated raster. Masks may overlap actor support at uncertain contacts;
    that overlap still prevents a clear-background vote. Support is preserved.
    """
    arrays = [np.asarray(a) for a in (support, unknown, valid)]
    if any(a.dtype != np.bool_ or a.ndim != 2 for a in arrays):
        raise ValueError('Expected boolean calibrated raster masks')
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Source raster mismatch')
    support, unknown, valid = arrays
    merged = unknown.copy()
    sources = {}
    for name, mask in occluders.items():
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != unknown.shape:
            raise ValueError('Occluder raster mismatch: ' + str(name))
        sources[str(name)] = dict(pixels=int(mask.sum()),
                                 overlaps_support=int((mask & support).sum()))
        merged |= mask
    old_clear = valid & ~support & ~unknown
    new_clear = valid & ~support & ~merged
    if np.any(new_clear & ~old_clear):
        raise AssertionError('Adding UNKNOWN must not create clear evidence')
    return support.copy(), merged, valid.copy(), dict(
        sources=sources, added_unknown_pixels=int((merged & ~unknown).sum()),
        removed_clear_pixels=int((old_clear & ~new_clear).sum()),
        actor_support_unchanged=True, no_depth_based_free_space_claim=True)
