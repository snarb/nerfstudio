"""Hash-bound, opt-in target hair overrides for existing GT pixel transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dec5_hair import confidence_override, composite_layers


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_override(manifest_path, base_depth, *, base_depth_path, mesh_manifest_path, image_name):
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    if (record.get('depth_convention') != 'OpenCV camera-z'
            or record.get('base_depth_sha256') != digest(base_depth_path)
            or record.get('mesh_depth_manifest_sha256') != digest(mesh_manifest_path)
            or record.get('target_image_name') != image_name):
        raise ValueError('Hair override has incompatible base depth, camera or normalization')
    path = (manifest_path.parent/record['archive']).resolve()
    if record.get('archive_sha256') != digest(path):
        raise ValueError('Hair archive hash mismatch')
    with np.load(path, allow_pickle=False) as source:
        arrays = {key:source[key] for key in source.files}
    if not {'front_depth', 'roi', 'confidence'}.issubset(arrays):
        raise ValueError('Incomplete hair depth override')
    shape = np.asarray(base_depth).shape
    if any(arrays[k].shape != shape for k in ('front_depth', 'roi', 'confidence')):
        raise ValueError('Hair override raster shape mismatch')
    confidence = arrays['confidence']
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('Invalid confidence')
    threshold = float(record.get('minimum_confidence', .75))
    if not .5 <= threshold <= 1:
        raise ValueError('Confidence threshold must be in [.5,1]')
    refined, changed = confidence_override(base_depth, arrays['front_depth'],
                                          confidence >= threshold, arrays['roi'])
    layer_keys = {'back_depth', 'alpha', 'foreground_premultiplied', 'background_rgb'}
    if layer_keys & arrays.keys() and not layer_keys.issubset(arrays):
        raise ValueError('Incomplete two-layer archive')
    if layer_keys.issubset(arrays) and record.get('color_space') != 'scene_linear':
        raise ValueError('Two-layer RGB must explicitly declare scene_linear color space')
    for key in ('back_depth', 'alpha'):
        if key in arrays and arrays[key].shape != shape:
            raise ValueError('Layer shape mismatch')
    for key in ('foreground_premultiplied', 'background_rgb'):
        if key in arrays and arrays[key].shape != (*shape, 3):
            raise ValueError('Layer RGB shape mismatch')
    arrays['gate'] = (confidence >= threshold) & arrays['roi'].astype(bool)
    return refined, arrays, dict(manifest_sha256=digest(manifest_path), changed_depth_pixels=int((changed & (refined != base_depth)).sum()),
                                 two_layers=layer_keys.issubset(arrays), minimum_confidence=threshold)


def apply_layers(rgb, arrays):
    if 'foreground_premultiplied' not in arrays:
        return np.asarray(rgb).copy(), np.zeros(np.asarray(rgb).shape[:2], bool)
    return composite_layers(rgb, arrays['foreground_premultiplied'], arrays['alpha'], arrays['background_rgb'],
                            arrays['front_depth'], arrays['back_depth'], arrays['gate'], arrays['roi'])
