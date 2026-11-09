#!/usr/bin/env python3
"""Export sealed offline arrays as documented camera-z EXRs without refitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

from run_dec5_hair_campaign import atomic_json, sha


def write_exr(path, array):
    array = np.asarray(array, np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite export: '+str(path))
    encoded = array[..., ::-1] if array.ndim == 3 else array
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError('EXR write failed')
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, encoded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipt = json.loads((args.prediction/'complete.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in receipt['records']:
        name = row['camera']
        source = args.prediction/(name+'.npz')
        if sha(source) != row['npz_sha256']:
            raise ValueError('Prediction changed after sealing')
        if sha(args.prediction/(name+'.png')) != row['png_sha256']:
            raise ValueError('Display RGB changed after sealing')
        arrays = dict(np.load(source, allow_pickle=False))
        base = np.load(args.baseline/(name+'.npz'), allow_pickle=False)
        folder = args.output/name
        folder.mkdir()
        fields = dict(base_depth=base['front_depth'], hair_front_depth=arrays['front_depth'],
                      hair_confidence=arrays['support'].astype(np.float32)/8,
                      hair_semantic_probability=arrays['hair_probability'],
                      source_id=arrays['source_id'], rgb_linear=arrays['rgb'],
                      edit_roi=arrays['roi'], independent_reprojection_error=arrays['reprojection_error'])
        if 'back_depth' in arrays:
            fields['background_candidate_depth'] = arrays['back_depth']
        if 'layer_alpha' in arrays:
            fields.update(hair_alpha=arrays['layer_alpha'], hair_back_depth=arrays['layer_back_depth'],
                          hair_foreground_premultiplied=arrays['foreground_premultiplied'],
                          layer_gate=arrays['layer_gate'])
        for key, value in fields.items():
            write_exr(folder/(key+'.exr'), value)
        shutil.copy2(args.prediction/(name+'.png'), folder/'rgb_display.png')
        metadata = dict(camera=name, depth_convention='OpenCV camera-z', units='calibration units',
            metric_scale_certified=False, resolution=[arrays['front_depth'].shape[1], arrays['front_depth'].shape[0]],
            color_space='scene-linear EXR; fixed gain/Reinhard/sRGB display PNG',
            confidence_definition='original measured depth-valid witnesses / fixed 8 sources; not calibrated probability',
            alpha_available='layer_alpha' in arrays,
            semantic_probability_is_not_alpha=True, background_candidate_is_not_certified_layer=True,
            missing_source_id=-1, neural_fallback_retains_missing_geometry_evidence=True,
            source_id_to_train_camera=receipt['train_sources'],
            source_selection=receipt['source_selection'],
            fallback_receipt_sha256=receipt.get('fallback_receipt_sha256'),
            prediction_receipt_sha256=sha(args.prediction/'complete.json'),
            selection_sha256=sha(args.selection), source_archive_sha256=sha(source),
            files={f.name:sha(f) for f in sorted(folder.iterdir())})
        atomic_json(folder/'metadata.json', metadata)
        records.append(dict(camera=name, metadata_sha256=sha(folder/'metadata.json')))
    atomic_json(args.output/'complete.json', dict(records=records, selection_sha256=sha(args.selection),
                implementation_sha256=sha(__file__), exr_roundtrip_exact=True))


if __name__ == '__main__':
    main()
