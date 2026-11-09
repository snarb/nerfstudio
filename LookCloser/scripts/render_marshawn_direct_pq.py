"""Render actual code-space model rays, then fixed vendor EOTF→matched SDR.

No learned exposure, GT fitting, compositing, denoising or occupancy change.
Original model-native PQ codes remain in the preview helper's NPZ buffers.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import render_marshawn_previews as preview

ROOT = Path('/fsx/tmp/marshawn_001660_hd')
VENDOR = Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-width', type=int, default=640)
    p.add_argument('--chunk-size', type=int, default=512)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('vendor_pq', VENDOR)
    vendor = importlib.util.module_from_spec(spec); spec.loader.exec_module(vendor)
    audit_path = ROOT/'report/canonical_colour_independent/audit.json'
    audit = json.loads(audit_path.read_text())
    if preview.sha(VENDOR) != audit['source']['vendor_sha256']:
        raise ValueError('Vendor color source changed')
    matrix = np.array(audit['matrices']['AP1_D60_to_Rec709_D65_Bradford'])
    gain = audit['source_bounds']['canonical_gain_to_nits']
    def display(value, transform):
        assert transform == 'PQ_AP1_D60_EOTF_fixed_gain4_Reinhard_sRGB'
        linear = np.maximum((vendor.pq_decode(value) @ matrix.T)*(4/gain), 0)
        return preview_original_display(linear, 'linear-reinhard-srgb')
    preview_original_display = preview.display_rgb
    preview.display_rgb = display
    args.code_root = ROOT/'code_native'
    args.data = None; args.camera = None; args.bbox = None
    args.skip_strict_bounds = True
    args.display_transform = 'PQ_AP1_D60_EOTF_fixed_gain4_Reinhard_sRGB'
    for directory in [ROOT/'code_sfm/LookCloser/scripts', ROOT/'code_native/LookCloser/scripts',
                      ROOT/'code_native', ROOT/'final_pair/code', args.config.parent/'code']:
        sys.path.insert(0, str(directory))
    preview.run(args)
    receipt_path = args.output/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['display_contract'] = dict(vendor_pq_sha256=preview.sha(VENDOR),
        independent_audit_sha256=preview.sha(audit_path), AP1_white='D60',
        code_to_nits='ST2084 absolute EOTF', matched_display_gain_per_nit=4/gain,
        physical_compositing=False, gt_used_to_fit_display=False,
        source_script_sha256=preview.sha(__file__))
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__': main()
