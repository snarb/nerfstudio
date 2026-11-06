"""CPU-only camera feasibility audit for supplemental 4K EVAL; never fits or renders."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    diagnostic = json.loads(a.diagnostic.read_text())
    native = json.loads(a.native.read_text())
    assert diagnostic['training_eligible'] is False
    lookup = {row['physical_camera']: row for row in native['frames']}
    transform = np.asarray(diagnostic['world_to_box'])
    rows = []
    for row in diagnostic['frames']:
        name = row['physical_camera']
        original = lookup[name]
        restored = np.linalg.inv(transform) @ np.asarray(row['transform_matrix'])
        error = float(np.max(np.abs(restored - np.asarray(original['transform_matrix']))))
        assert error < 1e-12
        ratios = {key: row[key] / original[key] for key in ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy']}
        assert all(abs(value - 2) < 1e-12 for value in ratios.values())
        assert all(row[key] == original[key] == 0 for key in ['k1', 'k2', 'p1', 'p2'])
        rows.append(dict(camera=name, floor_to_native_pose_max_abs=error, intrinsic_size_ratios=ratios,
                         RGB_sha256=sha(a.diagnostic.parent / row['file_path']),
                         valid_sha256=sha(a.diagnostic.parent / row['mask_path'])))
    output = dict(kind='FEASIBILITY_ONLY_NO_GPU_NO_TRAINING', diagnostic_sha256=sha(a.diagnostic),
                  native_descriptor_sha256=sha(a.native), cameras=rows,
                  primary_FHD_score_unchanged=True, training_eligible=False,
                  proposed_renderer='Deepcopy original native EVAL camera; scale intrinsics/size by two. Keep native world pose and field AABB. Do not feed floor-frame poses into field.',
                  pixel_center_note='4K centers sample different subpixel positions; continuous coordinates map by x_HD=(x_4K+.5)/2-.5, likewise y. No claim of identical discrete rays.',
                  visual_protocol='Actual 4K GT | native100k | HD additional-step | TRAIN4K same additional-step; fixed actor/hair/hand/foot crops doubled from FHD. No upsampled predictions, no 4K selection/stop metric.',
                  resources='Four times pixels; approximately four times render work. Chunked field queries can keep query memory fixed, but full ray/output buffers grow fourfold. Resource probe required before actual rendering.')
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
