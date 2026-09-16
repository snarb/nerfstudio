"""Post-hoc object ownership of the corrected blue-wedge core, not mesh carving.

Uses already reviewed train-only masks. Near depth alone does not establish
object identity or visibility. The diagnostic polygon never enters prediction.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from study_multiview_face_prior import read, save, sha
from study_query_support_quorum import ROOT, FRAME
from study_lipstick_instance_masks import ROOT as MASK_ROOT, portrait_xy
from audit_lipstick_instance_witnesses import signed_mask_distance
from study_confidence_depth_prior import project_integer

HAND_ROOT = Path('/mnt/data/dec5_lipstick_hand_mask_000995')


def main():
    root = ROOT / FRAME / 'blue_wedge_support'
    out = root / 'semantic_witnesses'
    assert not out.exists()
    result = read(root / 'result.json')
    for name, digest in result['outputs'].items():
        assert sha(root / name) == digest
    evidence = np.load(root / 'evidence.npz')
    points = evidence['points']
    request = read(MASK_ROOT / 'request.json')
    source = read(ROOT / FRAME / 'rgb/K004_B005_1210DS/frames' / FRAME / 'result.json')
    names = source['source_cameras']
    bindings = {str(root / 'result.json'): sha(root / 'result.json'),
                str(root / 'evidence.npz'): sha(root / 'evidence.npz'),
                str(MASK_ROOT / 'request.json'): sha(MASK_ROOT / 'request.json'),
                str(Path(__file__)): sha(__file__)}
    selections = []
    for folder, run in [(MASK_ROOT, 'sam_v2'), (HAND_ROOT, 'sam_v1')]:
        review = read(folder / 'mask_review.json')
        assert review['status'] == 'usable_for_bounded_semantic_diagnostic'
        assert review['result_sha256'] == sha(folder / run / 'result.json')
        records = read(folder / run / 'result.json')
        for path in [folder / 'mask_review.json', folder / run / 'result.json']:
            bindings[str(path)] = sha(path)
        selections.append((folder / run, review, records))
    out.mkdir()
    stats = []
    for row in request['views']:
        name = row['camera']; ci = names.index(name)
        path = Path(row['image']); assert sha(path) == row['image_sha256']
        bindings[str(path)] = sha(path)
        im = Image.open(path).convert('RGB'); w, h = im.size
        uv, z = project_integer(row['camera_parameters'], points)
        xy = np.rint(portrait_xy(uv, 1920) - np.array(row['crop'][:2])).astype(int)
        inside = (z > 0) & (xy >= 0).all(1) & (xy < [w, h]).all(1)
        assert inside.all(), 'Core must remain inside saved source crop'
        distances = []
        for folder, review, records in selections:
            index = review['selected'][name]
            path = folder / name / f'mask_{index}.png'
            record = next(r for r in records['views'] if r['camera'] == name)
            assert record['outputs'][str(path.relative_to(folder))] == sha(path)
            bindings[str(path)] = sha(path)
            mask = np.array(Image.open(path)) > 0
            assert mask.shape == (h, w)
            distances.append(signed_mask_distance(mask)[xy[:, 1], xy[:, 0]])
        near = evidence['near_by_camera'][ci]
        tube, hand = distances
        outside_both = (tube < -2) & (hand < -2)
        delta = evidence['depth_deltas'][ci]
        record = dict(camera=name, points=len(points), near=int(near.sum()),
                      near_inside_tube=int((near & (tube > 2)).sum()),
                      near_inside_hand=int((near & (hand > 2)).sum()),
                      near_outside_both=int((near & outside_both).sum()),
                      all_outside_both=int(outside_both.sum()),
                      near_boundary_or_overlap=int((near & ~(outside_both | (tube > 2) | (hand > 2))).sum()),
                      depth_delta_median=float(np.nanmedian(delta)))
        marked = im.copy(); draw = ImageDraw.Draw(marked)
        for (x, y), is_near in zip(xy, near):
            draw.point((int(x), int(y)), fill='red' if is_near else 'cyan')
        canvas = Image.new('RGB', (w * 2, h + 44))
        canvas.paste(im, (0, 44)); canvas.paste(marked, (w, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((2, 2), name, fill='white')
        draw.text((2, 17), f'near={record["near"]}/171; red=near, cyan=not near', fill='white')
        draw.text((2, 30), f'near in tube/hand/outside: {record["near_inside_tube"]}/{record["near_inside_hand"]}/{record["near_outside_both"]}', fill='white')
        canvas.save(out / f'{name}.png'); stats.append(record)
    save(out / 'result.json', dict(records=stats, input_hashes=bindings,
        images={p.name: sha(p) for p in out.glob('*.png')},
        semantic_band_native_pixels=2, posthoc_only=True, geometry_changed=False,
        heldout_used=False, masks_not_independent_shape_truth=True, visual_status='pending'))
    print(stats, flush=True)


if __name__ == '__main__':
    main()
