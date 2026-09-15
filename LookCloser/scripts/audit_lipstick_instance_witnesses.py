"""Read-only semantic attribution of existing measured-depth fin protections."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project
from study_lipstick_instance_masks import ROOT, portrait_xy

DIAG = Path('/mnt/data/dec5_lipstick_fin_depth/000995')


def signed_mask_distance(mask):
    value = np.asarray(mask, np.uint8)
    if value.ndim != 2 or not np.isin(value, [0, 1]).all():
        raise ValueError('Expected a binary image')
    return cv2.distanceTransform(value, cv2.DIST_L2, 5) - cv2.distanceTransform(1-value, cv2.DIST_L2, 5)


def run():
    output = ROOT / 'witness_audit'; assert not output.exists()
    request = read(ROOT / 'request.json'); results = read(ROOT / 'sam_v2/result.json')
    review = read(ROOT / 'mask_review.json')
    assert review['status'] == 'usable_for_bounded_semantic_diagnostic'
    assert results['request_sha256'] == sha(ROOT / 'request.json')
    assert review['result_sha256'] == sha(ROOT / 'sam_v2/result.json')
    old = read(DIAG / 'result.json')
    assert sha(DIAG / 'result.json') == request['diagnostic_result_sha256']
    for name, digest in old['hashes'].items():
        assert sha(DIAG / name) == digest
    evidence = np.load(DIAG / 'evidence.npz')
    points = evidence['points']; representative = old['representative_triangle']
    ri = int(np.flatnonzero(evidence['triangle_ids'] == representative)[0])
    original_rows = old['representative_observations']
    output.mkdir(); records = []; distances = []; classifications = []
    for row, candidate in zip(request['views'], results['views']):
        name = row['camera']; assert name == candidate['camera']
        for path, digest in candidate['outputs'].items():
            assert sha(ROOT / 'sam_v2' / path) == digest
        chosen = review['selected'][name]
        assert chosen == int(np.argmax(candidate['scores']))
        folder = ROOT / 'sam_v2' / name
        with Image.open(folder / f'mask_{chosen}.png') as im: mask = np.asarray(im) > 0
        sdf = signed_mask_distance(mask)
        uv, z = project(points.reshape(-1, 3), [row['camera_parameters']])
        xy = portrait_xy(uv[0], row['camera_parameters']['w']) - np.array(row['crop'][:2])
        inside_crop = ((xy >= 0) & (xy <= [mask.shape[1]-1, mask.shape[0]-1])).all(1) & (z[0] > 0)
        values = cv2.remap(sdf, xy.astype(np.float32).reshape(1, -1, 2), None, cv2.INTER_LINEAR)[0]
        # Unknown outside saved crop; never extrapolate a semantic rejection.
        kind = np.zeros(len(xy), np.int8)
        kind[inside_crop & (values >= 2)] = 1
        kind[inside_crop & (values <= -2)] = -1
        kind[inside_crop & (np.abs(values) < 2)] = 2
        ci = next(i for i, r in enumerate(original_rows) if r['camera'] == name)
        available = evidence['available'][ci]
        near = available & (np.abs(evidence['deltas'][ci]) <= .0015)
        kind = kind.reshape(points.shape[:2]); values = values.reshape(points.shape[:2])
        image = Image.open(row['image']).convert('RGB')
        marked = image.copy(); draw = ImageDraw.Draw(marked)
        for i, p in enumerate(xy.reshape(*points.shape[:2], 2)[ri]):
            x, y = p; color = 'red' if near[ri, i] else 'cyan'
            draw.ellipse((x-2, y-2, x+2, y+2), outline=color, width=1)
            draw.text((x+3, y+2), str(i), fill=color)
        sheet = Image.new('RGB', (480, 330)); sheet.paste(image, (0, 30)); sheet.paste(marked, (240, 30))
        ImageDraw.Draw(sheet).text((3, 3), f'{name} face {representative}; red = near depth', fill='white')
        sheet.save(output / f'{name}.png')
        records.append(dict(camera=name, selected_mask=chosen,
            near_sample_count=int(near.sum()), near_inside_object=int((near & (kind == 1)).sum()),
            near_outside_object=int((near & (kind == -1)).sum()),
            near_object_edge=int((near & (kind == 2)).sum()), near_unknown=int((near & (kind == 0)).sum()),
            representative_near=near[ri].tolist(), representative_signed_mask_distances=values[ri].tolist(),
            representative_semantic_class=kind[ri].tolist()))
        distances.append(values); classifications.append(kind)
    np.savez_compressed(output / 'evidence.npz', triangle_ids=evidence['triangle_ids'],
                        signed_mask_distances=distances, semantic_class=classifications)
    atomic_json(output / 'result.json', dict(status='diagnostic_not_geometry_acceptance',
        mask_review_sha256=sha(ROOT / 'mask_review.json'), request_sha256=sha(ROOT / 'request.json'),
        diagnostic_result_sha256=sha(DIAG / 'result.json'), script_sha256=sha(__file__),
        margin_native_pixels=2, near_depth_tolerance=.0015, representative_triangle=int(representative),
        semantic_classes={'inside_object': 1, 'outside_object': -1, 'edge_ambiguous': 2, 'unknown': 0},
        views=records, geometry_changed=False, heldout_used=False,
        visual_status='pending', hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(records, flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(2)
    run()
