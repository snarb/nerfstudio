"""Independent source-feature foot depths; no masks, clipping, NN or box fitting.

DC rectangles are manually selected boot regions, not silhouette certificates.
Other-camera feature detection uses full images, independent of the tested box.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linprog


def projection(row):
    pose = np.asarray(row['transform_matrix'], float)
    rotation = np.diag([1., -1., -1.]) @ pose[:3, :3].T
    intrinsic = np.array([[row['fl_x'], 0, row['cx'] - .5],
                          [0, row['fl_y'], row['cy'] - .5], [0, 0, 1.]])
    return intrinsic @ np.column_stack([rotation, -rotation @ pose[:3, 3]])


def triangulate(observations, matrices):
    equations = []
    for camera, uv in observations:
        p = matrices[camera]
        equations.extend([uv[0] * p[2] - p[0], uv[1] * p[2] - p[1]])
    _, _, vt = np.linalg.svd(equations, full_matrices=False)
    h = vt[-1]
    if abs(h[3]) < 1e-10:
        return None, None
    xyz = h[:3] / h[3]
    errors = []
    for camera, uv in observations:
        q = matrices[camera] @ np.r_[xyz, 1.]
        if q[2] <= 0:
            return None, None
        errors.append(float(np.linalg.norm(q[:2] / q[2] - uv)))
    return xyz, errors


def region_frustum_envelope(regions, matrices, padding=0.):
    """Bound ALL XYZ projecting into each supplied ROI, without a search AABB.

    Regions must enclose the same entire visible object in every supplied view.
    This proves conditional containment, not visibility of hidden object parts.
    Unbounded/infeasible systems fail explicitly, rather than borrowing test bounds.
    """
    if not np.isfinite(padding) or padding < 0:
        raise ValueError('ROI padding must be finite and nonnegative')
    planes = []
    for camera, rectangle in regions.items():
        x0, y0, x1, y1 = np.asarray(rectangle, float)
        if not np.isfinite(rectangle).all() or x1 <= x0 or y1 <= y0:
            raise ValueError('Expected ordered finite array-pixel rectangle')
        x0 -= padding; y0 -= padding; x1 += padding; y1 += padding
        p = matrices[camera]
        planes.extend([x0*p[2]-p[0], p[0]-x1*p[2],
                       y0*p[2]-p[1], p[1]-y1*p[2],
                       -p[2]+np.array([0., 0., 0., 1e-6])])
    planes = np.asarray(planes)
    limits = np.empty((2, 3))
    for axis in range(3):
        for side, sign in enumerate([1., -1.]):
            result = linprog(np.eye(3)[axis]*sign, A_ub=planes[:, :3],
                             b_ub=-planes[:, 3], bounds=[(None, None)]*3,
                             method='highs')
            if not result.success:
                raise ValueError('Foot frusta are infeasible or unbounded: '+result.message)
            limits[side, axis] = result.x[axis]
    return limits


def main(args):
    cv2.setNumThreads(2)
    args.output.mkdir(exist_ok=False, parents=True)
    source = json.loads(args.input.read_text())
    rows = {q['camera']: q for q in source['cameras']}
    matrices = {c: projection(q['camera_row']) for c, q in rows.items()}
    # Source pixels include visible boots; avoid the loose hat to the right.
    regions = {'black_near': [772, 548, 841, 596],
               'black_far': [955, 484, 1004, 528],
               'grey_rear': [914, 523, 978, 563],
               'grey_front': [1016, 554, 1080, 599]}
    sift = cv2.SIFT_create(nfeatures=18000, contrastThreshold=.008)
    image = cv2.imread(rows['DC']['source_rgb'], 0)
    mask = np.zeros_like(image)
    for x0, y0, x1, y1 in regions.values():
        mask[y0:y1, x0:x1] = 255
    keys, descriptors = sift.detectAndCompute(image, mask)
    uv = np.array([k.pt for k in keys])
    tracks = {i: [('DC', p)] for i, p in enumerate(uv)}
    matcher = cv2.BFMatcher()
    candidates = {i: [] for i in tracks}
    for camera, row in rows.items():
        if camera == 'DC':
            continue
        im = cv2.imread(row['source_rgb'], 0)
        otherkeys, otherdes = sift.detectAndCompute(im, None)
        if otherdes is None:
            continue
        for first, second in matcher.knnMatch(descriptors, otherdes, k=2):
            if first.distance >= .75 * second.distance:
                continue
            obs = [('DC', uv[first.queryIdx]),
                   (camera, np.asarray(otherkeys[first.trainIdx].pt))]
            xyz, errors = triangulate(obs, matrices)
            if xyz is None or max(errors) > 1.0:
                continue
            origins = [np.array(rows[c]['camera_row']['transform_matrix'])[:3, 3]
                       for c, _ in obs]
            rays = [xyz - o for o in origins]
            angle = np.degrees(np.arccos(np.clip(np.dot(*rays) /
                                np.prod([np.linalg.norm(v) for v in rays]), -1, 1)))
            if 5 < angle < 100:
                candidates[first.queryIdx].append((camera, obs[1][1], xyz))
        print('camera', camera, 'candidates', sum(map(len, candidates.values())), flush=True)
    output = []
    bounds = np.array(json.loads(args.box.read_text())['aabb'])
    for index, options in candidates.items():
        # Require DC plus at least two independently detected source views.
        best = []
        for _, _, hypothesis in options:
            accepted = []
            for camera, pixel, _ in options:
                q = matrices[camera] @ np.r_[hypothesis, 1.]
                if q[2] > 0 and np.linalg.norm(q[:2] / q[2] - pixel) < 2:
                    accepted.append((camera, pixel))
            if len(accepted) > len(best):
                best = accepted
        if len(best) < 2:
            continue
        observations = tracks[index] + best
        xyz, errors = triangulate(observations, matrices)
        if xyz is None or max(errors) > 1.5:
            continue
        labels = [name for name, (x0, y0, x1, y1) in regions.items()
                  if x0 <= uv[index, 0] < x1 and y0 <= uv[index, 1] < y1]
        output.append(dict(dc_keypoint=index, regions=labels, xyz=xyz.tolist(),
            observations=[dict(camera=c, array_xy=p.tolist()) for c, p in observations],
            reprojection_errors_hd=errors, lower_gaps=(xyz-bounds[0]).tolist(),
            upper_gaps=(bounds[1]-xyz).tolist(), inside=bool(((xyz >= bounds[0]) &
                                                          (xyz <= bounds[1])).all())))
    receipt = dict(kind='INDEPENDENT_MULTI_VIEW_FOOT_FEATURE_DIAGNOSTIC',
        points=output, regions=regions, aabb=bounds.tolist(),
        region_counts={k: sum(k in p['regions'] for p in output) for k in regions},
        outside_count=sum(not p['inside'] for p in output),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        box_sha256=hashlib.sha256(args.box.read_bytes()).hexdigest(),
        source_rgb_sha256={c: hashlib.sha256(Path(q['source_rgb']).read_bytes()).hexdigest()
                           for c, q in rows.items()},
        limitations='Needs manual patch review; feature interiors do not certify entire boot outline. '
                    'Evaluation RGB used for geometric debugging only; no photometric fit or training.',
        masks_used=False, aabb_used_to_select_or_filter_points=False, optimizer_updates=0)
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['region_counts', 'outside_count']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--box', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
