#!/usr/bin/env python3
"""Audit sparse train-view cord landmarks against original COLMAP cameras.

This is read-only geometry evidence. The held-out view is projected only after
both 3D landmarks have been fitted; no RGB masks or model outputs enter the fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cameras_from_colmap(sparse):
    cameras = {}
    for line in (sparse / 'cameras.txt').read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        if fields[1] != 'PINHOLE':
            raise ValueError('Only original PINHOLE cameras are supported')
        cameras[int(fields[0])] = dict(size=tuple(map(int, fields[2:4])),
                                        intrinsic=np.array(list(map(float, fields[4:8]))))
    poses = {}
    for line in (sparse / 'images.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) != 10 or not fields[-1].endswith('.jpg'):
            continue
        qw, qx, qy, qz = map(float, fields[1:5])
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        poses[int(fields[0])] = (rotation, np.array(list(map(float, fields[5:8]))), int(fields[8]))
    return cameras, poses


def project(camera_id, point, cameras, poses):
    rotation, translation, intrinsics_id = poses[camera_id]
    q = rotation @ point + translation
    fx, fy, cx, cy = cameras[intrinsics_id]['intrinsic']
    return np.array([fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy]), float(q[2])


def triangulate(observations, cameras, poses):
    aa, bb = [], []
    for camera_id, (x, y) in observations.items():
        rotation, translation, intrinsics_id = poses[camera_id]
        fx, fy, cx, cy = cameras[intrinsics_id]['intrinsic']
        center = -rotation.T @ translation
        direction = rotation.T @ np.array([(x - cx) / fx, (y - cy) / fy, 1.])
        direction /= np.linalg.norm(direction)
        a = np.eye(3) - np.outer(direction, direction)
        aa.append(a)
        bb.append(a @ center)
    seed = np.linalg.solve(np.sum(aa, axis=0), np.sum(bb, axis=0))
    refined = least_squares(
        lambda point: np.concatenate([project(i, point, cameras, poses)[0] - xy
                                      for i, xy in observations.items()]),
        seed, loss='soft_l1', f_scale=5., max_nfev=100)
    point = refined.x
    if not refined.success:
        raise ValueError('Landmark fit did not converge')
    errors = {str(i): float(np.linalg.norm(project(i, point, cameras, poses)[0] - xy))
              for i, xy in observations.items()}
    return point, errors


def normalization_from_camera_centers(frames, poses):
    raw = []
    normalized = []
    for frame in frames:
        camera_id = frame['camera_id']
        rotation, translation, _ = poses[camera_id]
        raw.append(-rotation.T @ translation)
        normalized.append(np.asarray(frame['transform_matrix'])[:3, 3])
    raw, normalized = np.stack(raw), np.stack(normalized)
    raw_center, normalized_center = raw.mean(axis=0), normalized.mean(axis=0)
    left, singular, right = np.linalg.svd((raw - raw_center).T @ (normalized - normalized_center))
    diagonal = np.eye(3)
    diagonal[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ diagonal @ right
    scale = np.trace(np.diag(singular) @ diagonal) / np.sum((raw - raw_center) ** 2)
    translation = normalized_center - scale * raw_center @ rotation
    residual = np.linalg.norm(scale * raw @ rotation + translation - normalized, axis=1)
    if max(residual) > 1e-5:
        raise ValueError(f'Original-to-training coordinate mapping is not rigid: {max(residual)}')
    return scale, rotation, translation, float(max(residual))


def closest_segment_point_to_ray(start, end, origin, direction):
    segment = end - start
    # For a segment point start+u*segment, minimize distance to the infinite
    # camera ray. The fitted segment is in front of the camera in this audit.
    projector = np.eye(3) - np.outer(direction, direction)
    u = -float(segment @ projector @ (start - origin)) / float(segment @ projector @ segment)
    u = float(np.clip(u, 0., 1.))
    point = start + u * segment
    ray_t = float((point - origin) @ direction)
    distance = float(np.linalg.norm(point - (origin + ray_t * direction)))
    return u, ray_t, distance


def pixel_ray(frame, xy):
    matrix = np.asarray(frame['transform_matrix'])
    x, y = xy
    direction = matrix[:3, :3] @ np.array([(x + .5 - frame['cx']) / frame['fl_x'],
                                           -(y + .5 - frame['cy']) / frame['fl_y'], -1.])
    direction /= np.linalg.norm(direction)
    return matrix[:3, 3], direction


def overlay(image_path, crop_box, camera_id, endpoints, cameras, poses, observations, destination):
    crop = Image.open(image_path).convert('RGB').crop(crop_box)
    draw = ImageDraw.Draw(crop)
    projected = [project(camera_id, (1 - u) * endpoints[0] + u * endpoints[1], cameras, poses)[0]
                 for u in np.linspace(0, 1, 33)]
    draw.line([(float(x - crop_box[0]), float(y - crop_box[1])) for x, y in projected],
              fill=(255, 255, 0), width=2)
    for label, point, color in [('junction', endpoints[0], (0, 255, 0)),
                                ('first_bead', endpoints[1], (0, 180, 255))]:
        x, y = project(camera_id, point, cameras, poses)[0] - crop_box[:2]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        if camera_id in observations[label]:
            x, y = np.array(observations[label][camera_id]) - crop_box[:2]
            draw.line((x - 5, y, x + 5, y), fill=(255, 0, 255))
            draw.line((x, y - 5, x, y + 5), fill=(255, 0, 255))
    crop.save(destination)


def audit(args):
    spec = json.loads(args.landmarks.read_text())
    transforms = json.loads(args.transforms.read_text())
    train = set(transforms['train_filenames'])
    frames = {f['camera_id']: f for f in transforms['frames']}
    cameras, poses = cameras_from_colmap(args.sparse_text)
    observations = {label: {int(i): np.asarray(xy, dtype=float) for i, xy in items.items()}
                    for label, items in spec['landmarks'].items()}
    if set(observations) != {'junction', 'first_bead'}:
        raise ValueError('Expected the two reviewed cord landmarks')
    if any(len(items) < 3 for items in observations.values()):
        raise ValueError('Each landmark requires at least three training views')
    for items in observations.values():
        for camera_id, xy in items.items():
            if frames[camera_id]['file_path'] not in train:
                raise ValueError('Held-out camera used for fitting')
            width, height = cameras[poses[camera_id][2]]['size']
            if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                raise ValueError('Annotation outside the original camera image')
    validation_id = int(spec['validation_camera_id'])
    if frames[validation_id]['file_path'] in train:
        raise ValueError('Validation camera must remain held out')
    source_ids = sorted(set().union(*(set(v) for v in observations.values()), {validation_id}))
    source_images = {i: args.source_images / f'cam_{i:03d}_{spec["frame"]}.jpg' for i in source_ids}
    source_hashes = {str(i): digest(path) for i, path in source_images.items()}
    scale, rotation, translation, map_error = normalization_from_camera_centers(list(frames.values()), poses)
    fits = {label: triangulate(items, cameras, poses) for label, items in observations.items()}
    endpoints = [fits[label][0] for label in ('junction', 'first_bead')]
    normalized_endpoints = [scale * point @ rotation + translation for point in endpoints]
    heldout_projection = {label: project(validation_id, fits[label][0], cameras, poses)[0].tolist()
                          for label in fits}
    leave_one_out = {}
    for label, items in observations.items():
        rows = []
        for omitted in items:
            point, _ = triangulate({i: xy for i, xy in items.items() if i != omitted}, cameras, poses)
            rows.append(dict(omitted_camera=omitted,
                             heldout_projection=project(validation_id, point, cameras, poses)[0].tolist(),
                             normalized_xyz=(scale * point @ rotation + translation).tolist()))
        leave_one_out[label] = rows
    ray_origin, ray_direction = pixel_ray(frames[validation_id], spec['validation_pixel_hd'])
    u, ray_t, ray_distance = closest_segment_point_to_ray(*normalized_endpoints, ray_origin, ray_direction)
    random = np.random.default_rng(470)
    bootstrap = []
    for _ in range(200):
        sampled = {}
        for label, items in observations.items():
            sigma = spec['annotation_uncertainty_px'][label]
            sampled[label] = {i: xy + random.normal(0, sigma, 2) for i, xy in items.items()}
        fitted = [triangulate(sampled[label], cameras, poses)[0] for label in ('junction', 'first_bead')]
        normalized = [scale * point @ rotation + translation for point in fitted]
        _, t, distance = closest_segment_point_to_ray(*normalized, ray_origin, ray_direction)
        bootstrap.append((t, distance))
    bootstrap = np.asarray(bootstrap)
    args.output.mkdir(parents=True, exist_ok=False)
    crop_manifest = json.loads(args.crop_manifest.read_text()) if args.crop_manifest else None
    if crop_manifest:
        crop_by_id = {r['camera']: r['crop_box'] for r in crop_manifest['photos']}
        for camera_id in source_ids:
            overlay(source_images[camera_id], crop_by_id[camera_id], camera_id, endpoints, cameras, poses,
                    observations, args.output / f'cam{camera_id:03d}_segment_projection.png')
    result = dict(frame=spec['frame'], train_only=True, coordinate_system='train_normalized_opengl',
                  fit_camera_ids={k: sorted(v) for k, v in observations.items()},
                  validation_camera_id=validation_id, source_image_sha256=source_hashes,
                  sparse_camera_sha256=digest(args.sparse_text / 'cameras.txt'),
                  sparse_image_sha256=digest(args.sparse_text / 'images.txt'),
                  transforms_sha256=digest(args.transforms), landmarks_sha256=digest(args.landmarks),
                  original_to_normalized=dict(scale=scale, rotation=rotation.tolist(),
                                              translation=translation.tolist(), max_camera_center_error=map_error),
                  fits={label: dict(original_world_xyz=point.tolist(),
                                    normalized_world_xyz=(scale * point @ rotation + translation).tolist(),
                                    reprojection_errors_px=errors)
                        for label, (point, errors) in fits.items()},
                  heldout_projection_original_px=heldout_projection,
                  heldout_ray=dict(pixel_hd=spec['validation_pixel_hd'],
                                   nearest_segment_fraction=u, cord_ray_t=ray_t,
                                   ray_to_segment_distance=ray_distance,
                                   bootstrap_t_05_50_95=np.quantile(bootstrap[:, 0], [.05, .5, .95]).tolist(),
                                   bootstrap_distance_05_50_95=np.quantile(bootstrap[:, 1], [.05, .5, .95]).tolist()),
                  leave_one_out=leave_one_out,
                  limitation='Two annotated landmarks approximate a straight centerline; neither tube radius nor local surface normal is established. Held-out RGB is validation only; no masks.')
    (args.output / 'fit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), heldout_ray=result['heldout_ray'],
                          maximum_reprojection_error_px={k: max(v[1].values()) for k, v in fits.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--sparse-text', type=Path, required=True)
    parser.add_argument('--source-images', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-manifest', type=Path)
    audit(parser.parse_args())
