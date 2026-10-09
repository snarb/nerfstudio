#!/usr/bin/env python3
"""Read-only feature/epipolar audit around the original 000470 belt cord."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from triangulate_luster_cord_landmarks import cameras_from_colmap, project


CAMERAS = (25, 26, 27, 95, 97, 98)
PAIRS = ((25, 26), (26, 27), (25, 27), (97, 98), (25, 97),
         (26, 98), (27, 97), (95, 97), (95, 98))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fundamental(left_id: int, right_id: int, cameras: dict, poses: dict) -> np.ndarray:
    left_rotation, left_translation, left_intrinsics = poses[left_id]
    right_rotation, right_translation, right_intrinsics = poses[right_id]
    matrices = []
    for camera_id in (left_intrinsics, right_intrinsics):
        fx, fy, cx, cy = cameras[camera_id]['intrinsic']
        matrices.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]))
    relative_rotation = right_rotation @ left_rotation.T
    translation = right_translation - relative_rotation @ left_translation
    cross = np.array([[0, -translation[2], translation[1]],
                      [translation[2], 0, -translation[0]],
                      [-translation[1], translation[0], 0]])
    return np.linalg.inv(matrices[1]).T @ cross @ relative_rotation @ np.linalg.inv(matrices[0])


def sampson_pixels(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.column_stack([left, np.ones(len(left))])
    y = np.column_stack([right, np.ones(len(right))])
    fx = (matrix @ x.T).T
    fty = (matrix.T @ y.T).T
    numerator = np.sum(y * fx, axis=1) ** 2
    denominator = fx[:, 0] ** 2 + fx[:, 1] ** 2 + fty[:, 0] ** 2 + fty[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-30))


def lower_band(points: np.ndarray, endpoints: tuple[np.ndarray, np.ndarray], radius: float) -> np.ndarray:
    start, end = endpoints
    segment = end - start
    fraction = (points - start) @ segment / (segment @ segment)
    distance = np.linalg.norm(points - (start + fraction[:, None] * segment), axis=1)
    return (fraction >= .6) & (fraction <= .92) & (distance < radius)


def audit(fit_path: Path, sparse: Path, images: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fit = json.loads(fit_path.read_text())
    if (fit.get('train_only') is not True or fit['validation_camera_id'] != 95
            or sha(sparse/'cameras.txt') != fit['sparse_camera_sha256']
            or sha(sparse/'images.txt') != fit['sparse_image_sha256']):
        raise ValueError('Cord fit does not match original train-only calibration')
    cameras, poses = cameras_from_colmap(sparse)
    start = np.asarray(fit['fits']['junction']['original_world_xyz'])
    end = np.asarray(fit['fits']['first_bead']['original_world_xyz'])
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.01)
    features = {}
    projected = {}
    source_hashes = {}
    feature_counts = {}
    for camera_id in CAMERAS:
        path = images/f'cam_{camera_id:03d}_{fit["frame"]}.jpg'
        source_hashes[str(camera_id)] = sha(path)
        if camera_id != 95 and source_hashes[str(camera_id)] != fit['source_image_sha256'][str(camera_id)]:
            raise ValueError(f'Train-camera image changed: {camera_id}')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        first, _ = project(camera_id, start, cameras, poses)
        last, _ = project(camera_id, end, cameras, poses)
        projected[camera_id] = (first, last)
        low = np.maximum(np.floor(np.minimum(first, last) - [350, 300]).astype(int), 0)
        high = np.minimum(np.ceil(np.maximum(first, last) + [350, 300]).astype(int),
                          [image.shape[1], image.shape[0]])
        keypoints, descriptors = sift.detectAndCompute(image[low[1]:high[1], low[0]:high[0]], None)
        points = np.array([keypoint.pt for keypoint in keypoints]) + low
        radius = 18 if camera_id in (95, 97, 98) else 12
        feature_counts[str(camera_id)] = dict(total=len(points),
                                              near_lower_cord=int(lower_band(points, (first, last), radius).sum()),
                                              radius_pixels=radius)
        features[camera_id] = (points, descriptors)
    matcher = cv2.BFMatcher()
    pairs = []
    for left_id, right_id in PAIRS:
        left_points, left_descriptors = features[left_id]
        right_points, right_descriptors = features[right_id]
        matches = [first for first, second in matcher.knnMatch(left_descriptors, right_descriptors, k=2)
                   if first.distance < .75 * second.distance]
        if len(matches) < 8:
            raise ValueError(f'Insufficient feature matches for {left_id}/{right_id}')
        left = left_points[[match.queryIdx for match in matches]]
        right = right_points[[match.trainIdx for match in matches]]
        estimated, mask = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, 2., .999, 10000)
        if estimated is None or mask is None:
            raise ValueError(f'No independent epipolar inliers for {left_id}/{right_id}')
        inliers = mask.ravel().astype(bool)
        error = sampson_pixels(fundamental(left_id, right_id, cameras, poses), left[inliers], right[inliers])
        left_radius = feature_counts[str(left_id)]['radius_pixels']
        right_radius = feature_counts[str(right_id)]['radius_pixels']
        lower = inliers & lower_band(left, projected[left_id], left_radius) & lower_band(right, projected[right_id], right_radius)
        lower_error = sampson_pixels(fundamental(left_id, right_id, cameras, poses), left[lower], right[lower])
        pairs.append(dict(left_camera=left_id, right_camera=right_id, ratio_matches=len(matches),
                          independent_ransac_inliers=int(inliers.sum()),
                          calibration_sampson_median_px=float(np.median(error)),
                          calibration_sampson_p90_px=float(np.quantile(error, .9)),
                          lower_cord_inliers=int(lower.sum()),
                          lower_cord_sampson_median_px=float(np.median(lower_error)) if len(lower_error) else None,
                          heldout_read_only=95 in (left_id, right_id)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(frame=fit['frame'],read_only=True,masks_used=False,
                                      heldout_camera_excluded_from_training=95,
                                      scope='SIFT ratio matches and independent RANSAC inliers in original JPG crops',
                                      fit_sha256=sha(fit_path),sparse_camera_sha256=sha(sparse/'cameras.txt'),
                                      sparse_image_sha256=sha(sparse/'images.txt'),source_image_sha256=source_hashes,
                                      feature_counts=feature_counts,pairs=pairs),indent=2)+'\n')
    print(json.dumps(pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--sparse', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.fit, args.sparse, args.images, args.output)
