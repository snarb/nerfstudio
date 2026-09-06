#!/usr/bin/env python3
"""Opt-in train-only common-rig bundle adjustment over separate temporal points.

One virtual observation image per physical camera combines the independent
point tracks from all fit times. Thus camera poses/intrinsics are tied exactly
across time; moving scene points never share a track across times. Eval camera
rows remain byte-equivalent as JSON values and no eval RGB is read.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pycolmap
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from audit_source_epipolar_residuals import projection


def align_similarity(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Need matching camera centers')
    x, y = source - source.mean(0), target - target.mean(0)
    u, s, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.array([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ np.diag(sign) @ vt
    scale = float(s @ sign / np.mean(np.sum(x * x, axis=1)))
    translation = target.mean(0) - scale * (rotation @ source.mean(0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid similarity gauge alignment')
    return scale, rotation, translation


def make_reconstruction(cameras, tracks):
    reconstruction = pycolmap.Reconstruction()
    image_points = [[] for _ in cameras]
    elements = []
    for track in tracks:
        seen, refs = set(), []
        for observation in track['observations']:
            camera = observation['camera']
            if camera in seen or not 0 <= camera < len(cameras):
                raise ValueError('Track camera conflict')
            seen.add(camera)
            refs.append(pycolmap.TrackElement(camera + 1, len(image_points[camera])))
            # OpenCV feature centers are zero-based; COLMAP uses +.5 centers.
            image_points[camera].append(np.asarray(observation['xy']) + .5)
        if len(seen) < 3:
            raise ValueError('Three views required')
        elements.append(refs)
    for index, frame in enumerate(cameras, 1):
        camera = pycolmap.Camera(camera_id=index, model='PINHOLE', width=frame['w'], height=frame['h'],
            params=[frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy']])
        reconstruction.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(name=frame['physical_camera'] + '.virtual', camera_id=index, image_id=index,
            keypoints=np.asarray(image_points[index - 1]).reshape(-1, 2))
        _, w2c = projection(frame, .5)
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(w2c[:3]))
    for track, refs in zip(tracks, elements):
        reconstruction.add_point3D(np.asarray(track['xyz']), pycolmap.Track(refs))
    return reconstruction


def reprojection_profile(reconstruction, tracks):
    groups = {}
    for point_id, track in zip(sorted(reconstruction.points3D), tracks):
        xyz = reconstruction.points3D[point_id].xyz
        for observation in track['observations']:
            camera_id = observation['camera'] + 1
            image = reconstruction.images[camera_id]
            camera = reconstruction.cameras[camera_id]
            p = image.cam_from_world() * xyz
            if p[2] <= 0:
                raise ValueError('Optimized point behind camera')
            fx, fy, cx, cy = camera.params
            predicted = p[:2] / p[2] * [fx, fy] + [cx, cy]
            error = float(np.linalg.norm(predicted - np.asarray(observation['xy']) - .5))
            if not np.isfinite(error):
                raise ValueError('Nonfinite reprojection')
            groups.setdefault((track['frame_id'], camera_id), []).append(error)
    return [dict(frame_id=t, camera_id=c, observations=len(e), median=float(np.median(e)), p90=float(np.quantile(e, .9)))
            for (t, c), e in sorted(groups.items())]


def optimize(reconstruction, mode, iterations=200, threads=12, regularized=False):
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = mode in {'focal', 'pinhole'}
    options.refine_principal_point = mode == 'pinhole'
    options.refine_extra_params = False
    options.print_summary = True
    options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    options.ceres.loss_function_scale = 1.
    options.ceres.use_gpu = False
    options.ceres.solver_options.max_num_iterations = iterations
    options.ceres.solver_options.num_threads = threads
    if regularized:
        options.ceres.solver_options.max_num_iterations = max(iterations, 1000)
        options.ceres.solver_options.function_tolerance = 1e-8
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.images:
        config.add_image(image_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    if regularized:
        import pyceres  # Explicit opt-in dependency; absent in ordinary mode.
        from pycolmap.cost_functions import AbsolutePosePriorCost
        adjuster = pycolmap.create_default_ceres_bundle_adjuster(options, config, reconstruction)
        problem = adjuster.problem
        covariance = np.diag([np.radians(.2) ** 2] * 3 + [.03 ** 2] * 3)
        for frame in reconstruction.frames.values():
            pose = frame.rig_from_world
            if not problem.has_parameter_block(pose.params):
                raise RuntimeError('Pose prior must attach to the actual optimized parameter block')
            prior = pycolmap.Rigid3d(pose.matrix().copy())
            problem.add_residual_block(AbsolutePosePriorCost(covariance, prior), None, [pose.params])
        for camera in reconstruction.cameras.values():
            values = camera.params
            for index, value in enumerate(values.copy()):
                delta = value * .05 if index < 2 else 32.
                problem.set_parameter_lower_bound(values, index, float(value - delta))
                problem.set_parameter_upper_bound(values, index, float(value + delta))
        result = adjuster.solve()
    else:
        result = pycolmap.create_default_bundle_adjuster(options, config, reconstruction).solve()
    if not result.is_solution_usable():
        raise RuntimeError('Unusable common-rig solution')
    return result, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['poses', 'focal', 'pinhole'], required=True)
    parser.add_argument('--regularized', action='store_true', help='Common pose priors and bounded intrinsic corrections.')
    a = parser.parse_args()
    if a.output.exists():
        parser.error('Preserve existing calibration control')
    payload = json.loads(a.tracks.read_text())
    if any(payload[k] for k in ['uses_eval_rgb', 'uses_semantic_masks', 'cross_time_matching', 'changes_calibration']):
        raise ValueError('Need unmodified train-only temporal tracks')
    request_path = Path(payload['request'])
    if sha256(request_path) != payload['request_sha256']:
        raise ValueError('Track request changed')
    track_request = json.loads(request_path.read_text())
    for file, digest in track_request['input_hashes'].items():
        if sha256(Path(file)) != digest:
            raise ValueError(f'Changed source evidence: {file}')
    template_path = Path(payload['calibration'])
    if sha256(template_path) != payload['calibration_sha256']:
        raise ValueError('Template changed')
    template = json.loads(template_path.read_text())
    by_camera = {f['physical_camera']: f for f in template['frames']}
    if len(by_camera) != len(template['frames']):
        raise ValueError('Duplicate physical camera')
    names = payload['physical_cameras']
    if len(names) != 62 or len(set(names)) != 62 or set(names) & {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}:
        raise ValueError('Exactly 62 unique train cameras required')
    cameras = [by_camera[n] for n in names]
    reconstruction = make_reconstruction(cameras, payload['tracks'])
    before = reprojection_profile(reconstruction, payload['tracks'])
    result, options = optimize(reconstruction, a.mode, regularized=a.regularized)
    source_centers = np.array([reconstruction.images[i].projection_center() for i in range(1, 63)])
    original_centers = np.array([np.asarray(f['transform_matrix'])[:3, 3] for f in cameras])
    scale, rotation, translation = align_similarity(source_centers, original_centers)
    reconstruction.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation))
    after = reprojection_profile(reconstruction, payload['tracks'])
    output = copy.deepcopy(template)
    output_rows = {f['physical_camera']: f for f in output['frames']}
    changes = []
    for camera_id, name in enumerate(names, 1):
        camera = reconstruction.cameras[camera_id]
        w2c = np.eye(4)
        w2c[:3] = reconstruction.images[camera_id].cam_from_world().matrix()
        pose = np.linalg.inv(w2c) @ np.diag([1., -1., -1., 1.])
        old_pose = np.asarray(by_camera[name]['transform_matrix'])
        row = output_rows[name]
        row['transform_matrix'] = pose.tolist()
        row.update(dict(zip(['fl_x', 'fl_y', 'cx', 'cy'], map(float, camera.params))))
        angle = np.degrees(np.arccos(np.clip((np.trace(old_pose[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
        changes.append(dict(physical_camera=name, center_shift_world=float(np.linalg.norm(pose[:3, 3] - old_pose[:3, 3])), rotation_degrees=float(angle),
            focal_ratio_xy=[row[k] / by_camera[name][k] for k in ['fl_x', 'fl_y']], principal_shift_xy=[row[k] - by_camera[name][k] for k in ['cx', 'cy']]))
    for name in set(by_camera) - set(names):
        if output_rows[name] != by_camera[name]:
            raise ValueError('Held-out calibration changed')
    a.output.mkdir(parents=True)
    atomic_json(a.output / 'transforms.json', output)
    model = a.output / 'sparse_model'
    model.mkdir()
    reconstruction.write(str(model))
    atomic_json(a.output / 'manifest.json', dict(method='one_shared_train_rig_multiple_time_separated_point_sets',
        mode=a.mode, regularized=a.regularized, uses_eval_rgb=False, uses_semantic_masks=False, eval_camera_calibration_unchanged=True,
        regularization=(dict(pose_rotation_std_degrees=.2, pose_translation_std_world=.03, focal_bound_fraction=.05, principal_bound_pixels=32.) if a.regularized else None),
        changes_source_frame_identity=False, per_time_camera_optimization=False, train_cameras=62, frames=payload['frames'],
        points3D=reconstruction.num_points3D(), observations=sum(r['observations'] for r in after),
        before=before, after=after, camera_changes=changes, solver_report=result.brief_report(),
        solver_options=options.summary(), pycolmap_version=pycolmap.__version__,
        optimizer_note='PyCOLMAP used for sparse BA only. Dense PatchMatch remains pinned CUDA commit 5509fffe.',
        gauge_alignment=dict(scale=scale, rotation=rotation.tolist(), translation=translation.tolist()),
        input_hashes={str(a.tracks): sha256(a.tracks), str(template_path): sha256(template_path), str(Path(__file__)): sha256(Path(__file__))},
        output_hashes={str(path.relative_to(a.output)): sha256(path) for path in sorted(a.output.rglob('*')) if path.is_file()},
        status='candidate_calibration_not_a_render_pass'))
    print(json.dumps(dict(mode=a.mode, output=str(a.output), points=reconstruction.num_points3D(),
        median_camera_time_reprojection_before_after=[float(np.median([r['median'] for r in rows])) for rows in [before, after]],
        max_center_change=max(r['center_shift_world'] for r in changes), max_rotation_degrees=max(r['rotation_degrees'] for r in changes))), flush=True)


if __name__ == '__main__':
    main()
