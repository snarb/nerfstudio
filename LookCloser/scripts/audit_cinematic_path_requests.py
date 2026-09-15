"""Independent request audit: raw-calibration motion, time and native-pose hold.

Uses the saved matrices, not the path generator or its reported statistics.
Optical focal changes are reported independently of camera-center movement.
This is not a visual-quality approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from colmap_patchmatch_tsdf_campaign_common import atomic_json

CALIBRATION = Path('/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150/config/calibration/transforms.json')
PARENT = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150/request.json')
HELD = {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}
VARIANTS = ['locked_arc', 'free_arc', 'rising_arc', 'soft_diagonal']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_matrix(matrix, metadata, calibration):
    matrix = np.asarray(matrix, float).copy()
    matrix[:3, 3] /= metadata['dataparser_scale']
    transform = np.eye(4); transform[:3] = metadata['dataparser_transform']
    applied = np.eye(4)
    if 'applied_transform' in calibration: applied[:3] = calibration['applied_transform']
    return applied @ np.linalg.inv(transform) @ matrix


def audit(base):
    calibration, parent = read(CALIBRATION), read(PARENT)
    source = {r['physical_camera']: r for r in calibration['frames']}
    train = [r for name, r in source.items() if name not in HELD]
    assert len(train) == 62
    hull = ConvexHull(np.asarray([r['transform_matrix'] for r in train])[:, :3, 3])
    expected = [f'{i:06d}' for i in range(899, 1198, 2)]
    records = []
    for variant in VARIANTS:
        path = base/variant/'request.json'; request = read(path)
        inventory = request['inventory']; report = request['camera_path_report']
        assert request['ordered_frame_ids'] == expected
        assert [r['frame_id'] for r in inventory] == expected
        assert len(inventory) == len(set(expected)) == 150
        assert request['source_rows'] == parent['source_rows']
        for key in ['profiles_sha256', 'exposure_sha256', 'calibration_sha256']:
            assert request[key] == parent[key]
        raw = []
        for row, previous in zip(inventory, parent['inventory']):
            assert {k:v for k,v in row.items() if k != 'camera'} == {k:v for k,v in previous.items() if k != 'camera'}
            assert sha(row['metadata']) == row['metadata_sha256']
            raw.append(raw_matrix(row['camera']['transform_matrix'], read(row['metadata']), calibration))
        raw = np.asarray(raw); centers = raw[:, :3, 3]
        endpoint = source[report['endpoint_train_camera']]
        assert endpoint['physical_camera'] not in HELD
        hold_error = np.max(np.abs(raw[126:] - np.asarray(endpoint['transform_matrix'])))
        assert hold_error < 1e-9
        residual = float((centers @ hull.equations[:, :3].T + hull.equations[:, 3]).max())
        assert residual < 1e-7
        # Convert the fixed reference target as a point using the reference gauge.
        reference = np.eye(4); reference[:3, 3] = report['fixed_target']
        target = raw_matrix(reference, read(report['reference_metadata']), calibration)[:3, 3]
        radii = np.linalg.norm(centers-target, axis=1)
        travel = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        assert travel.sum() > 0 and travel[126:].max(initial=0) < 1e-9
        assert np.diff(radii).max() < 1e-8
        focal = np.array([[r['camera'][k] for k in ['fl_x', 'fl_y']] for r in inventory])
        ratios = focal/np.array([endpoint['fl_x'], endpoint['fl_y']])
        np.testing.assert_allclose(ratios[126:], report['virtual_focal_multiplier_start_end'][1], atol=1e-12)
        principal = np.array([[r['camera'][k] for k in ['cx', 'cy']] for r in inventory])
        sensor_shift = principal-np.array([endpoint['cx'], endpoint['cy']])
        expected_shift = report.get('virtual_sensor_principal_x_shift_pixels', 0)
        np.testing.assert_allclose(sensor_shift[126:, 0], expected_shift, atol=1e-9)
        np.testing.assert_allclose(sensor_shift[126:, 1], 0, atol=1e-9)
        rays = (target-centers)/radii[:, None]
        optical = -raw[:, :3, 2]; optical /= np.linalg.norm(optical, axis=1)[:, None]
        look_error = np.rad2deg(np.arccos(np.clip((rays*optical).sum(1), -1, 1)))
        if variant == 'locked_arc': assert look_error.max() < 1e-4
        records.append(dict(variant=variant, request_sha256=sha(path),
            unique_dynamic_times=150, train_pose_hold_frames=24, endpoint_pose_max_error=float(hold_error),
            train_hull_max_residual=residual, physical_path_length_raw_calibration_units=float(travel.sum()),
            physical_start_end_distance=float(np.linalg.norm(centers[-1]-centers[0])),
            physical_radial_approach_fraction=float(1-radii[-1]/radii[0]),
            first24_path_fraction=float(travel[:24].sum()/travel.sum()),
            first48_path_fraction=float(travel[:48].sum()/travel.sum()),
            first24_center_displacement=float(np.linalg.norm(centers[24]-centers[0])),
            focal_multiplier_start_end=ratios[[0,-1],0].tolist(),
            virtual_principal_shift_start_end=sensor_shift[[0,-1]].tolist(),
            look_at_error_max_degrees=float(look_error.max()),
            actor_geometry_and_sources_identical_to_parent=True, visual_approval=False))
    result = dict(records=records, script_sha256=sha(__file__), calibration_sha256=sha(CALIBRATION),
        parent_request_sha256=sha(PARENT), actual_train_extrinsic_but_virtual_focal=True,
        no_metric_or_artifact_free_claim=True)
    atomic_json(base/'independent_path_audit.json', result)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    audit(parser.parse_args().base)
