"""Seven-time RGB gate for a smooth local cheek-hole camera workaround.

Report: experiments/dec5_local_train_waypoint.md. No geometry correction,
full-video publication, RGB averaging, crop, stabilization or actor retiming.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from joint_temporal_texture import read, sha, atomic_json, CALIBRATION, cameras
from smooth_train_waypoint import displace_poses, motion_summary

PARENT = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
OUTPUT = Path('/mnt/data/dec5_local_train_waypoint')
FRAMES = ['000899', '000929', '001149', '001169', '001189', '001193', '001197']
PHYSICAL = 'G004_C005_121037'


def initialize():
    from render_smooth_temporal_mesh_video import verify_request, calibration_pose
    from render_patchmatch_camera_path import normalize_frame
    parent = verify_request(PARENT); request = deepcopy(parent); cal = read(CALIBRATION)
    raw = [calibration_pose(r['camera'], cal, read(r['metadata'])) for r in parent['inventory']]
    poses = np.array([r['transform_matrix'] for r in raw])
    rows, _, metadata = cameras('001193')
    target = calibration_pose(next(r for r in rows if r['physical_camera'] == PHYSICAL), cal, read(metadata))
    changed, weights = displace_poses(poses, np.array(target['transform_matrix']), 147, 35)
    original_motion = motion_summary(poses); new_motion = motion_summary(changed)
    if new_motion['view_angle_extent_degrees'] < 20 or new_motion['step_min'] < original_motion['step_median'] * .05:
        raise ValueError('Camera motion collapsed')
    if new_motion['step_max'] > original_motion['step_max'] * 4:
        raise ValueError('Excessive camera step')
    for index, (record, row, pose) in enumerate(zip(request['inventory'], raw, changed)):
        if weights[index] == 0: continue
        row['transform_matrix'] = pose.tolist()
        # The old path's composition, convex weights and grid coordinates no
        # longer describe these poses. Never silently retain false motion proof.
        for key in ['convex_weights', 'rig_offset_xy', 'scene_landmark_portrait_xy',
                    'arc_length_fraction', 'pilot_sample_phase']:
            row.pop(key, None)
        row['physical_camera'] = f'local_train_waypoint_{index:05d}'
        record['camera'] = normalize_frame(row, cal, read(record['metadata']))
    report = dict(method='periodic_cosine4_pose_displacement', center_index=147,
        target_train_camera=PHYSICAL, half_width_samples=35,
        target_intrinsics='unchanged movie intrinsics, not native train intrinsics',
        original_motion=original_motion, candidate_motion=new_motion,
        unchanged_pose_count=int((weights == 0).sum()), weights=weights.tolist(),
        train_hull_clearance_status='requires_independent_audit_before_full_video',
        geometry_fixed=False, image_crop=False, image_stabilization=False,
        actor_times_unchanged=True, camera_periodic=True, artifact_free_approval=False)
    request['camera_path_report'] = report
    request['recipe']['camera_path_variant'] = 'local_train_waypoint_canary'
    request.update(partial_diagnostic_only=True, full_video_candidate=False,
        artifact_free_approval=False, required_initial_rgb_gate=FRAMES,
        inherited_camera_and_actor_inventory_unchanged=False,
        inherited_actor_inventory_unchanged=True,
        camera_workaround_parent=dict(path=str(PARENT / 'request.json'), sha256=sha(PARENT / 'request.json')))
    for name in ['smooth_train_waypoint.py', Path(__file__).name]:
        request['script_hashes'][name] = sha(Path(__file__).with_name(name))
    OUTPUT.mkdir(exist_ok=True); (OUTPUT / 'frames').mkdir(exist_ok=True)
    if (OUTPUT / 'request.json').exists() and read(OUTPUT / 'request.json') != request:
        raise ValueError('Immutable request mismatch')
    atomic_json(OUTPUT / 'request.json', request)
    print(report, flush=True)


def render(worker, workers):
    from run_view_consistent_dynamic_video import worker as run
    if workers < 1 or not 0 <= worker < workers: raise ValueError('Invalid worker partition')
    run(OUTPUT, worker, FRAMES[worker::workers])


def panels():
    from PIL import Image
    from review_jaw_repair_transfer import verified_image, panel
    records = []
    for frame in FRAMES:
        old, _ = verified_image(PARENT, frame); new, _ = verified_image(OUTPUT, frame)
        paths = []
        for label, box in [('head', (100, 420, 1000, 1270)), ('jaw', (250, 820, 950, 1230))]:
            path = OUTPUT / 'review' / (frame + '_' + label + '.png')
            panel(path, [old, new], ['old path / same actor time', 'smooth train waypoint'], box)
            paths.append(dict(path=str(path), sha256=sha(path)))
        path = OUTPUT / 'review' / (frame + '_overview.png')
        panel(path, [np.array(Image.fromarray(old).resize((360, 640))),
                     np.array(Image.fromarray(new).resize((360, 640)))],
              ['old / ' + frame, 'candidate / ' + frame], (0, 0, 360, 640))
        paths.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(frame=frame, panels=paths, visual_status='pending'))
    atomic_json(OUTPUT / 'review' / 'receipt.json', dict(records=records,
        request_sha256=sha(OUTPUT / 'request.json'), full_frame_metrics=False,
        comparison_is_different_camera_not_pixelwise_quality_score=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'render', 'panels'])
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'render': render(args.worker, args.workers)
    elif args.action == 'init': initialize()
    else: panels()
