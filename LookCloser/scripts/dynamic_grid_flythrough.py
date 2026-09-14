"""Opt-in moving-actor, open spline flight with measured two-axis rig travel.

Every rendered instant uses its own chronological mesh and train RGB. The short
capture traverses an open S curve, not a truncated circle or a frozen-actor loop.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline
from central_space_temporal_flythrough import motion_report
from joint_temporal_texture import cameras, read, sha, atomic_json, CALIBRATION, HELD_CAMERAS
from render_smooth_temporal_mesh_video import calibration_pose, verify_request
from render_patchmatch_camera_path import normalize_frame

PARENT = Path('/mnt/data/lookcloser_dec5_5a3_central_space_flight_150')
OUTPUT = Path('/mnt/data/dec5_dynamic_grid_150')


def open_grid_path(rows, center, size=3, count=150, fps=24):
    if size not in (3, 4) or count < 30 or fps <= 0:
        raise ValueError('Invalid dynamic grid options')
    names = ['G004_B005', 'I004_B005', 'I004_D005', 'G004_D005'] if size == 3 else [
        'F004_A005', 'I004_A005', 'I004_D005', 'F004_D005']
    by_name = {r['physical_camera'][:9]: r for r in rows}
    anchors = [by_name[n] for n in names]
    if {r['physical_camera'] for r in anchors} & HELD_CAMERAS:
        raise ValueError('Held-out anchor')
    corners = np.array([r['transform_matrix'] for r in anchors])[:, :3, 3]
    # A gentle nonstraight S, with 96% of BOTH available grid dimensions.
    control = np.array([[.02, .02], [.29, .26], [.71, .74], [.98, .98]])
    spline = CubicSpline(np.linspace(0, 1, 4), control, bc_type='natural')
    parameter = np.linspace(0, 1, 24001)
    uv = spline(parameter)
    def weights(q):
        u, v = q.T
        return np.column_stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v))
    dense = weights(uv) @ corners
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    q = spline(np.interp(np.linspace(0, distance[-1], count), distance, parameter))
    ww = weights(q)
    if np.min(ww) < 0 or not np.allclose(ww.sum(1), 1):
        raise ValueError('Path outside train hull')
    extent = np.ptp(q, axis=0) * (size-1)
    if (extent < .95*(size-1)).any():
        raise ValueError('Insufficient actual two-axis travel')
    reference = next(r for r in rows if r['physical_camera'] == 'H004_C005_1210SZ')
    ref = np.array(reference['transform_matrix'])
    target = ref[:3, 3] - ref[:3, 2] * ((ref[:3, 3]-center) @ ref[:3, 2])
    result = []
    for i, pos in enumerate(ww @ corners):
        z = pos-target; z /= np.linalg.norm(z)
        x = np.cross(ref[:3, 1], z); x /= np.linalg.norm(x)
        pose = np.eye(4); pose[:3, :3] = np.column_stack((x, np.cross(z, x), z)); pose[:3, 3] = pos
        row = deepcopy(reference); row.pop('file_path', None)
        row.update(transform_matrix=pose.tolist(), convex_weights=ww[i].tolist(), physical_camera=f'dynamic_grid_{i:05d}')
        result.append(row)
    motion = motion_report([r['transform_matrix'] for r in result], fps)
    if motion['speed_max_min_ratio'] > 1.002 or motion['consecutive_velocity_direction_change_max_degrees'] > 2.5:
        raise ValueError('Jerky path')
    if motion['angular_speed_degrees_per_second_min_median_max'][-1] > 6:
        raise ValueError('Excessive angular speed')
    return result, dict(frames=count, fps=fps, duration_seconds=count/fps, grid_size=size,
        anchors=[r['physical_camera'] for r in anchors], achieved_grid_interval_extent_xy=extent.tolist(),
        minimum_convex_weight=float(ww.min()), continuous_periodic_loop=False, inside_train_hull=True,
        fixed_intrinsics=True, fixed_optical_target=target.tolist(), raw_calibration_motion=motion)


def initialize(output, size=3, fps=24, guard=False):
    previous = verify_request(PARENT)
    rows, mesh, meta = cameras('000973'); cal = read(CALIBRATION)
    center = np.asarray(o3d.io.read_triangle_mesh(str(mesh)).vertices).mean(0)
    path, report = open_grid_path(rows, center, size, len(previous['inventory']), fps)
    raw = [calibration_pose(p, cal, read(meta)) for p in path]
    request = deepcopy(previous); request.pop('composition', None)
    request['parent_geometry_request'] = {'path': str(PARENT/'request.json'), 'sha256': sha(PARENT/'request.json')}
    request['recipe'].update(fps=fps, camera_path_kind='open_cubic_spline_grid', camera_periodic=False,
        grid_size=size, source_time_count=150, temporal_frame_repetition=False, train_foreground_guard=guard)
    for i, record in enumerate(request['inventory']):
        if sha(record['mesh']) != record['mesh_sha256'] or sha(record['metadata']) != record['metadata_sha256']:
            raise ValueError('Reusable geometry changed')
        record['camera'] = normalize_frame(raw[i], cal, read(record['metadata']))
    ids = [r['frame_id'] for r in request['inventory']]
    if len(set(ids)) != 150 or ids != sorted(ids) or ids != request['ordered_frame_ids']:
        raise ValueError('A dynamic flight must have 150 distinct chronological source times')
    request['camera_path_report'] = report
    request['script_hashes'][Path(__file__).name] = sha(__file__)
    request['script_hashes']['central_space_temporal_flythrough.py'] = sha(Path(__file__).with_name('central_space_temporal_flythrough.py'))
    if guard:
        request['script_hashes']['train_foreground_guard.py'] = sha(Path(__file__).with_name('train_foreground_guard.py'))
    output.mkdir(parents=True, exist_ok=True); (output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json') != request:
        raise ValueError('Immutable dynamic request mismatch')
    atomic_json(output/'request.json', request)
    atomic_json(output/'progress.json', {'stage': 'initialized', 'unique_source_times': len(ids), 'complete': 0})
    print(report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['init', 'canary'])
    p.add_argument('--output', type=Path, default=OUTPUT); p.add_argument('--grid', type=int, default=3)
    p.add_argument('--fps', type=int, default=24); p.add_argument('--guard', action='store_true')
    p.add_argument('--frames', nargs='+', default=['000899', '000971', '000973', '001047', '001197']); a=p.parse_args()
    if a.action == 'init': initialize(a.output, a.grid, a.fps, a.guard)
    else:
        import render_smooth_temporal_mesh_video as renderer
        from temporal_texture_view_prior import install
        renderer.torch.set_num_threads(4); install(renderer)
        if read(a.output/'request.json')['recipe'].get('train_foreground_guard'):
            from train_foreground_guard import install as install_guard
            install_guard(renderer)
        renderer.render(a.output, a.frames)
