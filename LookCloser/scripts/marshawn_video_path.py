#!/usr/bin/env python3
"""CPU camera-flight and exact-time GT splice contract for Marshawn.

The last neural sample is t=71/24, NOT the endpoint. The endpoint is t=3,
the first actual photograph. Zoom is completed in neural camera intrinsics;
photographic frames use integer translations of a fixed native pixel crop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import CubicSpline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def smooth(value):
    t = np.clip(value, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def native_row(row):
    r = dict(row)
    for key in ['fl_x', 'cx']: r[key] *= 8192 / row['w']
    for key in ['fl_y', 'cy']: r[key] *= 4320 / row['h']
    r.update(w=8192, h=4320)
    return r


def cropped_row(row, crop):
    x, y, w, h = map(int, crop)
    if not (0 <= x <= row['w']-w and 0 <= y <= row['h']-h):
        raise ValueError('Native crop leaves photographic canvas')
    return dict(row, w=w, h=h, cx=row['cx']-x, cy=row['cy']-y)


def bezier(points, t):
    """Convex combination of actual TRAIN centers; no camera-center overshoot."""
    n = len(points)-1
    return sum(math.comb(n,k)*(1-t)**(n-k)*t**k*p for k,p in enumerate(points))


def camera_at(seconds, controls, endpoint_crop, zoom_start=1.4):
    if seconds == 3.:
        return cropped_row(native_row(controls[-1]), endpoint_crop)
    u = float(smooth(seconds/3.))
    centers = np.array([r['transform_matrix'] for r in controls])[:, :3, 3]
    matrices = np.array([r['transform_matrix'] for r in controls])[:, :3, :3]
    rotations = Rotation.from_matrix(matrices)
    rotation = RotationSpline(np.linspace(0,1,len(controls)), rotations)(u).as_matrix()
    # Stored float32 calibration is orthogonal only to ~1e-7. Preserve its
    # endpoint roundoff continuously, avoiding a tiny final hard pose change.
    normalized=rotations.as_matrix()
    rotation+=(1-u)*(matrices[0]-normalized[0])+u*(matrices[-1]-normalized[-1])
    pose = np.eye(4); pose[:3,:3] = rotation; pose[:3,3] = bezier(centers,u)
    # The same smooth clock gives zero velocity and acceleration at both ends.
    row = dict(controls[0], transform_matrix=pose.tolist(), w=endpoint_crop[2], h=endpoint_crop[3])
    z = float(smooth((seconds-zoom_start)/(3.-zoom_start)))
    endpoint = cropped_row(native_row(controls[-1]), endpoint_crop)
    for key in ['fl_x','fl_y','cx','cy']:
        axis = 'w' if key in ['fl_x','cx'] else 'h'
        vals = np.array([r[key]*row[axis]/r[axis] for r in controls])
        base = float(CubicSpline(np.linspace(0,1,len(controls)),vals)(u))
        row[key] = (1-z)*base + z*endpoint[key]
    row['physical_camera'] = 'interpolated_train_flight'
    return row


def raw_file(camera, frame, roots):
    local = frame-1252
    candidates = [Path(r)/camera/f'{camera}.{local:06d}.tif' for r in roots]
    found = [p for p in candidates if p.exists()]
    return str(found[0]) if found else None


def build(args):
    meta = json.loads(args.transforms.read_text())
    rows = {r['physical_camera']:r for r in meta['frames']}
    train = {r['physical_camera'] for r in meta['frames'] if r['file_path'] in meta['train_filenames']}
    names = args.controls
    if not set(names) <= train:
        raise ValueError('Flight controls and endpoint MUST be actual TRAIN cameras: '+str(set(names)-train))
    controls = [rows[n] for n in names]
    crops = json.loads(args.crop_track.read_text()) if args.crop_track else {}
    fixed = list(args.crop)
    output_wh = fixed[2:]
    if output_wh not in [[1920,1012],[3840,2024]]:
        raise ValueError('Use HD1920x1012 or native4K3840x2024 fixed output')
    endpoint_crop = crops.get('2020', fixed)
    records=[]
    for i in range(120):
        t=i/24; frame=1660+i*5; neural=i<72
        crop = crops.get(str(frame), endpoint_crop)
        if list(crop[2:]) != output_wh or any(int(v)!=v for v in crop):
            raise ValueError('Photographic tracks allow integer crop/shift ONLY, never varying sizes')
        row = camera_at(t,controls,endpoint_crop) if neural else cropped_row(native_row(controls[-1]),crop)
        records.append(dict(index=i,time_seconds=t,frame_id=f'{frame:06d}',kind='neural' if neural else 'actual_gt',
            camera=row,gt_native_crop_xywh=None if neural else crop,
            raw_source=None if neural else raw_file(names[-1],frame,args.raw_roots)))
    end = camera_at(3.,controls,endpoint_crop)
    if end != records[72]['camera']: raise AssertionError('Boundary camera mismatch')
    np.testing.assert_array_equal(end['transform_matrix'],controls[-1]['transform_matrix'])
    p=dict(version=1,fps=24,source_fps=120,start_frame=1660,neural_count=72,gt_count=48,
        duration_seconds=5.,neural_duration_seconds=3.,output_wh=output_wh,controls=names,
        endpoint_train_camera=names[-1],endpoint_seconds=3.,endpoint_camera=end,
        source_transforms=str(args.transforms.resolve()),source_transforms_sha256=sha(args.transforms),
        gt_policy='native8192x4320 corrected undistorted image; fixed output-size integer crop+shift; no resize',
        zoom_policy='continuous neural-intrinsics zoom ends EXACTLY at t3 with GTcrop pinhole; GT scale fixed',
        geometry_policy='degreeN TRAIN-center Bezier convex hull; RotationSpline; quintic time warp C2 endpoints',
        future_60fps=dict(source_step=2,neural_frame_ids=[f'{1660+2*i:06d}' for i in range(180)],weights_reuse=False),
        raw_roots=args.raw_roots,missing_gt_frames=[r['frame_id'] for r in records[72:] if not r['raw_source']],
        visual_status='pending_actual_model_render_review',records=records,script_sha256=sha(__file__))
    write(args.output/'path.json',p)
    # A stock centered-FOV-only path cannot encode the translated principal
    # point of a native crop. This companion requires the exact-K renderer.
    write(args.output/'camera_path_intrinsics_exact.json',dict(requires_renderer='marshawn_video_render.py',
        stock_centered_FOV_parser_compatible=False,camera_type='perspective',render_width=output_wh[0],render_height=output_wh[1],
        seconds=3.,fps=24,camera_path=[dict(camera_to_world=np.asarray(r['camera']['transform_matrix']).reshape(-1).tolist(),
            aspect=output_wh[0]/output_wh[1],**{k:r['camera'][k] for k in ['fl_x','fl_y','cx','cy']}) for r in records[:72]]))
    write(args.output/'validation.json',dict(neural_frames=72,GT_frames=48,first_GT_time=records[72]['time_seconds'],
        last_neural_time=records[71]['time_seconds'],endpoint_pose_exact=True,endpoint_intrinsics_exact=True,
        no_eval_camera_controls=True,missing_gt_count=len(p['missing_gt_frames']),
        static_duplicate_GT_allowed=False,neural_71_is_not_endpoint=records[71]['camera']!=end))
    print(json.dumps(dict(path=str(args.output/'path.json'),endpoint=names[-1],missing_gt=len(p['missing_gt_frames']))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transforms',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',nargs='+',default=['DC','EC','FC','GC','HC'])
    p.add_argument('--crop',nargs=4,type=int,default=[3680,920,1920,1012])
    p.add_argument('--crop-track',type=Path)
    p.add_argument('--raw-roots',nargs='+',default=['/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local',
        '/fsx/oregon/gta_marshawn_2_1_tiff16_f1901-2020_20261105/by_local'])
    build(p.parse_args())


if __name__=='__main__': main()
