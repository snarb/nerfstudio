#!/usr/bin/env python3
"""Gauge-align a train-only COLMAP screen and export an isolated scene root."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nerfstudio.data.utils.colmap_parsing_utils import read_model, qvec2rotmat
from dec5_hair import cv_pose, VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def align_similarity(source, target):
    source, target = np.asarray(source), np.asarray(target)
    x, y = source-source.mean(0), target-target.mean(0)
    u, singular, vt = np.linalg.svd(y.T@x/len(x))
    sign = np.ones(3); sign[-1] = np.linalg.det(u@vt)
    rotation = u@np.diag(sign)@vt
    scale = (singular*sign).sum()/np.square(x).sum(1).mean()
    translation = target.mean(0)-scale*rotation@source.mean(0)
    return scale, rotation, translation


def export(root, model, output):
    preparation = json.loads((root/'prepare.json').read_text())
    cameras, images, points = read_model(model, ext='.bin')
    by = {image.name: image for image in images.values()}
    payload = json.loads((root/'dataset/transforms.json').read_text())
    train = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(train) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in train): raise ValueError('Requires train59')
    if set(by) != {f['file_path'] for f in train}: raise ValueError('SfM did not retain the exact train split')
    poses = {}
    for frame in train:
        image = by[frame['file_path']]; camera = cameras[image.camera_id]
        if camera.model != 'PINHOLE': raise ValueError('Requires undistorted PINHOLE')
        np.testing.assert_allclose(camera.params, [frame[k] for k in ('fl_x','fl_y','cx','cy')], rtol=0, atol=1e-8)
        pose = np.eye(4); pose[:3,:3] = qvec2rotmat(image.qvec).T; pose[:3,3] = -pose[:3,:3]@image.tvec
        poses[frame['file_path']] = pose
    source = np.stack([poses[f['file_path']][:3,3] for f in train])
    target = np.stack([cv_pose(f)[:3,3] for f in train])
    scale, rotation, translation = align_similarity(source, target)
    result = deepcopy(payload); shifts = []; rotations = []
    for frame in result['frames']:
        if frame['file_path'] not in poses: continue
        pose = poses[frame['file_path']].copy()
        pose[:3,:3] = rotation@pose[:3,:3]
        pose[:3,3] = scale*rotation@pose[:3,3]+translation
        original = cv_pose(frame)
        shifts.append(np.linalg.norm(pose[:3,3]-original[:3,3]))
        rotations.append(Rotation.from_matrix(original[:3,:3].T@pose[:3,:3]).magnitude()*180/np.pi)
        pose[:3,1:3] *= -1
        frame['transform_matrix'] = pose.tolist()
    for old, new in zip(payload['frames'], result['frames']):
        if old['file_path'] not in payload['train_filenames'] and old != new: raise ValueError('Held-out camera changed')
    output.mkdir(parents=True, exist_ok=False)
    (output/'dataset').mkdir(); (output/'dataset/images').symlink_to((root/'dataset/images').resolve(), target_is_directory=True)
    atomic_json(output/'dataset/transforms.json', result)
    for name in ('semantics','source','matting_v3','validation_reference','neural_validation'):
        if (root/name).exists(): (output/name).symlink_to((root/name).resolve(), target_is_directory=True)
    preparation['dataset_transforms_sha256'] = sha(output/'dataset/transforms.json')
    preparation['calibration_screen'] = str(model)
    atomic_json(output/'prepare.json', preparation)
    atomic_json(output/'calibration_receipt.json', dict(train_cameras=59, fitting_points=len(points),
        gauge_alignment='least-squares similarity over all 59 original training camera centers',
        similarity=dict(scale=float(scale), rotation=rotation.tolist(), translation=translation.tolist()),
        camera_center_shift_quantiles_50_90_max=np.quantile(shifts,[.5,.9,1]).tolist(),
        camera_rotation_degrees_quantiles_50_90_max=np.quantile(rotations,[.5,.9,1]).tolist(),
        held_out_cameras_unchanged=True, intrinsics_fixed=True, original_dataset_sha256=sha(root/'dataset/transforms.json'),
        output_dataset_sha256=sha(output/'dataset/transforms.json'), implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','model','output'): p.add_argument('--'+key, type=Path, required=True)
    a=p.parse_args(); export(a.root,a.model,a.output)


if __name__=='__main__': main()
