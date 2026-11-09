#!/usr/bin/env python3
"""Freeze a 96-pose calibrated static-time arc without reading target RGB."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dec5_hair import TEST
from run_dec5_hair_campaign import atomic_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root/'path.json'
    if output.exists():
        raise FileExistsError('Camera path is frozen')
    dataset = json.loads((args.root/'dataset/transforms.json').read_text())
    frames = [f for f in dataset['frames'] if f['file_path'] in dataset['train_filenames']]
    reserved = json.loads((args.root/'reserved_test.json').read_text())['records']
    reference = next(r['frame'] for r in reserved if r['frame']['physical_camera'] == TEST[0])
    pose = np.array(reference['transform_matrix'])
    centers = np.array([f['transform_matrix'] for f in frames])[:, :3, 3]
    offset = centers-pose[:3, 3]
    side = offset @ pose[:3, 0]
    distance = np.linalg.norm(offset, axis=-1)
    left = int(np.argmin(np.where(side < 0, distance, np.inf)))
    right = int(np.argmin(np.where(side > 0, distance, np.inf)))
    anchors = [np.asarray(frames[i]['transform_matrix']) for i in (left, right)]
    focus = np.array(json.loads((args.root/'tsdf/mesh.json').read_text())['bounds']).mean(0)
    vectors = np.stack([p[:3, 3]-focus for p in anchors])
    radius = np.linalg.norm(vectors, axis=-1)
    directions = vectors/radius[:, None]
    angle = np.arccos(np.clip(directions[0] @ directions[1], -1, 1))
    if not 0 < angle < np.pi/2:
        raise ValueError('No supported short camera arc')
    rotations = Slerp([0., 1.], Rotation.from_matrix(np.stack([p[:3, :3] for p in anchors])))
    rows = []
    for index, t in enumerate(np.linspace(0, 1, 96)):
        direction = (np.sin((1-t)*angle)*directions[0]+np.sin(t*angle)*directions[1])/np.sin(angle)
        matrix = np.eye(4)
        matrix[:3, :3] = rotations([t]).as_matrix()[0]
        matrix[:3, 3] = focus+direction*((1-t)*radius[0]+t*radius[1])
        row = deepcopy(reference)
        row.update(physical_camera='novel_%04d'%index, transform_matrix=matrix.tolist())
        row.pop('file_path', None)
        rows.append(row)
    atomic_json(output, dict(frames=rows, frame='001083', fps=24, duration_seconds=4.,
        anchors=[frames[i]['physical_camera'] for i in (left, right)], reference_intrinsics_camera=TEST[0],
        camera_arc_degrees=float(np.rad2deg(angle)), target_rgb_read=False,
        static_time=True, source_sha256=sha(args.root/'dataset/transforms.json'),
        focus_train_geometry_sha256=sha(args.root/'tsdf/mesh.json')))
    print('frozen poses=96 arc_degrees='+str(round(float(np.rad2deg(angle)), 3)), flush=True)


if __name__ == '__main__':
    main()
