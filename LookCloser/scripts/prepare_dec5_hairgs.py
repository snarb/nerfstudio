#!/usr/bin/env python3
"""Stage split-clean HairGS inputs and explicitly gate missing scalp geometry."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from dec5_hair import orientations, unproject
from export_nerfstudio_colmap_model import export_model
from run_dec5_hair_campaign import atomic_json, sha
from study_dec5_hair_geometry import Scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--head-reconstruction', type=Path)
    p.add_argument('--mvs', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = Scene(args.root, load_rgb=False, mvs=args.mvs)
    export_model(scene.data, args.output/'sparse/0', split='train', camera_model='PINHOLE')
    for folder in ('images', 'masks', 'orientations'):
        (args.output/folder).mkdir()
    points, colors = [], []
    for camera, frame, depth, semantic in zip(scene.cameras, scene.frames, scene.depth, scene.semantic):
        source = scene.data/frame['file_path']
        name = source.stem
        shutil.copy2(source, args.output/'images'/source.name)
        image = cv2.imread(str(source))[..., ::-1]
        full_semantic = np.load(args.root/'semantics'/(frame['physical_camera']+'.npz'))['probability'][..., 0]/65535
        o, q = orientations(image.astype(np.float32)/255)
        angle = np.mod(.5*np.arctan2(o[..., 1], o[..., 0]), np.pi)
        cv2.imwrite(str(args.output/'masks'/(name+'.png')), (full_semantic > .5).astype(np.uint8)*255)
        cv2.imwrite(str(args.output/'orientations'/(name+'_orientation.png')), np.rint(angle*255/np.pi).astype(np.uint8))
        cv2.imwrite(str(args.output/'orientations'/(name+'_confidence.png')), np.rint(q*full_semantic*255).astype(np.uint8))
        yy, xx = np.nonzero((depth > 0) & (semantic[..., 0] > .8))
        yy, xx = yy[::32], xx[::32]
        points.append(unproject(np.stack((xx+.5, yy+.5), -1), depth[yy, xx], camera))
        native_x = np.clip(np.rint((xx+.5)*frame['w']/camera['w']-.5).astype(int), 0, frame['w']-1)
        native_y = np.clip(np.rint((yy+.5)*frame['h']/camera['h']-.5).astype(int), 0, frame['h']-1)
        colors.append(image[native_y, native_x])
    points, colors = np.concatenate(points), np.concatenate(colors)
    with (args.output/'sparse/0/points3D.txt').open('w') as stream:
        stream.write('# Measured train-only MVS initialization; no SfM tracks or held-out RGB.\n')
        for index, (point, rgb) in enumerate(zip(points, colors), 1):
            stream.write('%d %.9g %.9g %.9g %d %d %d 0\n'%((index,)+tuple(point)+tuple(rgb)))
    reason = []
    if args.head_reconstruction is not None:
        head = np.load(args.head_reconstruction, allow_pickle=False)
        for key in ('head_verts', 'scalp_verts'):
            if key not in head or head[key].ndim != 2 or head[key].shape[1] != 3 or not np.isfinite(head[key]).all():
                raise ValueError('Invalid head/scalp geometry')
        shutil.copy2(args.head_reconstruction, args.output/'head_reconstruction_data.npz')
    else:
        reason.append('No calibrated head_verts/scalp_verts supplied; the hair TSDF is not a certified scalp.')
    dependencies = {name:importlib.util.find_spec(name) is not None for name in ('pytorch3d','pyvista')}
    if not all(dependencies.values()):
        reason.append('Missing isolated HairGS training dependencies; no build attempted before scalp qualification.')
    atomic_json(args.output/'preflight.json', dict(ready_for_training=not reason, reasons=reason,
        cameras=59, initialization_points=len(points), upstream_commit='16588656b1f6f048bc3bc83f3cb98c2da8596754',
        dataset_sha256=sha(scene.data/'transforms.json'), orientation_encoding='theta modulo pi, uint8 theta*255/pi',
        source_mvs_request_sha256=sha(scene.mvs/'request.json'),
        uses_eval_rgb=False, gt_strands_required_for_inference=False, dependencies=dependencies,
        upstream_compatibility_work=['np.bool removed in modern NumPy', 'final evaluation guard must respect absent strand GT'],
        gaussian_geometry_used_in_track_N=False, training_launched=False))
    print('HairGS staged cameras=59 ready='+str(not reason), flush=True)


if __name__ == '__main__':
    main()
