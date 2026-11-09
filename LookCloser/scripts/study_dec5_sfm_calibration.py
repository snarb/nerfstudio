#!/usr/bin/env python3
"""Bounded train59 pose-only SfM screen; never edits the original calibration.

Rebuilds observations from current RGB, without inherited sparse points. The
result must still pass independent correspondence and held-out rendering gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sqlite3
import time

import numpy as np

from audit_dec5_calibration import calibrate_database
from dec5_hair import cv_pose, VALIDATION, TEST
from export_nerfstudio_colmap_model import export_model
from run_dec5_hair_campaign import atomic_json, sha


def bind_database_ids(output):
    """COLMAP's clear_points remapping is insufficient for this rig-aware build."""
    with sqlite3.connect(str(output/'database.db')) as database:
        ids = {name: (i, c) for i, c, name in database.execute('SELECT image_id,camera_id,name FROM images')}
    manifest = json.loads((output/'fixed_model/export_manifest.json').read_text())
    by_old = {r['image_id']: ids[r['name']] for r in manifest['images']}
    if len(ids) != 59 or len(by_old) != 59: raise ValueError('Database/model split mismatch')
    dest = output/'database_bound_model'; dest.mkdir(exist_ok=False)
    cameras = []
    for line in (output/'fixed_model/cameras.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip(): continue
        fields = line.split(); fields[0] = str(by_old[int(fields[0])][1]); cameras.append(' '.join(fields))
    images = []
    lines = (output/'fixed_model/images.txt').read_text().splitlines()
    for line in lines:
        if line.startswith('#') or not line.strip(): continue
        fields = line.split(); old = int(fields[0]); fields[0], fields[8] = map(str, by_old[old])
        images.extend([' '.join(fields), ''])
    (dest/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (dest/'images.txt').write_text('\n'.join(images)+'\n')
    (dest/'points3D.txt').write_text('# Empty initial points\n')
    atomic_json(dest/'binding.json', dict(images=ids, inherited_points=False))
    return dest


def run(root, output, colmap, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    data = root/'dataset'
    payload = json.loads((data/'transforms.json').read_text())
    frames = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):
        raise ValueError('Expected frozen train59')
    if resume:
        previous = json.loads((output/'prepare.json').read_text())
        if previous['dataset_sha256'] != sha(data/'transforms.json'): raise ValueError('Resume data mismatch')
    else:
        atomic_json(output/'prepare.json', dict(frames=frames, dataset_sha256=sha(data/'transforms.json'),
            train_only=True, inherited_sparse_points=False, implementation_sha256=sha(__file__)))
        export_model(data, output/'fixed_model', split='train', camera_model='PINHOLE')
    (output/'train_images.txt').write_text('\n'.join(f['file_path'] for f in frames)+'\n')
    centers = np.stack([cv_pose(f)[:3, 3] for f in frames]); pairs = set()
    for i in range(len(frames)):
        for j in np.argsort(np.linalg.norm(centers-centers[i], axis=-1))[1:13]:
            pairs.add(tuple(sorted((frames[i]['file_path'], frames[j]['file_path']))))
    (output/'pairs.txt').write_text('\n'.join(a+' '+b for a,b in sorted(pairs))+'\n')
    receipts = json.loads((output/'stages.json').read_text()) if resume else []
    def execute(name, arguments):
        start = time.monotonic(); command = [str(colmap)]+arguments
        with (output/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = dict(stage=name, command=command, returncode=result.returncode, seconds=time.monotonic()-start)
        receipts.append(row); atomic_json(output/'stages.json', receipts)
        if result.returncode: raise RuntimeError(name+' failed; see log')
        print('SfM stage '+name+' complete %.1fs'%row['seconds'], flush=True)
    database = str(output/'database.db')
    if not resume:
        execute('features', ['feature_extractor', '--database_path', database, '--image_path', str(data),
            '--image_list_path', str(output/'train_images.txt'), '--ImageReader.camera_model', 'PINHOLE',
            '--SiftExtraction.use_gpu', '1', '--SiftExtraction.num_threads', '2', '--SiftExtraction.max_num_features', '8192'])
        calibrate_database(output)
        execute('matching', ['matches_importer', '--database_path', database, '--match_list_path', str(output/'pairs.txt'),
            '--match_type', 'pairs', '--SiftMatching.use_gpu', '1', '--SiftMatching.num_threads', '2'])
    bound = bind_database_ids(output)
    (output/'triangulated_bound').mkdir()
    execute('triangulation', ['point_triangulator', '--database_path', database, '--image_path', str(data),
        '--input_path', str(bound), '--output_path', str(output/'triangulated_bound'),
        '--Mapper.num_threads', '2', '--Mapper.fix_existing_frames', '1',
        '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_extra_params', '0',
        '--Mapper.filter_max_reproj_error', '8', '--Mapper.tri_complete_max_reproj_error', '8',
        '--Mapper.tri_merge_max_reproj_error', '8'])
    (output/'optimized').mkdir()
    execute('pose_bundle', ['bundle_adjuster', '--input_path', str(output/'triangulated_bound'),
        '--output_path', str(output/'optimized'), '--BundleAdjustment.refine_focal_length', '0',
        '--BundleAdjustment.refine_principal_point', '0', '--BundleAdjustment.refine_extra_params', '0',
        '--BundleAdjustment.max_num_iterations', '100', '--BundleAdjustment.use_gpu', '0'])
    atomic_json(output/'complete.json', dict(stages=receipts, uses_validation_rgb=False, uses_test_rgb=False,
        calibration_optimized=True, intrinsics_fixed=True, original_dataset_unchanged=True,
        gauge_alignment_and_independent_evaluation='pending'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--colmap', type=Path, default=Path('/usr/local/bin/colmap'))
    p.add_argument('--resume-optimization', action='store_true')
    a = p.parse_args(); run(a.root, a.output, a.colmap, a.resume_optimization)


if __name__ == '__main__': main()
