#!/usr/bin/env python3
"""Opt-in fixed mesh-space hard-source texture canary, with global color leveling.

No Poisson seam blending, texture hole synthesis, semantic masks or held-out RGB.
Requires the separately built, pinned official mvs-texturing texrecon executable.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from render_patchmatch_camera_path import normalize_frame, ANCHORS

PIN = 'f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1'


def mve_camera(frame):
    """Return world-to-OpenCV extrinsic and exact MVE normalized intrinsics."""
    pose = np.asarray(frame['transform_matrix'], dtype=float)
    extrinsic = np.linalg.inv(pose @ np.diag([1., -1., -1., 1.]))
    w,h,fx,fy = (float(frame[k]) for k in ('w','h','fl_x','fl_y'))
    if min(w,h,fx,fy) <= 0 or not np.isfinite(extrinsic).all():
        raise ValueError('Invalid camera')
    aspect = fy/fx
    focal = fy/h if w/h*aspect < 1 else fx/w
    intrinsic = [focal, 0., 0., aspect, frame['cx']/w, frame['cy']/h]
    return extrinsic, intrinsic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('data','mesh','mesh-metadata','output','texrecon','texrecon-repo'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--camera-color-calibration',type=Path)
    p.add_argument('--skip-global-seam-leveling',action='store_true')
    p.add_argument('--labeling',type=Path,help='Reuse identical source labels for a matched leveling control')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--outlier-removal',choices=('none','gauss_damping','gauss_clamping'),default='none')
    a=p.parse_args()
    if a.mesh.suffix.lower()!='.ply':p.error('Expected extracted TSDF .ply mesh')
    def git(path,*args):
        return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
    if git(a.texrecon_repo,'rev-parse','HEAD')!=PIN:
        raise ValueError('Unverified mvs-texturing source revision')
    if git(a.texrecon_repo,'diff','--name-only','HEAD'):
        raise ValueError('Modified mvs-texturing source')
    payload=json.loads((a.data/'transforms.json').read_text())
    metadata=json.loads(a.mesh_metadata.read_text())
    names=set(payload['train_filenames'])
    frames=[normalize_frame(f,payload,metadata) for f in payload['frames'] if f['file_path'] in names]
    if len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62:
        raise ValueError('Expected 62 distinct train cameras')
    if set(ANCHORS)&{f['physical_camera'] for f in frames} or any(f.get('mask_path') for f in frames):
        raise ValueError('Held-out RGB or masks in source cameras')
    calibration=None
    if a.camera_color_calibration:
        calibration=json.loads(a.camera_color_calibration.read_text())
        if calibration['uses_eval_rgb'] is not False or calibration['domain']!='inverse_srgb_inverse_reinhard_exposed_linear':
            raise ValueError('Invalid train-only exposure calibration')
    a.output.mkdir(parents=True,exist_ok=False)
    scene=a.output/'scene';scene.mkdir()
    rows=[]
    for i,f in enumerate(frames):
        source=(a.data/f['file_path']).resolve(strict=True)
        digest=sha256(source)
        stem=f'{i:03d}_{f["physical_camera"]}'
        gain=None
        if calibration:
            import torch
            from patchmatch_color_calibration import apply_camera_gain
            row=calibration['cameras'][f['physical_camera']]
            if row['image_sha256']!=digest:raise ValueError('JPEG changed since calibration')
            gain=row['exposure_gain']
            rgb=torch.tensor(np.asarray(Image.open(source).convert('RGB')).copy()).permute(2,0,1).float()/255
            corrected=apply_camera_gain(rgb,gain).permute(1,2,0).numpy()
            destination=scene/(stem+'.png')
            Image.fromarray(np.rint(corrected*255).astype(np.uint8)).save(destination)
        else:
            destination=scene/(stem+source.suffix)
            shutil.copyfile(source,destination)
        ext,intr=mve_camera(f)
        cam=scene/(stem+'.cam')
        cam.write_text(' '.join(f'{v:.17g}' for v in np.r_[ext[:3,3],ext[:3,:3].ravel()])+'\n'+
                       ' '.join(f'{v:.17g}' for v in intr)+'\n')
        rows.append({'physical_camera':f['physical_camera'],'source':str(source),'source_sha256':digest,
                     'scene_image':str(destination),'scene_sha256':sha256(destination),'camera_sha256':sha256(cam),'gain':gain})
    target=a.output/'textured';target.mkdir()
    command=[str(a.texrecon.resolve()),'--skip_local_seam_leveling','--skip_hole_filling','--keep_unseen_faces',
             '--write_timings',f'--num_threads={a.threads}','-o',a.outlier_removal]
    if a.skip_global_seam_leveling:command.append('--skip_global_seam_leveling')
    if a.labeling:command.extend(['-L',str(a.labeling.resolve())])
    command.extend([str(scene.resolve()),str(a.mesh.resolve()),str((target/'mesh').resolve())])
    request={'schema_version':1,'method':'fixed_per_triangle_source_mvs_global_color_leveling',
             'texrecon_commit':PIN,'texrecon_binary_sha256':sha256(a.texrecon),
             'dependency_commits':{n:git(a.texrecon_repo/'elibs'/n,'rev-parse','HEAD') for n in ('mve','mapmap','rayint')},
             'mesh_sha256':sha256(a.mesh),'mesh_metadata_sha256':sha256(a.mesh_metadata),
             'data_sha256':sha256(a.data/'transforms.json'),'script_sha256':sha256(Path(__file__)),
             'normalization_helper_sha256':sha256(Path(__file__).with_name('render_patchmatch_camera_path.py')),
             'camera_color_calibration_sha256':sha256(a.camera_color_calibration) if calibration else None,
             'labeling_sha256':sha256(a.labeling) if a.labeling else None,
             'command':command,'sources':rows,'uses_eval_rgb':False,'semantic_masks':False,
             'source_averaging':False,'local_poisson_leveling':False,'texture_hole_filling':False,
             'global_seam_leveling':not a.skip_global_seam_leveling}
    atomic_json(a.output/'request.json',request)
    with (a.output/'texrecon.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OMP_NUM_THREADS=str(a.threads)))
    if not (target/'mesh.obj').is_file():raise RuntimeError('Missing textured mesh')
    atomic_json(a.output/'result.json',{'state':'textured_pending_visual_review','request_sha256':sha256(a.output/'request.json'),
                'artifacts':{str(f.relative_to(a.output)):sha256(f) for f in sorted(target.iterdir()) if f.is_file()}})
    print(f'Textured mesh completed: {target}',flush=True)


if __name__=='__main__':main()
