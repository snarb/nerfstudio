#!/usr/bin/env python3
"""Render selected existing review-path frames without changing model or camera paths."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from nerfstudio.cameras.cameras import Cameras,CameraType
from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import sha,write


def render(selection,path_file,output,frames):
    if output.exists():raise FileExistsError('Preserve previous path samples')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,_=load(selected);model=trainer.pipeline.model
    path=json.loads(path_file.read_text());intr=path['intrinsics']
    before=field_fingerprint(model);output.mkdir(parents=True)
    with torch.no_grad():
        for frame in frames:
            camera=Cameras(camera_to_worlds=torch.tensor(path['poses'][frame][:3],dtype=torch.float32)[None],
                fx=intr['fx'],fy=intr['fy'],cx=intr['cx'],cy=intr['cy'],width=intr['w'],height=intr['h'],
                camera_type=CameraType.PERSPECTIVE).to(model.device)
            rgb=model.get_outputs_for_camera(camera)['rgb'].float().cpu().numpy()
            if not np.isfinite(rgb).all():raise ValueError('Nonfinite path RGB')
            Image.fromarray(np.rint(rgb.clip(0,1)*255).astype('uint8')).save(output/f'frame_{frame:04d}.png')
    after=field_fingerprint(model)
    if before!=after:raise ValueError('Path renderer changed field')
    write(output/'manifest.json',dict(selection=str(selection),path=str(path_file),path_sha256=sha(path_file),
        source_sha256=sha(Path(__file__)),frames=frames,field_before=before,field_after=after,
        masks_used=False,sampling='Native checkpoint',visual_review='pending'))
    print(json.dumps(dict(output=str(output),frames=frames)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[18,23,25,26])
    args=parser.parse_args();render(args.selection,args.path,args.output,args.frames)
