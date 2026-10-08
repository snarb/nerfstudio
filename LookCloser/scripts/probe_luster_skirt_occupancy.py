#!/usr/bin/env python3
"""Read-only before/after occupancy dilation on fixed skirt-artifact orbit views."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import sha, write


def probe(selection, export, output):
    if output.exists():
        raise FileExistsError('Preserve previous probes')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,_=load(selected);model=trainer.pipeline.model
    manifest=json.loads((export/'orbit/path.json').read_text())
    intrinsics=manifest['intrinsics']
    original_radius=model.config.occupancy_eval_dilation_radius
    before=field_fingerprint(model)
    output.mkdir(parents=True)
    sheet=Image.new('RGB',(660,355*4),'#222');draw=ImageDraw.Draw(sheet)
    rows=[]
    try:
        with torch.no_grad():
            for row,index in enumerate([18,23,25,26]):
                camera=Cameras(camera_to_worlds=torch.tensor(manifest['poses'][index][:3],dtype=torch.float32)[None],
                    fx=intrinsics['fx'],fy=intrinsics['fy'],cx=intrinsics['cx'],cy=intrinsics['cy'],
                    width=intrinsics['w'],height=intrinsics['h'],camera_type=CameraType.PERSPECTIVE).to(model.device)
                arrays=[]
                for radius in [0,1]:
                    model.train();model.config.occupancy_eval_dilation_radius=radius;model.eval()
                    rgb=model.get_outputs_for_camera(camera)['rgb'].float().cpu().numpy()
                    if not np.isfinite(rgb).all():raise ValueError('Nonfinite RGB')
                    arrays.append(np.rint(rgb.clip(0,1)*255).astype('uint8'))
                    Image.fromarray(arrays[-1]).save(output/f'orbit_{index:02d}_radius{radius}.png')
                saved=np.asarray(Image.open(export/'orbit'/f'frame_{index:04d}.png').convert('RGB'))
                if not np.array_equal(arrays[0],saved):raise ValueError('Reconstructed path differs from original native render')
                crops=[a[300:630,130:350] for a in arrays]
                delta=np.abs(crops[0].astype(np.int16)-crops[1].astype(np.int16))
                for col,a in enumerate(crops+[np.clip(delta*8,0,255).astype('uint8')]):
                    sheet.paste(Image.fromarray(a),(220*col,row*355+22))
                draw.text((5,row*355+3),f'Orbit{index}: native | radius1 | difference x8',fill='white')
                rows.append(dict(frame=index,original_png_identical=True,crop_rgb_mae=float(delta.mean()),
                                 changed_fraction=float((delta.max(-1)>0).mean())))
    finally:
        model.train();model.config.occupancy_eval_dilation_radius=original_radius;model.eval()
    after=field_fingerprint(model)
    if before!=after:raise ValueError('Read-only probe changed field')
    sheet.save(output/'contact.jpg')
    write(output/'manifest.json',dict(selection=str(selection),source_sha256=sha(Path(__file__)),
          field_before=before,field_after=after,frames=rows,
          scope='Frozen-field temporary occupancy dilation; no new training or depth on novel views',visual_review='pending'))
    print(json.dumps(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();probe(args.selection,args.export,args.output)
