#!/usr/bin/env python3
"""Forward-only test of whether the existing cord density needs hardening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import score, sha, write


def screen(selection: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Preserve prior screens')
    chosen=json.loads(selection.read_text())
    chosen=chosen.get('selected',chosen)
    trainer,request=load(chosen)
    model=trainer.pipeline.model
    fit_path=Path(model.config.cord_sampling_fit_path)
    if sha(fit_path)!=model.config.cord_sampling_fit_sha256:
        raise ValueError('Cord geometry fit changed')
    start,end=model.cord_sampling_endpoints.to(model.device)
    segment=end-start
    length_sq=(segment*segment).sum()
    before=field_fingerprint(model)
    output.mkdir(parents=True)
    settings=[('native',1.,.003,0.),('x2_r003',2.,.003,0.),
              ('x4_r003',4.,.003,0.),('x8_r003',8.,.003,0.),
              ('x4_r005',4.,.005,0.),('x8_r005',8.,.005,0.),
              ('x4_lower',4.,.003,.55)]
    boxes={95:(850,1200,925,1406),97:(810,1150,900,1406)}
    all_rows=[]
    for camera_id,box in boxes.items():
        ds=(trainer.pipeline.datamanager.eval_dataset if camera_id==95 else
            trainer.pipeline.datamanager.train_dataset)
        index=next(i for i,p in enumerate(ds.image_filenames) if int(p.stem.split('_')[1])==camera_id)
        x0,y0,x1,y1=box
        yy,xx=torch.meshgrid(torch.arange(y0,y1,device=model.device),
                             torch.arange(x0,x1,device=model.device),indexing='ij')
        camera=ds.cameras[index:index+1].to(model.device)
        rays=camera.generate_rays(0,coords=torch.stack([yy,xx],-1).float()+.5)
        gt=ds[index]['image'][y0:y1,x0:x1].to(model.device).float()
        images=[('GT',gt)]
        rows=[]
        for label,factor,radius,lower_fraction in settings:
            count=[0]
            def harden(_module,args,values):
                if factor==1.:
                    return values
                points=args[0].frustums.get_positions().reshape(-1,3)
                fraction=((points-start)*segment).sum(-1)/length_sq
                closest=start+fraction.clamp(0,1)[:,None]*segment
                selected=((points-closest).square().sum(-1)<=radius**2)
                selected &= (fraction>=lower_fraction)&(fraction<=1.)
                density=values[torch_field_density]
                values[torch_field_density]=density*torch.where(
                    selected.reshape(density.shape),factor,1.)
                count[0]+=int(selected.sum())
                return values
            from nerfstudio.field_components.field_heads import FieldHeadNames
            torch_field_density=FieldHeadNames.DENSITY
            hook=model.field.register_forward_hook(harden)
            try:
                with torch.no_grad():
                    rgb=model.get_outputs_for_camera_ray_bundle(rays)['rgb'].float()
            finally:
                hook.remove()
            metrics=score(model,rgb,gt)
            rgb8=np.rint(rgb.cpu().numpy().clip(0,1)*255).astype('uint8')
            Image.fromarray(rgb8).save(output/f'cam{camera_id:03d}_{label}.png')
            images.append((label,rgb))
            rows.append(dict(label=label,factor=factor,radius=radius,
                             lower_fraction=lower_fraction,modified_samples=count[0],**metrics))
        all_rows.append(dict(camera_id=camera_id,box=box,results=rows))
        scale=3
        canvas=Image.new('RGB',(len(images)*(x1-x0)*scale,(y1-y0)*scale+25),'white')
        draw=ImageDraw.Draw(canvas)
        for i,(label,image) in enumerate(images):
            im=Image.fromarray(np.rint(image.cpu().numpy().clip(0,1)*255).astype('uint8'))
            im=im.resize(((x1-x0)*scale,(y1-y0)*scale))
            x=i*im.width
            draw.text((x+3,5),label,fill='black')
            canvas.paste(im,(x,25))
        canvas.save(output/f'cam{camera_id:03d}_comparison.png')
        print(json.dumps(dict(camera_id=camera_id,results=rows)),flush=True)
    after=field_fingerprint(model)
    if before!=after:
        raise ValueError('Forward-only scaling changed field parameters')
    write(output/'result.json',dict(scope='Frozen rendering only; artificial scaling of existing density in fitted cord tube',
          selection=str(selection),checkpoint=chosen['checkpoint'],checkpoint_sha256=sha(chosen['checkpoint']),
          fit_sha256=sha(fit_path),source_sha256=sha(Path(__file__)),
          field_before=before,field_after=after,masks_used=False,results=all_rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    screen(args.selection,args.output)
