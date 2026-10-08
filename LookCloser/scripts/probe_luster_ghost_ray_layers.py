#!/usr/bin/env python3
"""Inspect color/termination layers behind skirt ghosts on fixed novel-view rays."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType

from export_luster_opacity import load
from run_luster_opacity import sha, write


def probe(selection, path_file, output):
    if output.exists():raise FileExistsError('Preserve prior ray-layer probes')
    selected=json.loads(selection.read_text());selected=selected.get('selected',selected)
    trainer,_=load(selected);model=trainer.pipeline.model
    path=json.loads(path_file.read_text());intrinsics=path['intrinsics']
    camera=Cameras(camera_to_worlds=torch.tensor(path['poses'][23][:3],dtype=torch.float32)[None],
                   fx=intrinsics['fx'],fy=intrinsics['fy'],cx=intrinsics['cx'],cy=intrinsics['cy'],
                   width=intrinsics['w'],height=intrinsics['h'],camera_type=CameraType.PERSPECTIVE).to(model.device)
    xy=np.array([[190,480],[150,515],[190,380],[245,480],[250,520]],dtype=np.int64)
    with torch.no_grad():
        rays=camera.generate_rays(0,coords=torch.tensor(xy[:,::-1].copy(),device=model.device)+.5)
        result=model(rays)
        indices=result['packed_ray_indices'].reshape(-1).long()
        start,end=[result[k].reshape(-1) for k in ['packed_spacing_starts','packed_spacing_ends']]
        near,far=rays.nears.reshape(-1),rays.fars.reshape(-1)
        t=near[indices]+(far-near)[indices]*(start+end)*.5
        weights=result['packed_weights'].reshape(-1).float()
        points=rays.origins[indices]+rays.directions[indices]*t[:,None]
        density,colors=model.field.query_points(points,rays.directions[indices])
        colors=colors.float()
        reconstruction=torch.zeros_like(result['rgb']).index_add(0,indices,weights[:,None]*colors)
        error=float((reconstruction-result['rgb']).abs().max())
        if error>2e-4:raise ValueError(f'Point-color reconstruction differs from native renderer: {error}')
        rows=[]
        for i,pixel in enumerate(xy):
            take=indices==i
            tt,ww,ss,cc=[v[take].cpu().numpy() for v in [t,weights,density.reshape(-1),colors]]
            active=np.flatnonzero(ww>1e-5)
            groups=np.split(active,np.flatnonzero(np.diff(tt[active])>.03)+1) if len(active) else []
            clusters=[]
            for ids in groups:
                mass=float(ww[ids].sum())
                clusters.append(dict(t_min=float(tt[ids].min()),t_max=float(tt[ids].max()),
                    mass=mass,conditional_rgb=((ww[ids,None]*cc[ids]).sum(0)/max(mass,1e-8)).tolist()))
            rows.append(dict(xy=pixel.tolist(),rgb=result['rgb'][i].cpu().tolist(),
                 accumulation=float(result['accumulation'][i]),clusters=clusters,
                 samples=[dict(t=float(a),weight=float(b),sigma=float(c),rgb=d.tolist())
                          for a,b,c,d in zip(tt,ww,ss,cc)]))
    write(output,dict(selection=str(selection),frame=23,path=str(path_file),source_sha256=sha(Path(__file__)),
          native_color_reconstruction_max_error=error,clusters_rule='Weights>1e-5, split on gaps>.03 normalized scene units; diagnostic only',rays=rows))
    print(json.dumps([dict(xy=x['xy'],rgb=x['rgb'],accumulation=x['accumulation'],clusters=x['clusters']) for x in rows]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();probe(args.selection,args.path,args.output)
