#!/usr/bin/env python3
"""Cache train-only learned termination points for an opaque surface warm start."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import sha, write


def build(selection, output, grid=32, maximum_width=.02):
    if output.exists():
        raise FileExistsError('Preserve previous geometry caches')
    selected=json.loads(selection.read_text())
    selected=selected.get('selected',selected)
    trainer,request=load(selected)
    model=trainer.pipeline.model
    if getattr(model.config,'representation','volume')!='volume':
        raise ValueError('Geometry teacher must be a learned volume checkpoint')
    dataset=trainer.pipeline.datamanager.train_dataset
    records=[]
    counts=[]
    with torch.no_grad():
        for camera_index,path in enumerate(dataset.image_filenames):
            camera=dataset.cameras[camera_index:camera_index+1].to(model.device)
            height,width=int(camera.height.item()),int(camera.width.item())
            x,y=np.meshgrid(np.linspace(0,width-1,grid).astype(int),np.linspace(0,height-1,grid).astype(int))
            coords=np.stack([y.ravel(),x.ravel()],-1)
            rays=camera.generate_rays(0,coords=torch.tensor(coords,device=model.device).float()+.5)
            result=model(rays)
            indices=result['packed_ray_indices'].cpu().numpy()
            if np.any(indices[1:]<indices[:-1]):
                raise ValueError('Packed samples must be ordered along each ray')
            weights=result['packed_weights'].flatten().cpu().numpy()
            starts=result['packed_spacing_starts'].flatten().cpu().numpy()
            ends=result['packed_spacing_ends'].flatten().cpu().numpy()
            near,far=rays.nears.flatten().cpu().numpy(),rays.fars.flatten().cpu().numpy()
            mid=near[indices]+(far-near)[indices]*(starts+ends)/2
            delta=(far-near)[indices]*(ends-starts)
            boundaries=np.concatenate([[0],np.bincount(indices,minlength=len(coords)).cumsum()])
            origins=rays.origins.cpu().numpy();directions=rays.directions.cpu().numpy()
            accepted=0
            for i in range(len(coords)):
                lo,hi=boundaries[i:i+2];w,t,dt=weights[lo:hi],mid[lo:hi],delta[lo:hi]
                total=float(w.sum())
                if not len(w) or total<.95:
                    continue
                cdf=np.cumsum(w.astype('float64'))/total
                left,right=np.searchsorted(cdf,[.05,.95]).clip(0,len(w)-1)
                spread=float(t[right]-t[left])
                if spread>maximum_width:
                    continue
                distance=float(t[np.argmax(w/dt.clip(1e-8))])
                records.append((origins[i],directions[i],distance,spread,camera_index,coords[i]))
                accepted+=1
            counts.append(dict(image=path.name,sampled=len(coords),accepted=accepted))
            if camera_index%20==0:
                print(json.dumps(dict(camera=camera_index,accepted_total=len(records))),flush=True)
    if not records:
        raise ValueError('No localized teacher surfaces')
    output.mkdir(parents=True)
    np.savez_compressed(output/'points.npz',origins=np.stack([r[0] for r in records]),
        directions=np.stack([r[1] for r in records]),distance=np.array([r[2] for r in records],dtype='float32'),
        width=np.array([r[3] for r in records],dtype='float32'),camera=np.array([r[4] for r in records]),
        pixel_yx=np.stack([r[5] for r in records]))
    write(output/'manifest.json',dict(train_only=True,measured_depth=False,source='Learned volume termination modes',
        selected=selected,checkpoint_sha256=sha(selected['checkpoint']),transforms_sha256=sha(Path(request['data'])/'transforms.json'),
        file_sha256=sha(output/'points.npz'),points=len(records),grid=grid,maximum_90pct_width=maximum_width,
        near_plane=model.config.near_plane,eval_use_training_near_plane=model.config.eval_use_training_near_plane,
        counts=counts,scope='Geometry initialization candidate only; no source masks or MVS targets; confidence is not ground-truth certification'))
    print(json.dumps(dict(points=len(records),cameras=len(counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=32)
    parser.add_argument('--maximum-width',type=float,default=.02)
    args=parser.parse_args()
    build(args.selection,args.output,args.grid,args.maximum_width)
