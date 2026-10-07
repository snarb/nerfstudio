#!/usr/bin/env python3
"""Compare learned termination profiles and actual surface hits on fixed skin rays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from export_luster_opacity import load
from run_luster_opacity import write


def inspect(selections,output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    figure,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for selection in selections:
        selected=json.loads(selection.read_text())
        if 'selected' in selected:selected=selected['selected']
        trainer,request=load(selected)
        model=trainer.pipeline.model
        model.config.adaptive_coarse_step_size=.0005
        model.config.max_steps_per_ray=4096
        model.config.surface_num_samples=512
        model.config.surface_refine_steps=12
        trainer.pipeline.eval()
        dataset=trainer.pipeline.datamanager.train_dataset
        index=next(i for i,p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1])==97)
        camera=dataset.cameras[index:index+1].to(model.device)
        name=dataset.image_filenames[index].name
        x0,y0,x1,y1=request['roi_boxes'][name]['face']
        xy=np.array([[x0+(x1-x0)*a,y0+(y1-y0)*b] for a,b in [(.3,.55),(.7,.55),(.4,.4),(.5,.7)]],dtype=np.float32)
        rays=camera.generate_rays(0,coords=torch.tensor(xy[:,::-1].copy(),device=model.device)+.5)
        with torch.no_grad():
            rendered=model(rays)
            edges=torch.linspace(0,1,4097,device=model.device)
            starts=rays.nears+(rays.fars-rays.nears)*edges[None,:-1]
            ends=rays.nears+(rays.fars-rays.nears)*edges[None,1:]
            mids=(starts+ends)*.5
            positions=rays.origins[:,None]+rays.directions[:,None]*mids[...,None]
            if model.config.representation=='volume':
                sigma=model.field.density_fn(positions.reshape(-1,3)).reshape(4,4096).float()
                alpha=1-torch.exp(-sigma*(ends-starts))
                trans=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),(1-alpha).clamp_min(1e-10)],-1),-1)[:,:-1]
                weights=(alpha*trans).cpu().numpy()
                values=sigma.cpu().numpy()
            else:
                values=model.field.query_surface(positions.reshape(-1,3)).reshape(4,4096).cpu().numpy()
                weights=None
        label=selection.parent.name+f" step{selected['step']}"
        distances=mids.cpu().numpy()
        for ray in range(4):
            axis=axes.reshape(-1)[ray]
            record=dict(selection=str(selection),step=selected['step'],representation=model.config.representation,
                        train_camera=97,pixel_xy=xy[ray].tolist(),rendered_depth=float(rendered['depth'][ray]),
                        rendered_accumulation=float(rendered['accumulation'][ray]))
            if weights is not None:
                total=weights[ray].sum()
                cdf=np.cumsum(weights[ray])/max(total,1e-8)
                low,high=np.searchsorted(cdf,[.05,.95]).clip(0,4095)
                record.update(dense_profile_accumulation=float(total),
                              termination_90pct_width=float(distances[ray,high]-distances[ray,low]) if total>1e-5 else None)
                axis.plot(distances[ray],weights[ray]/(ends-starts)[ray].cpu().numpy(),label=label)
            else:
                changes=np.flatnonzero(values[ray,:-1]*values[ray,1:]<=0)
                record['dense_first_crossing']=float(distances[ray,changes[0]]) if len(changes) else None
                # Later sign changes are legitimate occluded surfaces, not
                # a volume-opacity distribution or a fog-width measurement.
                axis.axvline(record['rendered_depth'],label=label,linestyle='--')
            axis.set_title(f'cam097 skin ray{ray} ({xy[ray,0]:.0f},{xy[ray,1]:.0f})')
            axis.set_xlabel('Ray distance (normalized scene units)')
            axis.set_ylabel('Termination density / surface hit')
            axis.legend(fontsize=7)
            rows.append(record)
        np.savez_compressed(output/(selection.parent.name+f"_{selected['step']}_profiles.npz"),
                            distance=distances,field_values=values,
                            weights=weights if weights is not None else np.empty((0,)),pixels=xy)
        del trainer,model
        torch.cuda.empty_cache()
    figure.savefig(output/'profiles.png',dpi=140)
    plt.close(figure)
    write(output/'profiles.json',dict(train_only_diagnostic=True,rays=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selections',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    inspect(args.selections,args.output)
