#!/usr/bin/env python3
"""Measure real train RGB and learned colour at the fitted cord depth."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from nerfstudio.field_components.field_heads import FieldHeadNames

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, sha, write


def audit(request_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    request=deepcopy(json.loads(request_path.read_text()))
    if not request.get('resume') or not request['model'].get('cord_sampling_fit_path'):
        raise ValueError('Need a checkpoint and train-only cord line')
    request['output']=str(output)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    config=configuration(request)
    trainer=config.setup(local_rank=0,world_size=1)
    trainer.setup()
    trainer.pipeline.eval()
    model=trainer.pipeline.model
    model.current_train_step=int(Path(request['resume']).stem.split('-')[-1])
    before=field_fingerprint(model)
    fit=json.loads(Path(request['model']['cord_sampling_fit_path']).read_text())
    camera_ids=sorted(set.intersection(*(set(fit['fit_camera_ids'][key])
                                         for key in ('junction','first_bead'))))
    transforms=json.loads((Path(request['data'])/'transforms.json').read_text())
    frames={int(frame['camera_id']):frame for frame in transforms['frames']}
    dataset=trainer.pipeline.datamanager.train_dataset
    indices={int(path.stem.split('_')[1]):i for i,path in enumerate(dataset.image_filenames)}
    start,end=model.cord_sampling_endpoints.to(model.device)
    fractions=torch.linspace(.08,.92,12,device=model.device)
    points=start[None]+fractions[:,None]*(end-start)[None]
    offsets=(-.012,-.008,-.006,-.004,-.003,-.002,-.001,0.,.001,.002,.003,.004,.006,.008,.012)
    sweep_halfwidth=.0015
    rows=[]
    with torch.no_grad():
        for camera_id in camera_ids:
            frame=frames[camera_id]
            source_hash=sha(Path(request['data'])/frame['file_path'])
            index=indices[camera_id]
            camera=dataset.cameras[index:index+1].to(model.device)
            pose=torch.tensor(frame['transform_matrix'],device=model.device,dtype=points.dtype)
            local=(points-pose[:3,3])@pose[:3,:3]
            forward=-local[:,2]
            x=frame['fl_x']*local[:,0]/forward+frame['cx']
            y=-frame['fl_y']*local[:,1]/forward+frame['cy']
            if not bool(((forward>0)&(x>=0)&(x<frame['w'])&(y>=0)&(y<frame['h'])).all()):
                raise ValueError(f'Cord point leaves training camera {camera_id}')
            px=x.round().long().clamp(0,frame['w']-1)
            py=y.round().long().clamp(0,frame['h']-1)
            coords=torch.stack([py,px],-1).float()+.5
            rays=camera.generate_rays(0,coords=coords)
            captured={}

            def capture(_module,args,values):
                samples=args[0]
                captured['t']=((samples.frustums.starts+samples.frustums.ends)/2).flatten()
                captured['color']=values[FieldHeadNames.RGB].float()

            hook=model.field.register_forward_hook(capture)
            try:
                prediction=model(rays)
            finally:
                hook.remove()
            ids=prediction['packed_ray_indices'].flatten().long()
            weights=prediction['packed_weights'].flatten().float()
            t,color=captured['t'],captured['color']
            if len(ids)!=len(t):
                raise RuntimeError('Captured field samples disagree with packed rays')
            reproduced=torch.zeros(len(points),3,device=model.device).index_add_(0,ids,weights[:,None]*color)
            parity=float((reproduced-prediction['rgb']).abs().max())
            if parity>2e-4:
                raise RuntimeError(f'RGB reconstruction error {parity}')
            image=dataset[index]['image'].to(model.device)
            gt=image[py,px,:3].float()
            origin=rays.origins
            direction=rays.directions
            depth=((points-origin)*direction).sum(-1)/(direction*direction).sum(-1)
            for i in range(len(points)):
                take=ids==i
                wi,ti,ci=weights[take],t[take],color[take]
                band=(ti-depth[i]).abs()<=.003
                mass=wi[band].sum()
                rgb=(wi[band,None]*ci[band]).sum(0)/mass.clamp_min(1e-9)
                offset_bands={}
                for offset in offsets:
                    selected=(ti-(depth[i]+offset)).abs()<=sweep_halfwidth
                    selected_mass=wi[selected].sum()
                    selected_rgb=(wi[selected,None]*ci[selected]).sum(0)/selected_mass.clamp_min(1e-9)
                    offset_bands[f'{offset:+.3f}']=dict(mass=float(selected_mass),rgb=selected_rgb.cpu().tolist())
                rows.append(dict(camera_id=camera_id,fraction=float(fractions[i]),
                                 x=int(px[i]),y=int(py[i]),source_gt=gt[i].cpu().tolist(),
                                 rendered_rgb=prediction['rgb'][i].cpu().tolist(),
                                 band_rgb=rgb.cpu().tolist(),band_mass=float(mass),
                                 front_mass=float(wi[ti<depth[i]-.003].sum()),
                                 back_mass=float(wi[ti>depth[i]+.003].sum()),
                                 projected_depth=float(depth[i]),
                                 source_rgb_sha256=source_hash,offset_bands=offset_bands))
    if field_fingerprint(model)!=before:
        raise RuntimeError('Forward audit changed model field')
    def score(key,selected):
        truth=np.array([row['source_gt'] for row in selected])
        rgb=np.array([row[key] for row in selected])
        return float(-10*np.log10(np.mean((rgb-truth)**2)))
    summary={}
    for camera_id in camera_ids:
        selected=[row for row in rows if row['camera_id']==camera_id]
        summary[str(camera_id)]=dict(points=len(selected),
                                     rendered_rgb_psnr=score('rendered_rgb',selected),
                                     band_rgb_psnr=score('band_rgb',selected),
                                     mean_band_mass=float(np.mean([row['band_mass'] for row in selected])))
    sweep={}
    for offset in offsets:
        key=f'{offset:+.3f}'
        chosen=[row for row in rows if row['offset_bands']['+0.000']['mass']>=.15
                and row['offset_bands'][key]['mass']>=.15]
        if chosen:
            gt=np.array([row['source_gt'] for row in chosen])
            candidate=np.array([row['offset_bands'][key]['rgb'] for row in chosen])
            center=np.array([row['offset_bands']['+0.000']['rgb'] for row in chosen])
            psnr=lambda rgb: float(-10*np.log10(np.mean((rgb-gt)**2)))
            sweep[key]=dict(points=len(chosen),candidate_psnr=psnr(candidate),
                            center_same_points_psnr=psnr(center),
                            mean_candidate_mass=float(np.mean([row['offset_bands'][key]['mass'] for row in chosen])),
                            camera_counts={str(camera_id):sum(row['camera_id']==camera_id for row in chosen)
                                           for camera_id in camera_ids})
        else:
            sweep[key]=dict(points=0)
    write(output/'result.json',dict(scope='Read-only train-camera line colour diagnostic, not a mask',
          masks_used=False,heldout_camera_excluded=fit['validation_camera_id'],
          source_request=str(request_path),checkpoint=request['resume'],
          checkpoint_sha256=sha(Path(request['resume'])),
          fit_sha256=sha(Path(request['model']['cord_sampling_fit_path'])),
          transforms_sha256=sha(Path(request['data'])/'transforms.json'),
          field_sha256=before,band_halfwidth=.003,summary=summary,
          surface_sweep_halfwidth=sweep_halfwidth,surface_sweep=sweep,rows=rows))
    print(json.dumps(dict(summary=summary,surface_sweep=sweep)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.request,args.output)
