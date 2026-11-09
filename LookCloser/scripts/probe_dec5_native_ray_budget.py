#!/usr/bin/env python3
"""Frozen native6K hair rays: test whether the marcher cap truncates evidence."""
from pathlib import Path
import argparse
import numpy as np,torch
from dec5_sequence_checkpoint import load
from dec5_background_separation import network_fingerprint
from dec5_hair import packed_depth_quantile_vectorized
from run_dec5_sequence import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(20*2**30/torch.cuda.get_device_properties(0).total_memory)
    training=r/'native_boundary_control/frames/000901/training_native_boundary_control'
    pipe,provider,provenance=load(training,r/'native_frame901/data',r/'native96_bundle/control/inference.pt')
    model=pipe.model;before=network_fingerprint(model);old_cap=model.config.max_steps_per_ray
    ds=pipe.datamanager.train_dataset;lookup={Path(path).stem[len('frame_train_'):]:i for i,path in enumerate(ds.image_filenames)}
    records=[];rng=np.random.RandomState(42)
    try:
        for f in read(r/'source8_hd.json')['frames']:
            name=f['physical_camera'];prob=np.load(r/'frame/semantics'/(name+'.npz'))['probability'][...,0]/65535
            with np.load(r/'native96_bundle/control/source_low'/(name+'.npz')) as source:opacity=source['opacity']
            selections=[];groups=[]
            for label,domain in [('low',(opacity<.1)),('partial',(opacity>=.1)&(opacity<.9)),('opaque',opacity>=.9)]:
                yy,xx=np.nonzero((prob>.5)&domain)
                if not len(xx):continue
                selected=rng.choice(len(xx),min(32,len(xx)),replace=False)
                selections.append(np.stack([yy[selected]+.5,xx[selected]+.5],axis=1));groups.extend([label]*len(selected))
            if not selections:continue
            coords=np.concatenate(selections)*3.2;camera=ds.cameras[lookup[name]:lookup[name]+1].to('cuda')
            if (int(camera.width[0]),int(camera.height[0]))!=(6144,3072):raise ValueError('Native camera required')
            rows={};values={}
            for cap in [4096,16384]:
                model.config.max_steps_per_ray=cap
                rays=camera.generate_rays(0,coords=torch.as_tensor(coords,device='cuda',dtype=torch.float32),keep_shape=False)
                with torch.no_grad():result=model(rays)
                count=result['num_samples_per_ray'].float();alpha=result['accumulation'].reshape(-1).float().cpu().numpy()
                starts,ends,weights=[result[k].float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']]
                depth=packed_depth_quantile_vectorized(starts,ends,weights,result['packed_ray_indices'].cpu().numpy(),
                    rays.nears.cpu().numpy(),rays.fars.cpu().numpy(),quantile=.1,minimum_mass=.001)
                values[cap]=dict(alpha=alpha,depth=depth,rgb=result['rgb'].float().cpu().numpy())
                rows[str(cap)]=dict(rays=len(alpha),samples_mean=float(count.mean()),samples_max=int(count.max()),
                    saturation_rate=float(result['adaptive_saturation_rate']),opacity_below_01=int((alpha<.1).sum()))
            base,large=values[4096],values[16384];valid=(base['depth']>0)&(large['depth']>0)
            records.append(dict(camera=name,groups={label:groups.count(label) for label in set(groups)},caps=rows,
                opacity_threshold_crossings=int(((base['alpha']<.1)!=(large['alpha']<.1)).sum()),
                opacity_max_difference=float(np.abs(base['alpha']-large['alpha']).max()),
                rgb_max_difference=float(np.abs(base['rgb']-large['rgb']).max()),
                q10_max_relative_difference=float((np.abs(base['depth'][valid]-large['depth'][valid])/base['depth'][valid]).max()) if valid.any() else None))
    finally:model.config.max_steps_per_ray=old_cap
    provider.verify()
    if network_fingerprint(model)!=before:raise ValueError('Probe changed frozen tensors')
    write(r/'native_ray_budget.json',dict(records=records,provenance=provenance,native6k_training=True,
        query_camera_raster=[1920,960],coordinates_scaled_to_native_camera=True,training_rays_only=True,
        no_independent_depth_ground_truth=True,model_tensors_unchanged=True))
    print(dict(rays=sum(x['caps']['4096']['rays'] for x in records),threshold_crossings=sum(x['opacity_threshold_crossings'] for x in records)))


if __name__=='__main__':main()
