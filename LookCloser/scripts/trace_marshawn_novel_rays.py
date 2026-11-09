#!/usr/bin/env python3
"""Actual native/all-cells rays at matched-K novel pixels, with train-frustum support.

Frustum inclusion is not visibility or geometry truth. No carving, training,
objective, learned-density edits or persistent occupancy writes are performed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from verify_marshawn_release import runtime,write_json
from marshawn_release_common import load_camera_bundle,trace_profiles,point_cells
from probe_marshawn_occupancy import digest_model,file_sha,integrate


def project(frame,points):
    pose=np.asarray(frame['transform_matrix'],float);camera=(np.asarray(points)-pose[:3,3])@pose[:3,:3]
    depth=-camera[:,2];safe=np.where(np.abs(depth)>1e-12,depth,1e-12)
    uv=np.stack([frame['fl_x']*camera[:,0]/safe+frame['cx'],frame['cy']-frame['fl_y']*camera[:,1]/safe],axis=1)
    inside=(depth>0)&(uv[:,0]>=0)&(uv[:,0]<frame['w'])&(uv[:,1]>=0)&(uv[:,1]<frame['h'])
    return uv,depth,inside


def frustum_counts(frames,points):
    count=np.zeros(len(points),np.int16)
    for frame in frames:count+=project(frame,points)[2]
    return count


def component_support(weights,mask,count):
    value=np.asarray(weights)*np.asarray(mask)
    mass=float(value.sum());unsupported=float(value[np.asarray(count)==0].sum())
    return dict(termination_mass=mass,mass_outside_all_train_frusta=unsupported,
        fraction_outside_all_train_frusta=unsupported/mass if mass>1e-12 else None,
        mass_inside_at_least_one_train_frustum=mass-unsupported)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','grid','grid-receipt','camera-bundle','camera-bundle-receipt','queries','frames-directory','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--code-root',type=Path,action='append',default=[])
    p.add_argument('--chunk-size',type=int,default=512);p.add_argument('--memory-gib',type=float,default=24)
    p.add_argument('--width',type=int,default=640);p.add_argument('--prefix-distance',type=float,default=.5)
    a=p.parse_args();pipe,core,active,boxes,receipt=runtime(a);model=pipe.model;before=digest_model(model)
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    from nerfstudio.field_components.field_heads import FieldHeadNames
    data=a.data or Path(pipe.config.datamanager.dataparser.data);meta=json.loads((data/'transforms.json').read_text());train=set(meta['train_filenames']);frames=[f for f in meta['frames'] if f['file_path'] in train]
    arrays,contract=load_camera_bundle(a.camera_bundle,a.camera_bundle_receipt,frames)
    cameras=Cameras(camera_to_worlds=torch.tensor(arrays['camera_to_worlds'],dtype=torch.float32),
        fx=torch.tensor(arrays['fx'],dtype=torch.float32),fy=torch.tensor(arrays['fy'],dtype=torch.float32),
        cx=torch.tensor(arrays['cx'],dtype=torch.float32),cy=torch.tensor(arrays['cy'],dtype=torch.float32),
        width=torch.tensor(arrays['width']),height=torch.tensor(arrays['height']),camera_type=CameraType.PERSPECTIVE)
    cameras.rescale_output_resolution(a.width/int(arrays['width'][0]));cameras=cameras.to(model.device)
    a.output.mkdir(parents=True,exist_ok=False);query=json.loads(a.queries.read_text());rows=[];traces={}
    if model.config.background_color!='black':raise ValueError('Explicit black background required for RGB identity check')
    for index,q in enumerate(query):
        view,x,y=int(q['view']),int(q['x']),int(q['y']);camera=cameras[view:view+1]
        ray=camera.generate_rays(0,coords=torch.tensor([[y+.5,x+.5]],device=model.device),keep_shape=False)
        origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy();colors=[]
        def rgb_hook(_module,_args,output):colors.append(output[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy())
        handle=model.field.register_forward_hook(rgb_hook)
        try:profiles=trace_profiles(model,ray)
        finally:handle.remove()
        if len(colors)!=2:raise ValueError('Expected native/all-cells RGB calls')
        saved=np.load(a.frames_directory/f'view_{view:02d}_buffers.npz')['rgb'][y,x]
        row=dict(query=q,origin=origin.tolist(),direction=direction.tolist(),
            coordinate_units='Prepared normalized world, not claimed meters',profiles={})
        for (label,trace),rgb in zip(profiles.items(),colors):
            starts,ends,sigma,weights=[trace[k] for k in ['starts','ends','sigma','weights']]
            mid=(starts+ends)/2;points=origin+mid[:,None]*direction;count=frustum_counts(frames,points)
            actual_rgb=(weights[:,None]*rgb).sum(0);opacity=float(weights.sum());expected=float(np.sum(weights*mid)/opacity)
            alpha,T,_=integrate(starts,ends,sigma);prefix=[]
            for distance in [.05,.1,.2,.3,.5,1.]:
                length=np.maximum(0,np.minimum(ends,distance)-starts);mass=T*(-np.expm1(-sigma*length));tau=float(np.sum(sigma*length))
                prefix.append(dict(distance=distance,optical_thickness=tau,transmittance=float(np.exp(-tau)),termination_mass=float(mass.sum()),rgb_contribution=(mass[:,None]*rgb).sum(0).tolist()))
            positions=[]
            for quantile in [.1,.5,.9]:
                sample=min(int(np.searchsorted(np.cumsum(weights),opacity*quantile)),len(mid)-1)
                positions.append(dict(quantile=quantile,distance=float(mid[sample]),xyz=points[sample].tolist(),sigma=float(sigma[sample]),train_frustum_count=int(count[sample])))
            peak=int(np.argmax(weights));near=mid<=a.prefix_distance;bright=rgb.mean(1)>.8
            closest=[]
            for frame in frames:
                uv,depth,inside=project(frame,points[[peak]])
                closest.append(dict(camera=frame['physical_camera'],camera_center= np.asarray(frame['transform_matrix'])[:3,3].tolist(),
                    distance_from_novel_origin=float(np.linalg.norm(np.asarray(frame['transform_matrix'])[:3,3]-origin)),
                    peak_uv=uv[0].tolist(),peak_depth=float(depth[0]),peak_inside_frustum=bool(inside[0])))
            closest.sort(key=lambda r:r['distance_from_novel_origin'])
            ij,inbox,occupied=point_cells(points,active,boxes);_,_,in_core=point_cells(points,core,boxes)
            record=dict(actual_rgb=actual_rgb.tolist(),opacity=opacity,expected_depth=expected,
                weight_max_error=trace['weight_max_error'],prefix=prefix,termination_quantiles=positions,
                peak=dict(sample=peak,distance=float(mid[peak]),xyz=points[peak].tolist(),sigma=float(sigma[peak]),weight=float(weights[peak]),
                    cell_xyz=ij[peak].tolist(),active=bool(occupied[peak]),core=bool(in_core[peak]),train_frustum_count=int(count[peak])),
                train_frustum_support=dict(full=component_support(weights,np.ones(len(weights),bool),count),
                    near_prefix=component_support(weights,near,count),near_bright_prefix=component_support(weights,near&bright,count),
                    prefix_distance=a.prefix_distance,definition='Frustum inclusion only; occlusion and actual visibility unknown'),
                nearest_train_cameras=closest[:8])
            row['profiles'][label]=record
            for key,value in dict(trace,field_rgb=rgb,positions=points,train_frustum_count=count,cell_xyz=ij,cell_active=occupied,cell_core=in_core).items():
                if isinstance(value,np.ndarray):traces[f'r{index:02d}_{label}_{key}']=value
        error=float(np.max(np.abs(np.array(row['profiles']['native']['actual_rgb'])-saved)))
        if error>2e-5:raise ValueError('Ray does not reproduce exact saved matched-K image pixel: '+str(error))
        row.update(saved_rgb=saved.tolist(),saved_rgb_max_error=error);rows.append(row)
        write_json(a.output/'progress.json',dict(rays_completed=index+1,total=len(query)))
    if digest_model(model)!=before:raise ValueError('Frozen native state changed')
    np.savez_compressed(a.output/'traces.npz',**traces)
    receipt.update(trace_script_sha256=file_sha(__file__),camera_bundle_contract=contract,query_sha256=file_sha(a.queries),
        frame_receipt_sha256=file_sha(a.frames_directory/'render_receipt.json'),rows=rows,
        frozen_state_unchanged=True,state_sha256=before,train_cameras=len(frames),
        limitations=['Outside all captured train frusta supports lack of RGB constraints, not physical emptiness.',
            'Inside a frustum does not establish visibility; real GT patches and occlusion must be inspected.',
            'No grid carving or new density regularizer was applied.'])
    write_json(a.output/'audit.json',receipt);print(json.dumps(dict(output=str(a.output),rays=len(rows),state_unchanged=True)))


if __name__=='__main__':main()
