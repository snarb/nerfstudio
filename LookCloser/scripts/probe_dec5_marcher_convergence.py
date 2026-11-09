#!/usr/bin/env python3
"""Frozen-field numerical convergence/culling audit on fixed hair witnesses."""
from pathlib import Path
import argparse,json
import numpy as np,torch,yaml
import torch.nn.functional as F
from dec5_hair import ray_to_z,project,packed_depth_quantile
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def probe(data,training,checkpoint,correspondences,matching,output):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(20*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    if output.exists():raise FileExistsError(output)
    request=json.loads((training/'request.json').read_text())
    if request['data_sha256']!=sha(data/'transforms.json'):raise ValueError('Field/dataset mismatch')
    cfg=yaml.load(next(training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
    payload=json.loads((data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']};ds=pipe.datamanager.train_dataset
    frames=[by[Path(p).name] for p in ds.image_filenames];lookup={f['physical_camera']:i for i,f in enumerate(frames)}
    raw=np.asarray([f['transform_matrix'] for f in frames]);normalized=ds.cameras.camera_to_worlds.cpu().numpy();rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    witness=np.load(correspondences);receipt=json.loads(matching.read_text());points=torch.as_tensor(scale*witness['points']@rotation.T+translation,device='cuda',dtype=torch.float32)
    grid=model.occupancy_grid;original_binary=grid.binaries.clone();lo,hi=grid.aabbs[0,:3],grid.aabbs[0,3:];resolution=points.new_tensor(grid.binaries.shape[-3:])
    ij=torch.floor((points-lo)/(hi-lo)*resolution).long();inside=((ij>=0)&(ij<resolution)).all(-1);ij=ij.clamp_min(0)
    ij=torch.minimum(ij,resolution.long()-1);occupied=grid.binaries[0,ij[:,0],ij[:,1],ij[:,2]]&inside
    with torch.no_grad():sigma=model.field.density_fn(points).reshape(-1).float().cpu().numpy()
    culling=dict(witness_occupied_fraction=float(occupied.float().mean()),global_occupied_fraction=float(grid.binaries.float().mean()),
                 density_median_occupied=float(np.median(sigma[occupied.cpu().numpy()])) if occupied.any() else None,
                 density_median_culled=float(np.median(sigma[~occupied.cpu().numpy()])) if (~occupied).any() else None)
    original=(model.config.ray_sampling_mode,model.config.enable_adaptive_ray_marching,model.config.max_steps_per_ray,model.config.fixed_num_samples_per_ray)
    rows=[];pooled={}
    try:
        for label,mode,count,dilate in [('adaptive2048','adaptive',2048,False),('adaptive8192','adaptive',8192,False),('adaptive8192_dilate1','adaptive',8192,True),('uniform4096','fixed',4096,False),('uniform16384','fixed',16384,False)]:
            grid.binaries.copy_(F.max_pool3d(original_binary[None].float(),3,1,1)[0]>.5 if dilate else original_binary)
            model.config.ray_sampling_mode=mode;model.config.enable_adaptive_ray_marching=mode=='adaptive';model.config.max_steps_per_ray=count if mode=='adaptive' else original[2];model.config.fixed_num_samples_per_ray=count
            for key,row in zip(['uv_a','uv_b','uv_c'],receipt['rows']):
                frame=frames[lookup[row['frame']['physical_camera']]];uv=witness[key];camera=ds.cameras[lookup[frame['physical_camera']]:lookup[frame['physical_camera']]+1].to('cuda')
                rays=camera.generate_rays(0,coords=torch.as_tensor(uv[:,::-1].copy(),device='cuda'),keep_shape=False)
                with torch.no_grad():result=model(rays)
                if mode=='adaptive':
                    starts,ends,weights=[result[k].float().cpu().numpy().reshape(-1) for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']];indices=result['packed_ray_indices'].cpu().numpy()
                else:
                    weights=result['loss_weights'][...,0].float().cpu().numpy().reshape(-1);indices=np.repeat(np.arange(len(uv)),count)
                    starts=np.tile(np.arange(count)/count,len(uv));ends=starts+1/count
                distance=packed_depth_quantile(starts,ends,weights,indices,rays.nears.cpu().numpy(),rays.fars.cpu().numpy())
                median=ray_to_z(distance/scale,uv,frame);expected=ray_to_z(result['depth'].reshape(-1).float().cpu().numpy()/scale,uv,frame);_,reference=project(witness['points'],frame)
                counts=result.get('num_samples_per_ray')
                record=dict(mode=label,camera=frame['physical_camera'],saturation=float(result.get('adaptive_saturation_rate',torch.tensor(0))),
                            samples_mean=float(counts.float().mean()) if counts is not None else count,
                            samples_max=int(counts.max()) if counts is not None else count,metrics={})
                for statistic,value in [('median',median),('expected',expected)]:
                    valid=(value>0)&np.isfinite(value);error=np.where(valid,np.abs(value-reference)/reference,1.)
                    record['metrics'][statistic]=dict(median_absolute_relative=float(np.median(error)),p90_absolute_relative=float(np.quantile(error,.9)),valid_fraction=float(valid.mean()))
                    pooled.setdefault(label+'_'+statistic,[]).append(error)
                rows.append(record)
    finally:
        grid.binaries.copy_(original_binary)
        model.config.ray_sampling_mode,model.config.enable_adaptive_ray_marching,model.config.max_steps_per_ray,model.config.fixed_num_samples_per_ray=original
    if fingerprint(model)!=before:raise ValueError('Convergence probe changed the frozen field')
    aggregate={k:dict(median_absolute_relative=float(np.median(np.concatenate(v))),p90_absolute_relative=float(np.quantile(np.concatenate(v),.9))) for k,v in pooled.items()}
    atomic_json(output,dict(aggregate=aggregate,records=rows,culling=culling,checkpoint_sha256=sha(checkpoint),state_unchanged=True,
        caveat='fixed noisy triangulated train-RGB hair witnesses, not independent 3D GT',
        uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','training','checkpoint','correspondences','matching','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();probe(a.data,a.training,a.checkpoint,a.correspondences,a.matching,a.output)


if __name__=='__main__':main()
