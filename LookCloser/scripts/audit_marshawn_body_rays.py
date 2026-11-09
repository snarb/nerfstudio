#!/usr/bin/env python3
"""Matched, camera-balanced body-SfM ray geometry audit of a frozen model.

No training, masks, occupancy edits or whole-voxel carving. Reuse --selection
selection.npz for future exact matched comparisons. SfM depth remains fitted
train evidence; visibility errors, motion, transparency and virtual reflections
can invalidate first-surface assumptions.
"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
from probe_marshawn_occupancy import integrate,file_sha,digest_model,assert_consistent_bounds


def select_balanced(point_idx,camera_idx,subject_ids,count=128,seed=42):
    eligible=np.flatnonzero(np.isin(point_idx,subject_ids));cameras=np.unique(camera_idx[eligible])
    if count<len(cameras) or count>len(eligible):raise ValueError('Ray count must cover all visible cameras and available observations')
    rng=np.random.RandomState(seed);order=rng.permutation(cameras);base,extra=divmod(count,len(cameras));result=[]
    for j,camera in enumerate(order):
        pool=eligible[camera_idx[eligible]==camera];take=base+(j<extra)
        if len(pool)<take:raise ValueError('Insufficient unique observations for balanced camera allocation')
        result.extend(rng.choice(pool,take,replace=False).tolist())
    result=np.array(result,dtype=np.int64);rng.shuffle(result);return result


def interval_mass(starts,ends,sigma,lower,upper):
    """Exact Beer-Lambert mass in a band for piecewise-constant sampled density."""
    starts,ends,sigma=[np.asarray(v,dtype=np.float64).reshape(-1) for v in [starts,ends,sigma]]
    _,trans,_=integrate(starts,ends,sigma)
    lo=np.maximum(starts,lower);hi=np.minimum(ends,upper);length=np.maximum(hi-lo,0)
    at_lo=trans*np.exp(-sigma*np.maximum(lo-starts,0))
    return float(np.sum(at_lo*(-np.expm1(-sigma*length))))


def geometry_hash(frames):
    keys=['physical_camera','w','h','fl_x','fl_y','cx','cy','k1','k2','k3','k4','p1','p2','transform_matrix']
    values=[{k:f[k] for k in keys if k in f} for f in sorted(frames,key=lambda f:f['physical_camera'])]
    import hashlib
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','data','observations','witnesses','subject','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--samples',type=int,default=8192);p.add_argument('--rays',type=int,default=128);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    import torch
    from scipy.spatial import cKDTree
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from render_marshawn_previews import display_rgb
    def configure(c):
        c.eval_checkpoint=a.checkpoint;c.pipeline.datamanager.data=a.data;c.pipeline.datamanager.dataparser.data=a.data;c.pipeline.model.occupancy_eval_dilation_radius=0;return c
    _,pipe,checkpoint,step=eval_setup(a.config,test_mode='val',update_config_callback=configure);pipe.eval();model=pipe.model;model.current_train_step=step;bounds=assert_consistent_bounds(model);before=digest_model(model)
    normalization=pipe.datamanager.train_dataparser_outputs
    if not np.allclose(normalization.dataparser_transform.cpu().numpy(),np.eye(4)[:3],atol=1e-7) or normalization.dataparser_scale!=1:raise ValueError('Prepared-world identity normalization required')
    observation=np.load(a.observations);xyz=np.load(a.witnesses)['xyz'];subject=np.load(a.subject)['xyz'];distance,subject_ids=cKDTree(xyz).query(subject)
    if distance.max()>1e-8:raise ValueError('Subject points do not match train witnesses')
    subject_ids=np.unique(subject_ids);point_idx=observation['point_indices'];camera_idx=observation['camera_indices'];names=observation['camera_names'];uv=observation['uv'];meta=json.loads((a.data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in meta['frames']};ds=pipe.datamanager.train_dataset;frames=[by[Path(f).name] for f in ds.image_filenames];name_to_camera={f['physical_camera']:i for i,f in enumerate(frames)};geom_sha=geometry_hash(frames)
    bindings=dict(observations_sha256=file_sha(a.observations),points_sha256=file_sha(a.witnesses),subject_sha256=file_sha(a.subject),geometry_sha256=geom_sha)
    if a.selection:
        with np.load(a.selection,allow_pickle=False) as saved:
            for key,value in bindings.items():
                if str(saved[key].item())!=value:raise ValueError('Matched ray selection binding mismatch: '+key)
            selected=saved['observation_idx'].copy()
        if len(selected)!=a.rays or len(np.unique(selected))!=len(selected) or not np.isin(point_idx[selected],subject_ids).all():raise ValueError('Invalid matched body observation selection')
    else:selected=select_balanced(point_idx,camera_idx,subject_ids,a.rays,a.seed)
    a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'selection.npz',observation_idx=selected,point_idx=point_idx[selected],camera_idx=camera_idx[selected],uv=uv[selected],witness_xyz=xyz[point_idx[selected]],seed=np.array(a.seed),**{k:np.array(v) for k,v in bindings.items()})
    grid=model.occupancy_grid.binaries.detach().cpu().numpy();box=model.occupancy_grid.aabbs[0].detach().cpu().numpy();shape=np.array(grid.shape[1:]);cameras=ds.cameras.to(model.device);rows=[];traces={};all_subject_points=xyz[subject_ids];ij=np.floor((all_subject_points-box[:3])/(box[3:]-box[:3])*shape).astype(int)
    if (ij<0).any() or (ij>=shape).any():raise ValueError('Body witnesses outside current AABB')
    occupied=grid[0,*ij.T]
    with torch.no_grad():all_sigma=model.field.density_fn(torch.tensor(all_subject_points,device=model.device,dtype=torch.float32)).float().cpu().numpy().reshape(-1)
    sampled_cameras=np.unique(camera_idx[selected]);crop_cameras=set(sampled_cameras[np.linspace(0,len(sampled_cameras)-1,min(12,len(sampled_cameras))).astype(int)].tolist());cropped=set();crop_paths=[]
    for index,obs_id in enumerate(selected):
        source_camera=int(camera_idx[obs_id]);name=str(names[source_camera]);ci=name_to_camera[name];coords=uv[obs_id];point=xyz[point_idx[obs_id]];ray=cameras[ci:ci+1].generate_rays(0,coords=torch.tensor(coords[::-1].copy()[None],device=model.device,dtype=torch.float32),keep_shape=False);ray=model.collider(ray) if model.collider else ray
        origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy();near=float(ray.nears.item());far=float(ray.fars.item());truth=float(np.dot(point-origin,direction));perpendicular=float(np.linalg.norm(point-origin-truth*direction))
        if not near<truth<far or perpendicular>max(1e-4,truth*2/frames[ci]['fl_x']):raise ValueError('Body ray/point calibration mismatch')
        margin=max(.01,.05*truth);free_end=max(near,truth-margin);band=max(.005,.01*truth);band_lo=max(near,truth-band);band_hi=min(far,truth+band);captured=[]
        def capture(module,args,output):
            samples=args[0];captured.append((samples.frustums.starts.detach(),samples.frustums.ends.detach(),output[FieldHeadNames.DENSITY].detach()))
        hook=model.field.register_forward_hook(capture)
        try:
            with torch.no_grad():native=model(ray)
        finally:hook.remove()
        if len(captured)>1:raise ValueError('Multiple field calls need an explicit trace adapter')
        if captured:ns,ne,nd=[v.float().cpu().numpy().reshape(-1) for v in captured[0]]
        else:ns=ne=nd=np.empty(0)
        actual=native['packed_weights'].float().cpu().numpy().reshape(-1);_,_,native_w=integrate(ns,ne,nd)
        discrepancy=float(np.max(np.abs(native_w-actual),initial=0))
        if discrepancy>1e-4:raise ValueError('Recomputed native integration disagrees with actual renderer weights')
        with torch.no_grad():
            edges=torch.linspace(near,far,a.samples+1,device=model.device);mid_t=(edges[:-1]+edges[1:])/2;positions=ray.origins+mid_t[:,None]*ray.directions;uniform_d=model.field.density_fn(positions).float().cpu().numpy().reshape(-1)
        us=edges[:-1].cpu().numpy();ue=edges[1:].cpu().numpy();ua,uT,uw=integrate(us,ue,uniform_d)
        record=dict(observation_idx=int(obs_id),point_idx=int(point_idx[obs_id]),camera_idx=source_camera,camera=name,uv=coords.tolist(),witness_depth=truth,witness_xyz=point.tolist(),near=near,far=far,perpendicular_error=perpendicular,free_margin=margin,free_end=free_end,surface_band_halfwidth=band,weight_max_error=discrepancy,profiles={})
        for label,starts,ends,density,weights in [('native',ns,ne,nd,actual),('uniform',us,ue,uniform_d,uw)]:
            alpha,T,recomputed=integrate(starts,ends,density);opacity=float(weights.sum());mid=(starts+ends)/2;expected=float(np.sum(weights*mid)/opacity) if opacity>1e-10 else None;median=float(mid[min(np.searchsorted(np.cumsum(weights),opacity*.5),len(mid)-1)]) if opacity>1e-10 else None
            prefix_tau=float(np.sum(density*np.maximum(0,np.minimum(ends,free_end)-starts)));prefix_mass=interval_mass(starts,ends,density,near,free_end);surface_mass=interval_mass(starts,ends,density,band_lo,band_hi)
            record['profiles'][label]=dict(samples=len(starts),opacity=opacity,conditional_depth=expected,conditional_depth_relative_error=(expected-truth)/truth if expected is not None else None,median_depth=median,median_depth_relative_error=(median-truth)/truth if median is not None else None,free_optical_thickness=prefix_tau,free_mass=prefix_mass,surface_band_mass=surface_mass,surface_band_conditional_mass=surface_mass/opacity if opacity>1e-10 else None)
            for key,value in [('starts',starts),('ends',ends),('sigma',density),('alpha',alpha),('T',T),('weights',weights)]:traces[f'r{index:03d}_{label}_{key}']=value
        pijk=np.floor((point-box[:3])/(box[3:]-box[:3])*shape).astype(int);record['witness_cell_xyz']=pijk.tolist();record['witness_cell_occupied']=bool(grid[0,*pijk]);record['witness_sigma']=float(all_sigma[np.searchsorted(subject_ids,point_idx[obs_id])]);rows.append(record)
        if source_camera in crop_cameras and source_camera not in cropped:
            image=display_rgb(ds.get_image_float32(ci).cpu().numpy()[...,:3],'linear-reinhard-srgb' if model.config.rgb_output_parameterization!='sigmoid' else 'identity');image=Image.fromarray(np.rint(np.clip(image,0,1)*255).astype(np.uint8));x,y=[int(round(v-.5)) for v in coords];crop=image.crop((x-64,y-64,x+64,y+64)).resize((256,256),Image.Resampling.NEAREST);canvas=Image.new('RGB',(280,320),'#202020');canvas.paste(crop,(0,60));draw=ImageDraw.Draw(canvas);draw.ellipse((120,180,136,196),outline='red',width=2);values=record['profiles']['native'];draw.text((3,3),f'{name} point{point_idx[obs_id]} cell={record["witness_cell_occupied"]}',fill='white');draw.text((3,21),f'depthErr={values["conditional_depth_relative_error"]:.1%}',fill='white');draw.text((3,39),f'freeMass={values["free_mass"]:.3f} bandMass={values["surface_band_mass"]:.3f}',fill='white');filename=f'body_crop_{len(cropped):02d}_{name}.jpg';canvas.save(a.output/filename);crop_paths.append(filename);cropped.add(source_camera)
        if (index+1)%16==0:(a.output/'progress.json').write_text(json.dumps(dict(completed=index+1,total=len(selected))))
    if digest_model(model)!=before:raise ValueError('Frozen model changed during body ray audit')
    np.savez_compressed(a.output/'traces.npz',**traces);np.savez_compressed(a.output/'all_subject_witnesses.npz',point_idx=subject_ids,xyz=all_subject_points,cell_xyz=ij,occupied=occupied,sigma=all_sigma)
    panel=Image.new('RGB',(4*280,3*320),'black')
    for i,path in enumerate(crop_paths):panel.paste(Image.open(a.output/path),(i%4*280,i//4*320))
    panel.save(a.output/'body12_contact.jpg')
    summary={}
    for label in ['native','uniform']:
        values=[row['profiles'][label] for row in rows];summary[label]={}
        for key in ['free_mass','free_optical_thickness','surface_band_mass','surface_band_conditional_mass','conditional_depth_relative_error','median_depth_relative_error']:
            v=np.array([row[key] for row in values if row[key] is not None]);summary[label][key]=dict(count=len(v),median=float(np.median(v)) if len(v) else None,p10=float(np.quantile(v,.1)) if len(v) else None,p90=float(np.quantile(v,.9)) if len(v) else None)
        summary[label]['free_mass_gt_10pct']=sum(row['free_mass']>.1 for row in values);summary[label]['free_mass_gt_50pct']=sum(row['free_mass']>.5 for row in values);summary[label]['surface_band_mass_ge_50pct']=sum(row['surface_band_mass']>=.5 for row in values);summary[label]['depth_abs_rel_lt_5pct']=sum(row['conditional_depth_relative_error'] is not None and abs(row['conditional_depth_relative_error'])<.05 for row in values)
    unique,counts=np.unique(camera_idx[selected],return_counts=True);report=dict(step=step,checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(a.config),implementation_sha256=file_sha(__file__),transforms_sha256=file_sha(a.data/'transforms.json'),bindings=bindings,selection_sha256=file_sha(a.output/'selection.npz'),seed=a.seed,source_subject_rows=len(subject),unique_body_points=len(subject_ids),actual_body_observations=int(np.isin(point_idx,subject_ids).sum()),selected_rays=len(selected),selected_unique_points=len(np.unique(point_idx[selected])),camera_counts={str(names[c]):int(n) for c,n in zip(unique,counts)},policy=dict(free_margin='max(.01,.05*witnessDepth)',surface_band='+-max(.005,.01*witnessDepth)',voxel_size_independent=True,uniform_samples=a.samples),summary=summary,all_subject_cell_misses=int((~occupied).sum()),selected_cell_misses=sum(not row['witness_cell_occupied'] for row in rows),all_subject_sigma_median=float(np.median(all_sigma)),bounds=bounds,frozen_state_unchanged=True,state_sha256=before,rows=rows,crops=crop_paths,uses_eval_rgb=False,limitations=['Body SfM points are fitted train evidence, not independent geometry ground truth.','Visibility/motion/correspondence errors, transparency and virtual reflections can invalidate first-surface assumptions.','Native vs uniform compares the same field; untrained density in skipped cells does not prove occupancy should admit it.','Band/prefix measures integrate visible lines, not complete voxels; no carving or grid certification.'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(summary=summary,selected_rays=len(selected),cameras=len(unique),cell_misses=report['all_subject_cell_misses'])),flush=True)


if __name__=='__main__':main()
