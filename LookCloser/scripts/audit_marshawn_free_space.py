#!/usr/bin/env python3
"""Frozen-field SfM witness free-ray audit. No optimizer steps or grid edits.

Observed train tracks provide candidate free ray segments, not full voxel truth.
Thin/transparent foreground, correspondence errors and occlusions remain caveats.
"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw
from probe_marshawn_occupancy import integrate,assert_consistent_bounds,digest_model,file_sha


def prefix_tau(starts, ends, sigma, stop):
    return float(np.sum(sigma*np.maximum(0., np.minimum(ends, stop)-starts)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','observations','witnesses','data','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--samples',type=int,default=8192);p.add_argument('--rays',type=int,default=128);p.add_argument('--display-transform',choices=['identity','linear-reinhard-srgb'],default='identity')
    a=p.parse_args()
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    def configure(c):
        c.eval_checkpoint=a.checkpoint;c.pipeline.model.occupancy_eval_dilation_radius=0
        c.pipeline.datamanager.data=a.data;c.pipeline.datamanager.dataparser.data=a.data
        return c
    _,pipe,checkpoint,step=eval_setup(a.config,test_mode='val',update_config_callback=configure)
    model=pipe.model;model.current_train_step=step;bounds=assert_consistent_bounds(model);before=digest_model(model)
    outputs=pipe.datamanager.train_dataparser_outputs
    if not np.allclose(outputs.dataparser_transform.numpy(),np.eye(4)[:3],atol=1e-7) or outputs.dataparser_scale!=1:
        raise ValueError('This witness audit requires prepared_world identity parser normalization')
    obs=np.load(a.observations);xyz=np.load(a.witnesses)['xyz'];meta=json.loads((a.data/'transforms.json').read_text());byfile={Path(f['file_path']).name:f for f in meta['frames']}
    ds=pipe.datamanager.train_dataset;frames=[byfile[Path(f).name] for f in ds.image_filenames];camera_indices={f['physical_camera']:i for i,f in enumerate(frames)}
    names=obs['camera_names'];rng=np.random.RandomState(42);selected=[]
    # Two observations per solve camera; UV separation avoids repeatedly probing one patch.
    for camera,name in enumerate(names):
        ids=np.flatnonzero(obs['camera_indices']==camera)
        first=int(rng.choice(ids));second=int(ids[np.argmax(np.linalg.norm(obs['uv'][ids]-obs['uv'][first],axis=1))]);selected.extend([first,second])
    if len(selected)<a.rays:selected.extend(rng.choice(np.setdiff1d(np.arange(len(obs['uv'])),selected),a.rays-len(selected),replace=False).tolist())
    selected=np.array(selected[:a.rays]);cameras=ds.cameras.to(model.device)
    grid=model.occupancy_grid.binaries.detach().cpu().numpy();box=model.occupancy_grid.aabbs[0].detach().cpu().numpy();shape=np.array(grid.shape[1:]);cell=(box[3:]-box[:3])/shape;voxel_diag=float(np.linalg.norm(cell))
    a.output.mkdir(parents=True,exist_ok=False);rows=[];traces={};free_counts=np.zeros(tuple(shape),np.uint32)
    def indices(points):
        rel=(points-box[:3])/(box[3:]-box[:3]);valid=((rel>=0)&(rel<1)).all(1);ijk=np.floor(rel[valid]*shape).astype(int);return valid,ijk
    surface_valid,surface_ijk=indices(xyz);surface_occupied=grid[0,*surface_ijk.T];surface_sigma=[]
    with torch.no_grad():
        for start in range(0,len(xyz),8192):surface_sigma.extend(model.field.density_fn(torch.tensor(xyz[start:start+8192],device=model.device,dtype=torch.float32)).reshape(-1).float().cpu().tolist())
    for j,observation in enumerate(selected):
        name=str(names[obs['camera_indices'][observation]]);ci=camera_indices[name];uv=obs['uv'][observation];point=xyz[obs['point_indices'][observation]]
        rays=cameras[ci:ci+1].generate_rays(0,coords=torch.tensor(uv[::-1].copy()[None],device=model.device,dtype=torch.float32),keep_shape=False)
        rays=model.collider(rays);origin=rays.origins[0].cpu().numpy();direction=rays.directions[0].cpu().numpy();near=float(rays.nears.item());far=float(rays.fars.item());distance=float(np.dot(point-origin,direction));perpendicular=float(np.linalg.norm(point-origin-distance*direction))
        if distance<=near or distance>=far or perpendicular>max(1e-4,distance*2/frames[ci]['fl_x']):raise ValueError('Witness/ray calibration mismatch')
        margin=max(3*voxel_diag,.05*distance);free_end=max(near,distance-margin)
        with torch.no_grad():
            result=model(rays)
            edges=torch.linspace(near,far,a.samples+1,device=model.device);mid=(edges[:-1]+edges[1:])/2;positions=rays.origins+mid[:,None]*rays.directions
            sigma_t,rgb=model.field.query_points(positions,rays.directions.expand(a.samples,-1),camera_indices=rays.camera_indices.expand(a.samples,-1))
        starts=edges[:-1].cpu().numpy();ends=edges[1:].cpu().numpy();sigma=sigma_t.reshape(-1).float().cpu().numpy();alpha,T,w=integrate(starts,ends,sigma);mid=(starts+ends)/2
        tau=prefix_tau(starts,ends,sigma,free_end);weights=result['packed_weights'].reshape(-1).float().cpu().numpy();ns=near+result['packed_spacing_starts'].reshape(-1).float().cpu().numpy()*(far-near);ne=near+result['packed_spacing_ends'].reshape(-1).float().cpu().numpy()*(far-near);nm=(ns+ne)/2
        native_pre=float(weights[nm<free_end].sum());expected=float((w*mid).sum()/max(w.sum(),1e-20));median=float(mid[min(np.searchsorted(np.cumsum(w),.5*w.sum()),len(mid)-1)]);native_cond=float((weights*nm).sum()/max(weights.sum(),1e-20))
        free_mask=mid<free_end;positions_np=origin+mid[:,None]*direction;valid,ijk=indices(positions_np[free_mask]);unique=np.unique(ijk,axis=0);free_counts[*unique.T]+=1
        admitted=grid[0,*ijk.T];surface_cell=np.floor((point-box[:3])/(box[3:]-box[:3])*shape).astype(int)
        row=dict(observation=int(observation),point_index=int(obs['point_indices'][observation]),camera=name,uv=uv.tolist(),witness_distance=distance,perpendicular_error=perpendicular,near=near,far=far,surface_margin=margin,free_end=free_end,free_optical_thickness=tau,free_opacity=float(-np.expm1(-tau)),T_at_free_end=float(np.exp(-tau)),native_weight_before_margin=native_pre,uniform_conditional_depth=expected,uniform_median_depth=median,native_conditional_depth=native_cond,uniform_depth_relative_error=(expected-distance)/distance,native_depth_relative_error=(native_cond-distance)/distance,free_sample_occupied_fraction=float(admitted.mean()),witness_cell_occupied=bool(grid[0,*surface_cell]),sigma_first_10pct_median=float(np.median(sigma[(mid-near)<.1*(distance-near)])),saturation=float(result['adaptive_saturation_rate']) if 'adaptive_saturation_rate' in result else None)
        rows.append(row);traces[f'{j}_starts']=starts;traces[f'{j}_ends']=ends;traces[f'{j}_sigma']=sigma;traces[f'{j}_alpha']=alpha;traces[f'{j}_T']=T;traces[f'{j}_weights']=w;traces[f'{j}_native_starts']=ns;traces[f'{j}_native_ends']=ne;traces[f'{j}_native_weights']=weights
        if j<16:
            from render_marshawn_previews import display_rgb
            source=ds.get_image_float32(ci).cpu().numpy()[...,:3];source=display_rgb(source,a.display_transform);gt=Image.fromarray(np.rint(np.clip(source,0,1)*255).astype(np.uint8));draw=ImageDraw.Draw(gt);x,y=uv;draw.ellipse((x-12,y-12,x+12,y+12),outline='red',width=4);gt.thumbnail((480,300));panel=Image.new('RGB',(640,550),'#202020');panel.paste(gt,(0,0));d=ImageDraw.Draw(panel)
            chart=(25,330,610,505);d.rectangle(chart,outline='white')
            def line(values,color):d.line([(chart[0]+float(t-near)/(far-near)*(chart[2]-chart[0]),chart[3]-float(v)*(chart[3]-chart[1])) for t,v in zip(mid[::8],values[::8])],fill=color,width=2)
            line(T,'lime');line(np.minimum(1,sigma/max(float(np.percentile(sigma,99)),1e-9)),'orange')
            for t,color in [(distance,'cyan'),(free_end,'red'),(median,'magenta')]:
                xx=chart[0]+(t-near)/(far-near)*(chart[2]-chart[0]);d.line((xx,chart[1],xx,chart[3]),fill=color,width=2)
            d.text((15,305),f'{name} tau_free={tau:.2f} T_free={np.exp(-tau):.3g} depth/witness={expected/distance:.3f}',fill='white');d.text((15,520),'green T; orange scaled sigma; cyan witness; red margin; pink median',fill='white');panel.save(a.output/f'review_{j:03d}_{name}.jpg')
        if (j+1)%16==0:(a.output/'progress.json').write_text(json.dumps({'completed':j+1,'total':len(selected)}))
    if digest_model(model)!=before:raise ValueError('Frozen field changed')
    np.savez_compressed(a.output/'traces.npz',**traces);np.savez_compressed(a.output/'known_ray_evidence.npz',candidate_free_ray_counts=free_counts,aabb=box,witness_xyz=xyz,witness_inside=surface_valid,witness_cell_occupied=surface_occupied,witness_sigma=np.array(surface_sigma),observation_indices=selected)
    def stats(key):
        values=np.array([r[key] for r in rows]);return dict(min=float(values.min()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),max=float(values.max()))
    summary={key:stats(key) for key in ['free_optical_thickness','free_opacity','T_at_free_end','native_weight_before_margin','uniform_depth_relative_error','native_depth_relative_error','free_sample_occupied_fraction','sigma_first_10pct_median']}
    summary.update(rays=len(rows),cameras=len(set(r['camera'] for r in rows)),rays_free_opacity_gt_50pct=sum(r['free_opacity']>.5 for r in rows),rays_free_opacity_gt_90pct=sum(r['free_opacity']>.9 for r in rows),surface_witness_count=len(xyz),surface_witness_inside=int(surface_valid.sum()),surface_witness_cell_occupied_fraction=float(surface_occupied.mean()),candidate_free_cells=int((free_counts>0).sum()),candidate_free_cells_admitted_fraction=float(grid[0][free_counts>0].mean()))
    report=dict(sampling_mode=model._resolved_ray_sampling_mode(),implementation_sha256=file_sha(__file__),display_transform=a.display_transform,summary=summary,rows=rows,checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(a.config),observations_sha256=file_sha(a.observations),witnesses_sha256=file_sha(a.witnesses),transforms_sha256=file_sha(a.data/'transforms.json'),live_bounds=bounds,frozen_state_unchanged=True,state_sha256=before,uses_eval_rgb=False,limitations=['SfM train witnesses are fitted evidence, not independent geometry GT.','Observed opaque features suggest free camera-to-surface rays; transparency/occlusion and correspondence errors remain possible.','Specular floor reflections may triangulate virtual points behind the first physical surface; SfM witness depth is not independent opaque-surface ground truth.','3 voxel diagonals or 5% witness distance excluded around surface.','Candidate free cells contain only measured ray segments; no claim whole voxel empty.','Uniform samples query the same frozen learned field; no optimizer updates or mask changes.'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
