"""Local train-vetted floor diagnosis only: no plane carving, training or edits."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def prepare(out):
    from PIL import Image,ImageDraw
    from scipy.spatial import Delaunay,cKDTree
    from trace_marshawn_novel_rays import project
    from probe_marshawn_occupancy import file_sha
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((ROOT/'data/transforms.json').read_text());frames={f['physical_camera']:f for f in meta['frames']}
    source=ROOT/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz'
    z=np.load(source);n=z['normal'];d=float(z['offset']);points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz']
    ids=z['point_indices'][z['selected_inliers']];support=points[ids];hull=Delaunay(support[:,:2]);tree=cKDTree(support)
    chosen=[('HC',1050,800),('HC',1150,820),('HC',1250,800),('DC',1150,820),('DC',1300,800),
            ('KA',1000,850),('KA',1250,850),('AA',350,650),('AA',550,650),('GA',1350,500),('GA',1450,550),('OA',1450,750)]
    queries=[];canvases=[]
    for index,(name,x,y) in enumerate(chosen):
        f=frames[name];pose=np.array(f['transform_matrix']);origin=pose[:3,3]
        direction=pose[:3,:3]@np.array([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-1.]);direction/=np.linalg.norm(direction)
        t=-(origin@n+d)/(direction@n);xyz=origin+t*direction
        q=dict(id=index,camera=name,x=x,y=y,split='eval'if name in ['DC','HC','KA']else'train',
               plane_xyz=xyz.tolist(),plane_distance=float(t),inside_15point_XY_hull=bool(hull.find_simplex(xyz[:2])>=0),
               nearest_15point_distance=float(tree.query(xyz)[0]),train_sources=[])
        page=index//4;row=index%4
        if row==0:canvases.append(Image.new('RGB',(3*230,4*240),'black'))
        canvas=canvases[-1];draw=ImageDraw.Draw(canvas)
        for col,camera in enumerate(['AA','GA','OA']):
            frame=frames[camera];uv,dep,inside=project(frame,xyz[None]);u,v=uv[0];im=Image.open(ROOT/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png'/frame['file_path']).convert('RGB')
            box=(round(u)-96,round(v)-96,round(u)+96,round(v)+96);crop=im.crop(box);cd=ImageDraw.Draw(crop)
            cd.line((88,96,104,96),fill='red',width=1);cd.line((96,88,96,104),fill='red',width=1)
            canvas.paste(crop,(col*230,row*240+40));draw.text((col*230+3,row*240+4),f'q{index:02d} {name}({x},{y}) -> {camera}',fill='white')
            draw.text((col*230+3,row*240+21),f'{u:.1f},{v:.1f} hull={q["inside_15point_XY_hull"]}',fill='yellow')
            q['train_sources'].append(dict(camera=camera,uv=uv[0].tolist(),inside=bool(inside[0]),crop=box))
        queries.append(q)
    for i,canvas in enumerate(canvases):canvas.save(out/f'train_source_page{i+1}.jpg',quality=94)
    result=dict(plane_source=str(source),plane_source_sha256=file_sha(source),normal=n.tolist(),offset=d,
        support_point_indices=ids.tolist(),support_residual_max=float(abs(support@n+d).max()),
        signed_normal_uncertainty=.03,coordinate_units='Prepared world; not claimed meters',
        source_selection='Query gray patches from actual predictions; physical floor classification ONLYtrainRGBprojectedsourcecrops, no eval RGB fit',
        caveats=['Thin floor-covering panels have genuine wrinkles/reflection; raised crashmat excluded',
                 'Plane is approximate sparse15pointreference; outsidehull explicit extrapolation',
                 'Train frusta/projection do not establish occlusion-free evalray or exactdepth; manual crops must qualify'],queries=queries)
    (out/'selection_DRAFT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(prepared=str(out),queries=len(queries))))


def metrics(trace,origin,direction,n,d,halfwidth):
    from probe_marshawn_occupancy import integrate
    from audit_marshawn_body_rays import interval_mass
    starts,ends,sigma,weights=[np.asarray(trace[k],float)for k in ['starts','ends','sigma','weights']]
    mid=(starts+ends)/2;opacity=float(weights.sum());t0=-(origin@n+d)/(direction@n)
    dt=halfwidth/abs(direction@n);lo,hi=t0-dt,t0+dt
    q=[]
    if opacity>1e-10:
        for fraction in [.1,.5,.9]:
            j=min(int(np.searchsorted(np.cumsum(weights),opacity*fraction)),len(mid)-1)
            q.append(dict(quantile=fraction,t=float(mid[j]),signed_height=float((origin+direction*mid[j])@n+d)))
    heights=(origin+mid[:,None]*direction)@n+d
    return dict(samples=len(mid),opacity=opacity,plane_t=float(t0),plane_band_t=[float(lo),float(hi)],
        before_plane_band_mass=interval_mass(starts,ends,sigma,.01,lo),
        plane_band_mass=interval_mass(starts,ends,sigma,max(.01,lo),hi),
        after_plane_band_mass=interval_mass(starts,ends,sigma,max(.01,hi),1e4),
        mean_signed_height=float((weights*heights).sum()/opacity)if opacity>1e-10 else None,
        depth_mean=float((weights*mid).sum()/opacity)if opacity>1e-10 else None,termination_quantiles=q,
        peak_sigma=float(sigma.max(initial=0)),weight_max_error=trace.get('weight_max_error'))


def trace(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_direct_pq_STAGED/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from nerfstudio.cameras.rays import RaySamples,Frustums
    from marshawn_release_common import trace_profiles,ray_box_interval,native_bounds
    from probe_marshawn_occupancy import digest_model,file_sha,integrate
    selection=json.loads(a.selection.read_text());queries=selection['queries'];n=np.array(selection['normal']);d=selection['offset']
    if not selection.get('manual_train_source_review_passed'):raise ValueError('Explicit manualtrain-source review required')
    torch.set_num_threads(2)
    def configure(cfg):cfg.eval_checkpoint=a.checkpoint;return cfg
    cfg,pipe,checkpoint,step=eval_setup(a.config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure)
    model=pipe.model;pipe.eval();before=digest_model(model);bounds=native_bounds(model);original=model.occupancy_grid.binaries.detach().clone()
    point_membership=None
    if a.kind=='DISTORTION':
        from marshawn_release_common import authoritative_grid,point_cells
        folder=ROOT/'native_png_linear_distortion_STAGED/grids';base=f'step-{step:09d}_grid'
        core,active,ema,boxes,policy=authoritative_grid(folder/(base+'.npz'),folder/(base+'.json'),checkpoint)
        if not np.array_equal(active,original.cpu().numpy())or not np.array_equal(ema.reshape(-1),model.occupancy_grid.occs.cpu().numpy()):raise ValueError('New sourcegrid binding mismatch')
        witnesses=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];body=np.load(ROOT/'final_pair/release_audit/radius0/body/all_witnesses.npz')
        if not np.array_equal(body['xyz'],witnesses):raise ValueError('Body witness ordering mismatch')
        cells,inside,occupied=point_cells(witnesses,active,boxes);subject=body['subject'].astype(bool)
        point_membership=dict(points=len(witnesses),body_points=int(subject.sum()),inside=int(inside.sum()),outside_indices=np.flatnonzero(~inside).tolist(),default_point_FN_indices=np.flatnonzero(inside&~occupied).tolist(),default_body_FN_indices=np.flatnonzero(subject&inside&~occupied).tolist(),source_grid_sha256=file_sha(folder/(base+'.npz')),source_grid_receipt_sha256=file_sha(folder/(base+'.json')),policy=policy['radius'],cascade=0,pointcoverage_not_physicalvolume_truth=True)
    if a.kind=='PNG':
        candidate=ROOT/'final_pair/release_audit/radius0/witness_union_halo1_candidate'
        receipt=json.loads((candidate/'receipt.json').read_text())
        if file_sha(checkpoint)!=receipt['checkpoint_sha256']or file_sha(candidate/'grid.npz')!=receipt['grid_archive_sha256']:raise ValueError('Candidatebinding mismatch')
        mask=np.load(candidate/'grid.npz')['active']
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.tensor(mask,device=model.device))
    background=model.renderer_rgb.background_color
    if isinstance(background,str):background=np.zeros(3)if background=='black'else None
    else:background=background.detach().cpu().numpy()
    if background is None:raise ValueError('Explicit fixedbackground required')
    active_digest=digest_model(model);a.output.mkdir(parents=True,exist_ok=False);rows=[];arrays={}
    try:
        for q in queries:
            dataset=getattr(pipe.datamanager,q['split']+'_dataset');ci=[Path(p).stem.split('_')[-1]for p in dataset.image_filenames].index(q['camera'])
            camera=dataset.cameras[ci:ci+1].to(model.device)
            ray=camera.generate_rays(0,coords=torch.tensor([[q['y']+.5,q['x']+.5]],device=model.device),keep_shape=False)
            origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy();colors=[]
            def hook(_module,_args,output):colors.append(output[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy())
            handle=model.field.register_forward_hook(hook)
            try:profiles=trace_profiles(model,ray)
            finally:handle.remove()
            if len(colors)!=2:raise ValueError('Expected native/allcells colors')
            for (label,p),rgb in zip(profiles.items(),colors):p['rgb_samples']=rgb
            entry,exit=ray_box_interval(origin,direction,model.scene_aabb.cpu().numpy().reshape(-1))
            with torch.no_grad():
                for count in [8192,16384]:
                    edges=torch.linspace(entry,exit,count+1,device=model.device);starts=edges[:-1,None];ends=edges[1:,None]
                    sample=RaySamples(frustums=Frustums(origins=ray.origins.expand(count,3),directions=ray.directions.expand(count,3),starts=starts,ends=ends,pixel_area=torch.ones((count,1),device=model.device)),camera_indices=ray.camera_indices.expand(count,1))
                    output=model.field(sample);sigma=output[FieldHeadNames.DENSITY].float().cpu().numpy().reshape(-1);rgb=output[FieldHeadNames.RGB].float().cpu().numpy()
                    st,en=starts[:,0].cpu().numpy(),ends[:,0].cpu().numpy();alpha,T,weights=integrate(st,en,sigma)
                    profiles['uniform'+str(count)]=dict(starts=st,ends=en,sigma=sigma,weights=weights,rgb_samples=rgb)
            row=dict(query=q,origin=origin.tolist(),direction=direction.tolist(),profiles={})
            for label,p in profiles.items():
                stat=metrics(p,origin,direction,n,d,.03);stat['actual_rgb']=((p['weights'][:,None]*p['rgb_samples']).sum(0)+(1-p['weights'].sum())*background).tolist();row['profiles'][label]=stat
                for key,v in p.items():
                    if isinstance(v,np.ndarray):arrays[f'q{q["id"]:02d}_{label}_{key}']=v
            if q['split']=='eval':
                x,y=q['x'],q['y']
                if a.kind=='DISTORTION':
                    folder=ROOT/'native_png_linear_distortion_STAGED'/('distortion_8k_floor_native_'+q['camera']);source=folder/f'eval_{q["camera"]}_buffers.npz';prior=json.loads((folder/'receipt.json').read_text());crop=prior['rows'][0]['bbox'];x-=crop[0];y-=crop[1]
                    if prior['checkpoint_sha256']!=file_sha(checkpoint):raise ValueError('Comparisonrender checkpoint changed')
                else:source=ROOT/'final_pair/release_audit/grid_candidate_hd_comparison/witness_union_halo1'/f'{q["camera"]}_buffers.npz'if a.kind=='PNG'else ROOT/'native_direct_pq_STAGED/best6000_HD'/f'eval_{q["camera"]}_buffers.npz'
                saved=np.load(source)['rgb'][y,x];error=float(abs(saved-np.array(row['profiles']['native']['actual_rgb'])).max());row['saved_RGB_max_error']=error;row['saved_RGB_source']=str(source)
                if error>2e-5:raise ValueError('Actual savedpixel mismatch '+str(error))
            rows.append(row)
            (a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(queries)))+'\n')
        if digest_model(model)!=active_digest:raise ValueError('Auditmutated model')
    finally:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(original)
        if digest_model(model)!=before:raise ValueError('Exactoriginal state restore failed')
    np.savez_compressed(a.output/'traces.npz',**arrays)
    result=dict(kind=a.kind,selection_sha256=file_sha(a.selection),checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),config_sha256=file_sha(a.config),tool_sha256=file_sha(__file__),step=step,live_bounds=bounds,state_unchanged=True,background=background.tolist(),rows=rows,default_point_membership=point_membership,
        limitations=selection['caveats'],no_global_carve_or_training=True)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(kind=a.kind,rays=len(rows),state_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','trace']);p.add_argument('--output',type=Path,required=True)
    for name in ['selection','config','checkpoint']:p.add_argument('--'+name,type=Path)
    p.add_argument('--kind',choices=['PNG','PQ','DISTORTION']);a=p.parse_args()
    prepare(a.output)if a.mode=='prepare'else trace(a)
