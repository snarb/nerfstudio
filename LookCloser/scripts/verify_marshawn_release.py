#!/usr/bin/env python3
"""Frozen native NGP release evidence: authoritative atlas, matched rays, novel path.

Never runs training callbacks, an objective or an optimizer. Stored radius1 is
used as-is. An all-cells ray counterfactual is temporary and exactly restored.
"""
from pathlib import Path
import argparse
import copy
import inspect
import json
import sys
import numpy as np
from PIL import Image, ImageDraw

from marshawn_release_common import (authoritative_grid, authoritative_labels,
    native_bounds, point_cells, ray_box_interval, trace_profiles, validate_review_path, edge_point_sets,
    load_camera_bundle)
from marshawn_occupancy_audit import write_atlas
from probe_marshawn_occupancy import file_sha, digest_model, integrate
from audit_marshawn_body_rays import geometry_hash, interval_mass

MATCHED_SELECTION_SHA = '06e131767fb9ecd09b15eb28664a1917b8b1c21d4b976dd288599d5953628cab'
TOOL_FILES = ['verify_marshawn_release.py','marshawn_release_common.py',
    'marshawn_occupancy_audit.py','probe_marshawn_occupancy.py','audit_marshawn_body_rays.py',
    'prepare_marshawn_matched_path.py']


def tool_sources():
    return {name:file_sha(Path(__file__).with_name(name)) for name in TOOL_FILES}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def atlas(args):
    core, active, ema, boxes, policy = authoritative_grid(args.grid, args.grid_receipt, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    labels = authoritative_labels(core, active)
    pages = write_atlas(active, boxes, args.output, args.cell_pixels, labels)
    witness_summary=None
    if args.witnesses:
        points=np.load(args.witnesses)['xyz'];ij,inside,occupied=point_cells(points,active,boxes)
        subject_ids=np.empty(0,dtype=int)
        if args.subject:
            from scipy.spatial import cKDTree
            distance,subject_ids=cKDTree(points).query(np.load(args.subject)['xyz'])
            if distance.max()>1e-8:raise ValueError('Atlas subject/world witness mismatch')
            subject_ids=np.unique(subject_ids)
        for page in pages:
            if page['level']!=0:continue
            axis='xyz'.index(page['axis']);selected=inside&(ij[:,axis]>=page['slice_start'])&(ij[:,axis]<=page['slice_end'])
            page['witness_points']=int(selected.sum());page['subject_witness_points']=int(np.isin(np.flatnonzero(selected),subject_ids).sum())
            page['active_point_misses']=int((selected&~occupied).sum())
        witness_summary=dict(source=str(args.witnesses),source_sha256=file_sha(args.witnesses),
            points=len(points),inside=int(inside.sum()),outside=int((~inside).sum()),active_point_misses=int((inside&~occupied).sum()),
            subject_points=len(subject_ids),subject_source_sha256=file_sha(args.subject) if args.subject else None,
            membership_cascade=0,geometry_truth=False)
    html = (args.output/'index.html').read_text().replace('blue admitted threshold core',
        'blue actual GPU-exported native core').replace('With EMA labels:', 'With authoritative source labels:')
    (args.output/'index.html').write_text(html)
    np.savez_compressed(args.output/'grid_aliases.npz', occupied=active, aabbs=boxes,
                        occs=ema, core=core, active=active, cascade_aabbs=boxes, ema=ema)
    report = dict(source_archive=str(args.grid), source_sha256=file_sha(args.grid),
        tool_source_sha256=tool_sources(),
        receipt=str(args.grid_receipt), receipt_sha256=file_sha(args.grid_receipt),
        source_policy=policy, no_additional_dilation=True, shape_lxyz=list(active.shape),
        aabbs=boxes.tolist(), core_cells=int(core.sum()), guard_cells=int((active & ~core).sum()),
        active_cells=int(active.sum()), skipped_unverified=int((~active).sum()), pages=pages,
        train_witness_support=witness_summary,
        palette=dict(blue='Actual exported GPU native core', amber='Declared one-cell guard',
                     black='Skipped, geometry unknown', red='Core skipped: invalid source policy'),
        limitations=['Authoritative mask describes traversal, not independent geometry truth.',
                     'CPU EMA reconstruction is diagnostic only and never controls atlas labels.'])
    write_json(args.output/'atlas.json', report)
    print(json.dumps(dict(pages=len(pages), active_cells=int(active.sum()), radius=policy['radius'])))


def runtime(args):
    # The CLI itself lives outside sealed packages. Prefer explicit source roots.
    for path in reversed(args.code_root):
        sys.path.insert(0, str(path.resolve()))
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    torch.set_num_threads(2)
    if args.memory_gib:
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    def configure(config):
        config.eval_checkpoint = args.checkpoint.resolve()
        config.pipeline.model.eval_num_rays_per_chunk = args.chunk_size
        if args.data:
            config.pipeline.datamanager.data = args.data
            config.pipeline.datamanager.dataparser.data = args.data
        return config
    config, pipe, checkpoint, step = eval_setup(args.config, test_mode='val',
        eval_num_rays_per_chunk=args.chunk_size, update_config_callback=configure)
    if checkpoint.resolve() != args.checkpoint.resolve():
        raise ValueError('Unexpected checkpoint loaded')
    pipe.eval()
    model = pipe.model
    bounds = native_bounds(model)
    normal = pipe.datamanager.train_dataparser_outputs
    if not np.allclose(normal.dataparser_transform.cpu().numpy(), np.eye(4)[:3], atol=1e-7) or normal.dataparser_scale != 1:
        raise ValueError('Prepared-world identity normalization required')
    if getattr(config.pipeline, 'final_grid_radius', None) not in (0, 1):
        raise ValueError('Explicit final source grid policy required')
    core, active, ema, boxes, policy = authoritative_grid(args.grid, args.grid_receipt, checkpoint)
    if int(config.pipeline.final_grid_radius) != int(policy['radius']):
        raise ValueError('Config/export radius mismatch')
    if not np.array_equal(model.occupancy_grid.binaries.detach().cpu().numpy(), active):
        raise ValueError('Loaded mask differs from authoritative stored active mask; possible double halo')
    if not np.array_equal(model.occupancy_grid.occs.detach().cpu().numpy().reshape(ema.shape), ema):
        raise ValueError('Loaded EMA differs from checkpoint-bound export')
    if getattr(model.config, 'rgb_output_activation', None) != 'sigmoid':
        raise ValueError('This release adapter requires the scalar SDR sigmoid color contract')
    if not np.allclose(boxes[0].reshape(2, 3), bounds['model.scene_box.aabb'], atol=1e-6, rtol=0):
        raise ValueError('Export/runtime bounds mismatch')
    sources = {name:dict(path=inspect.getsourcefile(type(module)),
        sha256=file_sha(inspect.getsourcefile(type(module)))) for name,module in
        [('pipeline',pipe), ('model',model), ('field',model.field), ('sampler',model.sampler)]}
    for name,key in [('model','native_model_source_sha256'), ('field','native_field_source_sha256')]:
        if key in policy and sources[name]['sha256'] != policy[key]:
            raise ValueError('Frozen runtime source mismatch: '+name)
    receipt = dict(config=str(args.config), config_sha256=file_sha(args.config),
        tool_source_sha256=tool_sources(),
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint), step=step,
        implementation_sha256=file_sha(__file__), adapter_sha256=file_sha(Path(__file__).with_name('marshawn_release_common.py')),
        live_bounds=bounds, source_policy=policy, loaded_radius1_redilated=False,
        runtime_sources=sources, normalization=dict(scale=1., transform=normal.dataparser_transform.tolist()))
    return pipe, core, active, boxes, receipt


def profile_metrics(trace, near, far, truth):
    starts, ends, sigma, weights = [np.asarray(trace[k], float) for k in ('starts','ends','sigma','weights')]
    opacity = float(weights.sum())
    midpoint = (starts+ends)/2
    expected = float(np.sum(weights*midpoint)/opacity) if opacity > 1e-10 else None
    median = float(midpoint[min(np.searchsorted(np.cumsum(weights), opacity*.5), len(midpoint)-1)]) if opacity > 1e-10 else None
    free_end = max(near, truth-max(.01,.05*truth))
    band = max(.005,.01*truth)
    return dict(samples=len(starts), zero_length_samples=int((ends<=starts).sum()), opacity=opacity,
        conditional_depth=expected, conditional_depth_relative_error=(expected-truth)/truth if expected is not None else None,
        median_depth=median, median_depth_relative_error=(median-truth)/truth if median is not None else None,
        free_mass=interval_mass(starts,ends,sigma,near,free_end),
        free_optical_thickness=float(np.sum(sigma*np.maximum(0,np.minimum(ends,free_end)-np.maximum(starts,near)))),
        surface_band_mass=interval_mass(starts,ends,sigma,max(near,truth-band),min(far,truth+band)),
        free_end=free_end, surface_band_halfwidth=band)


def summarize(rows):
    result = {}
    for profile in rows[0]['profiles']:
        values = [row['profiles'][profile] for row in rows]
        stats = {}
        for key in ['free_mass','free_optical_thickness','surface_band_mass',
                    'conditional_depth_relative_error','median_depth_relative_error']:
            array = np.array([row[key] for row in values if row[key] is not None])
            stats[key] = dict(count=len(array), median=float(np.median(array)) if len(array) else None,
                p10=float(np.quantile(array,.1)) if len(array) else None,
                p90=float(np.quantile(array,.9)) if len(array) else None)
        stats.update(free_mass_gt_10pct=sum(row['free_mass']>.1 for row in values),
                     free_mass_gt_50pct=sum(row['free_mass']>.5 for row in values),
                     surface_band_mass_ge_50pct=sum(row['surface_band_mass']>=.5 for row in values))
        result[profile] = stats
    return result


def save_crop(model, dataset, camera_index, uv, output, title):
    """Native pixel crop from the real field; GT is displayed separately only."""
    import torch
    image = dataset.get_image_float32(camera_index).cpu().numpy()[...,:3]
    height,width = image.shape[:2]
    x,y = np.floor(np.asarray(uv)-64).astype(int)
    x,y = int(np.clip(x,0,width-128)),int(np.clip(y,0,height-128))
    camera = copy.deepcopy(dataset.cameras[camera_index:camera_index+1]).to(model.device)
    camera.cx -= x; camera.cy -= y
    camera.width.fill_(128); camera.height.fill_(128)
    with torch.inference_mode():
        rgb = model.get_outputs_for_camera(camera)['rgb'].detach().float().cpu().numpy()
    gt = image[y:y+128,x:x+128]
    panel = Image.new('RGB',(512,280),'#202020')
    draw = ImageDraw.Draw(panel)
    draw.text((3,3),title+' GT | native field',fill='white')
    for col,value in enumerate([gt,rgb]):
        patch=Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
        panel.paste(patch,(col*256,24))
        u,v=(np.asarray(uv)-[x+.5,y+.5])*2
        draw.ellipse((col*256+u-6,24+v-6,col*256+u+6,24+v+6),outline='red',width=2)
    panel.save(output,quality=95)
    return dict(path=Path(output).name,bbox_xyxy=[x,y,x+128,y+128],camera_index=int(camera_index),uv=np.asarray(uv).tolist())


def body(args):
    import torch
    from scipy.spatial import cKDTree
    pipe,core,active,boxes,receipt = runtime(args)
    model=pipe.model;before=digest_model(model)
    ds=pipe.datamanager.train_dataset
    data=args.data or Path(pipe.config.datamanager.dataparser.data)
    transforms=json.loads((data/'transforms.json').read_text())
    by={Path(f['file_path']).name:f for f in transforms['frames']}
    frames=[by[Path(path).name] for path in ds.image_filenames]
    name_to_camera={f['physical_camera']:i for i,f in enumerate(frames)}
    obs=np.load(args.observations,allow_pickle=False);xyz=np.load(args.witnesses)['xyz']
    subject=np.load(args.subject)['xyz'];distance,subject_ids=cKDTree(xyz).query(subject)
    if distance.max()>1e-8:raise ValueError('Subject/world witness mismatch')
    subject_ids=np.unique(subject_ids);point_idx=obs['point_indices'];camera_idx=obs['camera_indices'];names=obs['camera_names'];uv=obs['uv']
    bindings=dict(observations_sha256=file_sha(args.observations),points_sha256=file_sha(args.witnesses),
        subject_sha256=file_sha(args.subject),geometry_sha256=geometry_hash(frames))
    if file_sha(args.selection)!=MATCHED_SELECTION_SHA:raise ValueError('Exact original BODY128 selection SHA required')
    with np.load(args.selection,allow_pickle=False) as saved:
        for key,value in bindings.items():
            if str(saved[key].item())!=value:raise ValueError('Matched selection binding mismatch: '+key)
        selected=saved['observation_idx'].copy()
    if len(selected)!=128 or len(np.unique(selected))!=128 or len(np.unique(point_idx[selected]))!=112 or len(np.unique(camera_idx[selected]))!=63:
        raise ValueError('Original 128/112/63 matched selection contract violated')
    args.output.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copyfile(args.selection,args.output/'selection.npz')
    ij,inside,occupied=point_cells(xyz,active,boxes);_,_,in_core=point_cells(xyz,core,boxes)
    with torch.no_grad():
        sigma=np.concatenate([model.field.density_fn(torch.tensor(p,device=model.device,dtype=torch.float32)).float().cpu().numpy().reshape(-1) for p in np.array_split(xyz,max(1,int(np.ceil(len(xyz)/8192))))])
    np.savez_compressed(args.output/'all_witnesses.npz',point_idx=np.arange(len(xyz)),xyz=xyz,
        cell_xyz=ij,inside=inside,occupied=occupied,core=in_core,sigma=sigma,subject=np.isin(np.arange(len(xyz)),subject_ids))
    cameras=ds.cameras.to(model.device);rows=[];traces={}
    for number,obs_id in enumerate(selected):
        source_cam=int(camera_idx[obs_id]);name=str(names[source_cam]);ci=name_to_camera[name];point_id=int(point_idx[obs_id]);point=xyz[point_id]
        ray=cameras[ci:ci+1].generate_rays(0,coords=torch.tensor(uv[obs_id][::-1].copy()[None],device=model.device,dtype=torch.float32),keep_shape=False)
        origin=ray.origins[0].detach().cpu().numpy();direction=ray.directions[0].detach().cpu().numpy()
        near,far=ray_box_interval(origin,direction,boxes[0],float(model.config.near_plane),float(model.config.far_plane))
        truth=float(np.dot(point-origin,direction));perpendicular=float(np.linalg.norm(point-origin-truth*direction))
        if not near<truth<far or perpendicular>max(1e-4,truth*2/frames[ci]['fl_x']):raise ValueError('Visible witness ray calibration/bounds mismatch')
        profiles=trace_profiles(model,ray)
        with torch.no_grad():
            for factor in [1,2]:
                count=args.uniform_samples*factor;edges=torch.linspace(near,far,count+1,device=model.device)
                mid=(edges[:-1]+edges[1:])/2;density=model.field.density_fn(ray.origins+mid[:,None]*ray.directions).float().cpu().numpy().reshape(-1)
                starts,ends=edges[:-1].cpu().numpy(),edges[1:].cpu().numpy();alpha,trans,weights=integrate(starts,ends,density)
                profiles['uniform'+str(count)]=dict(starts=starts,ends=ends,sigma=density,alpha=alpha,T=trans,weights=weights)
        row=dict(observation_idx=int(obs_id),point_idx=point_id,camera_idx=source_cam,camera=name,
            uv=uv[obs_id].tolist(),witness_xyz=point.tolist(),witness_depth=truth,perpendicular_error=perpendicular,
            cell_xyz=ij[point_id].tolist(),cell_occupied=bool(occupied[point_id]),cell_core=bool(in_core[point_id]),
            witness_sigma=float(sigma[point_id]),profiles={})
        for label,trace in profiles.items():
            row['profiles'][label]=profile_metrics(trace,near,far,truth)
            if 'weight_max_error' in trace:row['profiles'][label]['actual_weight_max_error']=trace['weight_max_error']
            for key,value in trace.items():
                if isinstance(value,np.ndarray):traces[f'r{number:03d}_{label}_{key}']=value
        row['uniform_convergence']=dict(free_mass_delta=row['profiles']['uniform'+str(args.uniform_samples*2)]['free_mass']-row['profiles']['uniform'+str(args.uniform_samples)]['free_mass'],
            surface_band_mass_delta=row['profiles']['uniform'+str(args.uniform_samples*2)]['surface_band_mass']-row['profiles']['uniform'+str(args.uniform_samples)]['surface_band_mass'])
        rows.append(row)
        if (number+1)%16==0:write_json(args.output/'progress.json',dict(rays_completed=number+1,total=128))
    np.savez_compressed(args.output/'traces.npz',**traces)
    crops=[];chosen=[]
    for camera in np.unique(camera_idx[selected])[np.linspace(0,62,12).astype(int)]:
        chosen.append(int(selected[np.flatnonzero(camera_idx[selected]==camera)[0]]))
    for number,obs_id in enumerate(chosen):
        name=str(names[camera_idx[obs_id]]);ci=name_to_camera[name]
        crops.append(save_crop(model,ds,ci,uv[obs_id],args.output/f'body_{number:02d}_{name}.jpg',f'{name} point{point_idx[obs_id]}'))
    edges={}
    for label,ids in edge_point_sets(inside,occupied,in_core,sigma,subject_ids).items():
        edges[label]=[]
        for point_id in ids:
            candidates=np.flatnonzero(point_idx==point_id)
            if not len(candidates):continue
            obs_id=int(candidates[0]);name=str(names[camera_idx[obs_id]]);ci=name_to_camera[name]
            crop=save_crop(model,ds,ci,uv[obs_id],args.output/f'{label}_{len(edges[label]):02d}_{name}_p{point_id}.jpg',f'{name} {label} {point_id} sigma{sigma[point_id]:.1f}')
            crop.update(point_idx=int(point_id),observation_idx=obs_id,sigma=float(sigma[point_id]),cell_xyz=ij[point_id].tolist(),kind=label);edges[label].append(crop)
    for title,items in [('body12',crops),*edges.items()]:
        if not items:continue
        contact=Image.new('RGB',(4*512,int(np.ceil(len(items)/4))*280),'#202020')
        for index,crop in enumerate(items):contact.paste(Image.open(args.output/crop['path']),(index%4*512,index//4*280))
        contact.save(args.output/(title+'_contact.jpg'),quality=95)
    if digest_model(model)!=before:raise ValueError('Frozen field/grid changed during release audit')
    def counts(ids):
        return dict(points=len(ids),inside=int(inside[ids].sum()),outside=int((~inside[ids]).sum()),
            active_cell_misses=int((inside[ids]&~occupied[ids]).sum()),core_cell_misses=int((inside[ids]&~in_core[ids]).sum()),
            missed_points=[dict(point_idx=int(i),cell_xyz=ij[i].tolist(),sigma=float(sigma[i])) for i in ids[inside[ids]&~occupied[ids]]])
    receipt.update(bindings=bindings,transforms_sha256=file_sha(data/'transforms.json'),selection_sha256=file_sha(args.selection),
        selected_rays=128,selected_unique_points=112,cameras=63,summary=summarize(rows),rows=rows,
        policy=dict(free_margin='max(.01,.05*witnessDepth)',surface_band='+-max(.005,.01*witnessDepth)',
            all_cells='Actual native sampler and field with temporary all-true mask, exactly restored',uniform_samples=[args.uniform_samples,args.uniform_samples*2]),
        all_witnesses=counts(np.arange(len(xyz))),subject_witnesses=counts(subject_ids),body_crops=crops,
        fn_edge_crops=edges['active_miss'],guard_edge_crops=edges['guard_admitted_core_miss'],
        frozen_state_unchanged=True,state_sha256=before,no_gt_compositing=True,uses_eval_rgb=False,
        limitations=['Visible fitted SfM witness rays are not hidden geometry truth.',
            'Core/guard are traversal policy labels, not occupied/empty ground truth.',
            'Outside-AABB points are reported separately, not counted as skipped in-bounds cells.',
            'Subject ROI contains some prop features; not a semantic body mask.'])
    write_json(args.output/'audit.json',receipt)
    print(json.dumps(dict(step=receipt['step'],radius=receipt['source_policy']['radius'],
        summary=receipt['summary'],all_points=receipt['all_witnesses']['points'],
        all_misses=receipt['all_witnesses']['active_cell_misses'],body_misses=receipt['subject_witnesses']['active_cell_misses'])))


def novel_path(args):
    import torch
    pipe,core,active,boxes,receipt=runtime(args);model=pipe.model;before=digest_model(model)
    data=args.data or Path(pipe.config.datamanager.dataparser.data)
    meta=json.loads((data/'transforms.json').read_text());train=set(meta['train_filenames'])
    frames=[frame for frame in meta['frames'] if frame['file_path'] in train]
    if args.camera_bundle:
        from nerfstudio.cameras.cameras import Cameras,CameraType
        arrays,path_contract=load_camera_bundle(args.camera_bundle,args.camera_bundle_receipt,frames)
        cameras=Cameras(camera_to_worlds=torch.tensor(arrays['camera_to_worlds'],dtype=torch.float32),
            fx=torch.tensor(arrays['fx'],dtype=torch.float32),fy=torch.tensor(arrays['fy'],dtype=torch.float32),
            cx=torch.tensor(arrays['cx'],dtype=torch.float32),cy=torch.tensor(arrays['cy'],dtype=torch.float32),
            width=torch.tensor(arrays['width']),height=torch.tensor(arrays['height']),camera_type=CameraType.PERSPECTIVE)
        for key in ['fx','fy','cx','cy']:
            if not np.allclose(getattr(cameras,key).numpy().reshape(-1),arrays[key],atol=2e-4,rtol=0):
                raise ValueError('Direct Cameras intrinsics differ from exact bundle')
        source_width=int(arrays['width'][0]);source_height=int(arrays['height'][0])
        if len(np.unique(arrays['width']))!=1 or len(np.unique(arrays['height']))!=1:raise ValueError('Uniform prepared path raster required')
        camera_source_sha=file_sha(args.camera_bundle)
    else:
        from nerfstudio.cameras.camera_paths import get_path_from_json
        path_contract=validate_review_path(args.camera_path,args.path_receipt,frames)
        path_contract.update(path_class='Wide-FOV extrapolation stress',matched_capture_intrinsics=False)
        path=json.loads(args.camera_path.read_text());cameras=get_path_from_json(path)
        source_width,source_height=int(path['render_width']),int(path['render_height'])
        camera_source_sha=file_sha(args.camera_path)
    if cameras.size!=24:raise ValueError('Exactly24 predeclared novel views required')
    cameras.rescale_output_resolution(args.width/source_width)
    cameras=cameras.to(model.device)
    args.output.mkdir(parents=True,exist_ok=False);rows=[];images=[]
    for index in range(24):
        with torch.inference_mode():outputs=model.get_outputs_for_camera(cameras[index:index+1])
        arrays={key:outputs[key].detach().float().cpu().numpy() for key in ['rgb','depth','accumulation']}
        if not all(np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite novel render')
        np.savez_compressed(args.output/f'view_{index:02d}_buffers.npz',**arrays)
        image=Image.fromarray(np.rint(np.clip(arrays['rgb'],0,1)*255).astype(np.uint8));image.save(args.output/f'view_{index:02d}.png');images.append(image)
        rows.append(dict(index=index,size=list(image.size),camera_to_world=cameras.camera_to_worlds[index].cpu().tolist(),
            rendered_intrinsics={key:float(getattr(cameras,key)[index].item()) for key in ['fx','fy','cx','cy']},
            rgb_sha256=file_sha(args.output/f'view_{index:02d}.png'),buffers_sha256=file_sha(args.output/f'view_{index:02d}_buffers.npz')))
        write_json(args.output/'progress.json',dict(rendered=index+1,total=24))
    def contact(indices,columns,tile_width,filename):
        tile_height=round(images[0].height*tile_width/images[0].width)
        canvas=Image.new('RGB',(columns*tile_width,int(np.ceil(len(indices)/columns))*(tile_height+24)),'#202020');draw=ImageDraw.Draw(canvas)
        for n,index in enumerate(indices):
            x,y=n%columns*tile_width,n//columns*(tile_height+24);canvas.paste(images[index].resize((tile_width,tile_height)),(x,y+24));draw.text((x+4,y+5),f'novel{index:02d} radius{receipt["source_policy"]["radius"]} step{receipt["step"]}',fill='white')
        canvas.save(args.output/filename,quality=95)
    contact(list(range(24)),6,320,'contact24.jpg')
    for start in range(0,24,8):contact(list(range(start,start+8)),2,args.width,f'page_{start:02d}_{start+7:02d}.jpg')
    if digest_model(model)!=before:raise ValueError('Novel rendering changed frozen state')
    receipt.update(camera_path=str(args.camera_path) if args.camera_path else None,
        camera_bundle=str(args.camera_bundle) if args.camera_bundle else None,
        path_contract=path_contract,camera_source_sha256=camera_source_sha,views=24,
        source_raster=[source_width,source_height],render_width=args.width,rows=rows,
        rgb='Actual native field renderer output, sRGB scalar-target contract, identity display',
        gt_compositing=False,gt_rgb_used_for_rendering=False,dataset_loader_may_cache_source_rgb=True,
        frozen_state_unchanged=True,state_sha256=before,
        limitations=['Near-training novel path has no novel-view ground-truth RGB.',
            'Depth/opacity are renderer outputs, not independent surface evidence.'])
    write_json(args.output/'render_receipt.json',receipt)
    print(json.dumps(dict(step=receipt['step'],views=24,output=str(args.output))))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['atlas','body','path']:
        cmd=sub.add_parser(name)
        for key in ['grid','grid-receipt','output']:cmd.add_argument('--'+key,type=Path,required=True)
        cmd.add_argument('--checkpoint',type=Path,required=name!='atlas')
        if name=='atlas':
            cmd.add_argument('--cell-pixels',type=int,default=2)
            cmd.add_argument('--witnesses',type=Path);cmd.add_argument('--subject',type=Path)
            continue
        cmd.add_argument('--config',type=Path,required=True);cmd.add_argument('--data',type=Path)
        cmd.add_argument('--code-root',type=Path,action='append',default=[])
        cmd.add_argument('--chunk-size',type=int,default=512);cmd.add_argument('--memory-gib',type=float,default=24)
        if name=='body':
            for key in ['selection','observations','witnesses','subject']:cmd.add_argument('--'+key,type=Path,required=True)
            cmd.add_argument('--uniform-samples',type=int,default=8192)
        else:
            cameras=cmd.add_mutually_exclusive_group(required=True)
            cameras.add_argument('--camera-path',type=Path)
            cameras.add_argument('--camera-bundle',type=Path)
            cmd.add_argument('--path-receipt',type=Path)
            cmd.add_argument('--camera-bundle-receipt',type=Path)
            cmd.add_argument('--width',type=int,default=640)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    {'atlas':atlas,'body':body,'path':novel_path}[args.command](args)
