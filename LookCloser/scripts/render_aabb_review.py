"""Reusable read-only AABB review for Nerfstudio packed volumetric models.

This tests reconstructed geometry, not ground-truth depth or hidden surfaces.
No masks or source RGB affect rendered pixels. Field normalization stays fixed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aabb_visualization import (overlay_depth_aware, validate_bounds,
                                validate_rigid_frame, camera_in_box_frame)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_model(model):
    import torch
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def display_rgb(value, transform):
    if transform == 'linear-reinhard-srgb':
        value = np.maximum(value,0)
        value = value/(1+value)
        value = np.where(value<=.0031308,12.92*value,1.055*value**(1/2.4)-.055)
    return np.rint(np.clip(value,0,1)*255).astype(np.uint8)


def first_opaque_depth(starts, ends, sigma, trans, alpha, ids, num_rays):
    """First absolute-opacity 0.5 crossing, interpolated inside its interval."""
    import torch
    crosses=(trans*(1-alpha)<=.5)&(sigma>0)
    crossing=starts+torch.log((trans/.5).clamp_min(1))/sigma.clamp_min(1e-30)
    crossing=crossing.clamp(min=starts,max=ends)
    hit=torch.where(crosses,crossing,torch.full_like(starts,torch.inf))
    depths=torch.full((num_rays,),torch.inf,device=hit.device,dtype=hit.dtype)
    depths.scatter_reduce_(0,ids,hit,reduce='amin',include_self=True)
    return depths


def main(a):
    helper_sha=sha(__file__)
    overlay_path=Path(__file__).with_name('aabb_visualization.py')
    overlay_sha=sha(overlay_path)
    # Explicit paths let historical snapshots import their own model/parser code.
    sys.path[:0] = [str(p.resolve()) for p in a.python_path]
    import torch
    import nerfacc
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    torch.set_num_threads(2)
    if sha(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Wrong immutable checkpoint')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'code').mkdir()
    shutil.copy2(__file__,a.output/'code'/Path(__file__).name)
    shutil.copy2(overlay_path,a.output/'code'/overlay_path.name)
    box_document = json.loads(a.box.read_text())
    bounds = validate_bounds(box_document['aabb'])
    world_to_box = validate_rigid_frame(box_document.get('world_to_box'))

    def configure(c):
        c.eval_checkpoint = a.checkpoint
        c.load_checkpoint = c.load_dir = c.load_step = c.load_config = None
        return c

    cfg, pipeline, actual, step = eval_setup(a.config, test_mode='val',
        eval_num_rays_per_chunk=512, update_config_callback=configure)
    if actual.resolve() != a.checkpoint.resolve():
        raise ValueError('Loaded different checkpoint')
    pipeline.eval(); model = pipeline.model; model.current_train_step = step
    from nerfstudio.model_components.ray_samplers import VolumetricSampler
    if not isinstance(model.sampler,VolumetricSampler):
        raise ValueError('Partition adapter requires a packed VolumetricSampler model')
    if not isinstance(model.renderer_rgb.background_color,str) or model.renderer_rgb.background_color not in ['black','random']:
        raise ValueError('Native parity currently requires black/random-eval-black background')
    if model.sampler.training or model.sampler.get_sigma_fn(None, None) is not None:
        raise ValueError('Shared sample generation must not use opacity early termination')
    before = digest_model(model)
    box = torch.tensor(bounds, device=model.device, dtype=torch.float32)
    frame = torch.tensor(world_to_box, device=model.device, dtype=torch.float32)
    names = ['full', 'inside', 'outside', 'inside_contribution', 'outside_contribution']
    ds = pipeline.datamanager.eval_dataset
    candidates=[i for i,p in enumerate(ds.image_filenames)
                if str(i)==a.camera or p.stem==a.camera or p.stem.endswith('_'+a.camera)]
    if len(candidates)!=1:
        raise ValueError('Specify one unambiguous camera index or filename stem')
    ci=candidates[0]
    camera = copy.deepcopy(ds.cameras[ci:ci+1]).to(model.device)
    if int(camera.camera_type.item()) != 1:
        raise ValueError('Depth overlay requires a perspective camera')
    if camera.distortion_params is not None and torch.count_nonzero(camera.distortion_params):
        raise ValueError('Depth overlay requires an undistorted raster')
    w,h = int(camera.width.item()), int(camera.height.item())
    c=dict(w=w,h=h,fl_x=float(camera.fx.item()),fl_y=float(camera.fy.item()),
           cx=float(camera.cx.item()),cy=float(camera.cy.item()),
           transform_matrix=camera.camera_to_worlds[0].cpu().tolist())
    box_camera = camera_in_box_frame(c, world_to_box)
    rays = camera.generate_rays(0, keep_shape=True).flatten()
    buffers = {k: np.zeros((h*w,4),np.float32) for k in names}
    queries = json.loads(a.queries.read_text()) if a.queries else {}
    if any(not(0<=x<w and 0<=y<h) for x,y in queries.values()):
        raise ValueError('Query outside actual calibrated raster')
    surface_depth=np.full(h*w,np.inf,np.float32)
    trace, parity, count = {}, [], 0
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0,h*w,a.chunk):
            rb = rays[offset:offset+a.chunk]
            if model.collider is not None:
                rb = model.collider(rb)
            samples, ids = model.sampler(ray_bundle=rb,
                near_plane=model.config.near_plane, far_plane=model.config.far_plane,
                render_step_size=model.config.render_step_size,
                alpha_thre=model.config.alpha_thre, cone_angle=model.config.cone_angle)
            values = model.field(samples)
            xyz = samples.frustums.get_positions()
            box_xyz = xyz @ frame[:3,:3].T + frame[:3,3]
            inside = ((box_xyz>=box[0]) & (box_xyz<=box[1])).all(-1)
            sigma = values[FieldHeadNames.DENSITY][...,0]
            rgb = values[FieldHeadNames.RGB]
            starts,ends = samples.frustums.starts[...,0],samples.frustums.ends[...,0]
            pack = nerfacc.pack_info(ids,len(rb))
            weights = {}
            for name, s in [('full',sigma),('inside',sigma*inside),('outside',sigma*(~inside))]:
                weight,trans,alpha = nerfacc.render_weight_from_density(
                    t_starts=starts,t_ends=ends,sigmas=s,packed_info=pack)
                weights[name]=weight
                if name=='full':
                    # First interval crossing absolute accumulated opacity 0.5.
                    # Interpolate within its constant-density integration segment.
                    depths=first_opaque_depth(starts,ends,sigma,trans,alpha,ids,len(rb))
                    surface_depth[offset:offset+len(rb)]=depths.cpu().numpy()
            weights['inside_contribution'] = weights['full']*inside
            weights['outside_contribution'] = weights['full']*(~inside)
            for name,weight in weights.items():
                color = nerfacc.accumulate_along_rays(weight,values=rgb,ray_indices=ids,n_rays=len(rb))
                alpha = nerfacc.accumulate_along_rays(weight,values=None,ray_indices=ids,n_rays=len(rb))
                buffers[name][offset:offset+len(rb)] = torch.cat([color,alpha],-1).float().cpu().numpy()
            if offset==0 or any(offset<=y*w+x<offset+len(rb) for x,y in queries.values()):
                native = model.get_outputs(rb)
                error = float(np.max(np.abs(native['rgb'].float().cpu().numpy()-buffers['full'][offset:offset+len(rb),:3])))
                parity.append(error)
                if error>2e-5:
                    raise ValueError('Custom/native compositing mismatch '+str(error))
            for name,(x,y) in queries.items():
                rid = y*w+x-offset
                if 0<=rid<len(rb):
                    use=ids==rid
                    arrays=dict(xyz=xyz[use].cpu().numpy(),start=starts[use].cpu().numpy(),
                        end=ends[use].cpu().numpy(),sigma=sigma[use].cpu().numpy(),
                        inside=inside[use].cpu().numpy(),rgb=rgb[use].cpu().numpy(),
                        weight=weights['full'][use].cpu().numpy())
                    np.savez_compressed(a.output/(name+'_ray.npz'),**arrays)
                    trace[name]=dict(pixel=[x,y],samples=int(use.sum()),
                        original_alpha_inside=float(weights['inside_contribution'][use].sum()),
                        original_alpha_outside=float(weights['outside_contribution'][use].sum()),
                        peak_xyz=arrays['xyz'][np.argmax(arrays['weight'])].tolist())
            count+=len(ids)
            if offset//a.chunk%256==0:
                progress=dict(pixels=offset+len(rb),total=h*w,seconds=time.monotonic()-started)
                (a.output/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
    for name,buf in buffers.items():
        buffers[name]=buf.reshape(h,w,4)
        Image.fromarray(display_rgb(buffers[name][...,:3],a.display_transform)).save(a.output/(a.camera+'_'+name+'.png'))
    surface_depth=surface_depth.reshape(h,w)
    np.savez_compressed(a.output/'raw_partition.npz',**buffers,surface_depth=surface_depth)
    split_error=float(np.max(np.abs(buffers['full']-buffers['inside_contribution']-buffers['outside_contribution'])))
    if split_error>2e-5:
        raise ValueError('Partition sum verification failed: '+str(split_error))
    if digest_model(model)!=before or sha(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Immutable model/checkpoint verification failed')
    source_array=ds.get_image_float32(ci).detach().cpu().numpy()[...,:3]
    source=Image.fromarray(display_rgb(source_array,a.display_transform))
    panels=[('SOURCE '+a.camera,source),('FULL MODEL',Image.open(a.output/(a.camera+'_full.png'))),
            ('INSIDE AABB ONLY',Image.open(a.output/(a.camera+'_inside.png'))),
            ('OUTSIDE CONTRIBUTION',Image.open(a.output/(a.camera+'_outside_contribution.png')))]
    try: font=ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError: font=ImageFont.load_default()
    # Auto-frame the projected box, or use an explicit same-camera source crop.
    from aabb_visualization import edge_samples
    lines=edge_samples(bounds,box_camera)
    if a.crop:
        crop=tuple(a.crop)
    elif lines:
        uv=np.concatenate([line['uv'] for line in lines])
        crop=tuple(np.r_[np.maximum(np.floor(uv.min(0)-24),0),np.minimum(np.ceil(uv.max(0)+24),[w,h])].astype(int))
    else:
        crop=(0,0,w,h)
    if not 0<=crop[0]<crop[2]<=w or not 0<=crop[1]<crop[3]<=h:
        raise ValueError('Comparison crop must be inside the camera raster')
    for filename in ['comparison.png']:
        pw,ph=crop[2]-crop[0],crop[3]-crop[1]
        contact=Image.new('RGB',(pw*len(panels),ph+55),'#181818');draw=ImageDraw.Draw(contact)
        for i,(label,im) in enumerate(panels):
            contact.paste(im.crop(crop),(i*pw,55))
            draw.text((i*pw+4,6),label,font=font,fill='white')
        contact.save(a.output/filename)
    tolerance=a.depth_tolerance if a.depth_tolerance is not None else float(model.config.render_step_size)
    overlay,overlay_receipt=overlay_depth_aware(panels[1][1],bounds,box_camera,surface_depth,
        tolerance=tolerance,hidden=a.hidden_edges)
    overlay.save(a.output/(a.camera+'_box_visible.png'))
    # Expose the volume on black while testing visibility against the ORIGINAL
    # scene. Ground buried beneath the real floor stays occluded in this view too.
    inside_overlay,_=overlay_depth_aware(panels[2][1],bounds,box_camera,surface_depth,
        tolerance=tolerance,hidden=a.hidden_edges)
    inside_overlay.save(a.output/(a.camera+'_inside_box_visible.png'))
    receipt=dict(kind='ACTUAL_FIELD_AABB_PARTITION_NO_TRAINING',camera=a.camera,
        aabb=bounds.tolist(),world_to_box=world_to_box.tolist(),
        camera_box_frame=box_camera,box_sha256=sha(a.box),checkpoint=str(actual),
        checkpoint_sha256=a.checkpoint_sha256,step=step,source_helper_sha256=helper_sha,
        overlay_helper_sha256=overlay_sha,
        model_before_after_sha256=before,native_compositor_maxerror=max(parity),
        full_equals_inside_plus_outside_contributions_maxerror=split_error,
        samples=count,seconds=time.monotonic()-started,trace=trace,
        field_normalization_and_grid_unchanged=True,mask_or_source_compositing=False,updates=0,
        rule='Native sample midpoint transformed by world_to_box, inside exact AABB; zero other density then reintegrate. '
             'Contribution maps retain ORIGINAL full-scene transmittance and sum to full RGB/alpha.',
        boundary_discretization_world=float(model.config.render_step_size)/2,
        camera_runtime=c,display_transform=a.display_transform,overlay=overlay_receipt,
        surface_depth='first absolute accumulated opacity 0.5 crossing, unit-ray distance; inf for no crossing',
        limitation='Historical full-scene model, not physical ground truth; cannot certify hidden anatomy or bad learned geometry.')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','box','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--camera',required=True);p.add_argument('--chunk',type=int,default=512)
    p.add_argument('--python-path',type=Path,action='append',default=[])
    p.add_argument('--queries',type=Path,help='Optional JSON mapping label to [x,y] trace pixel')
    p.add_argument('--crop',type=int,nargs=4)
    p.add_argument('--display-transform',choices=['identity','linear-reinhard-srgb'],default='identity')
    p.add_argument('--depth-tolerance',type=float)
    p.add_argument('--hidden-edges',choices=['omit','ghost'],default='omit')
    main(p.parse_args())
