"""Read-only field partition: identical rays/samples, density inside vs outside AABB.

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

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    root = Path('/fsx/tmp/marshawn_001660_hd')
    sys.path[:0] = [str(p) for p in [root/'code_native',
        root/'code_native/LookCloser/scripts', root/'code_sfm/LookCloser/scripts',
        root/'native_png_radialK4_COLD_R1_DEV3_STAGED/code',
        root/'two_actor_isolation/teacher_clip_tools_recipes']]
    import torch
    import nerfacc
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from probe_marshawn_occupancy import digest_model
    torch.set_num_threads(2)
    if sha(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Wrong immutable checkpoint')
    a.output.mkdir(parents=True, exist_ok=False)
    bounds = np.array(json.loads(a.box.read_text())['aabb'])

    def configure(c):
        c.eval_checkpoint = a.checkpoint
        c.load_checkpoint = c.load_dir = c.load_step = c.load_config = None
        return c

    cfg, pipeline, actual, step = eval_setup(a.config, test_mode='val',
        eval_num_rays_per_chunk=512, update_config_callback=configure)
    if actual.resolve() != a.checkpoint.resolve():
        raise ValueError('Loaded different checkpoint')
    pipeline.eval(); model = pipeline.model; model.current_train_step = step
    if model.sampler.training or model.sampler.get_sigma_fn(None, None) is not None:
        raise ValueError('Shared sample generation must not use opacity early termination')
    before = digest_model(model)
    box = torch.tensor(bounds, device=model.device, dtype=torch.float32)
    names = ['full', 'inside', 'outside', 'inside_contribution', 'outside_contribution']
    ds = pipeline.datamanager.eval_dataset
    ci = next(i for i,p in enumerate(ds.image_filenames) if p.stem.split('_')[-1] == a.camera)
    camera = copy.deepcopy(ds.cameras[ci:ci+1]).to(model.device)
    w,h = int(camera.width.item()), int(camera.height.item())
    rays = camera.generate_rays(0, keep_shape=True).flatten()
    buffers = {k: np.zeros((h*w,4),np.float32) for k in names}
    queries = {'black_near':[928,640], 'black_far':[1160,640],
               'grey_rear':[1090,648], 'grey_front':[1060,698]} if a.camera=='HC' else {}
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
            inside = ((xyz>=box[0]) & (xyz<=box[1])).all(-1)
            sigma = values[FieldHeadNames.DENSITY][...,0]
            rgb = values[FieldHeadNames.RGB]
            starts,ends = samples.frustums.starts[...,0],samples.frustums.ends[...,0]
            pack = nerfacc.pack_info(ids,len(rb))
            weights = {}
            for name, s in [('full',sigma),('inside',sigma*inside),('outside',sigma*(~inside))]:
                weights[name] = nerfacc.render_weight_from_density(
                    t_starts=starts,t_ends=ends,sigmas=s,packed_info=pack)[0]
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
        Image.fromarray(np.rint(np.clip(buffers[name][...,:3],0,1)*255).astype(np.uint8)).save(a.output/(a.camera+'_'+name+'.png'))
    np.savez_compressed(a.output/'raw_partition.npz',**buffers)
    split_error=float(np.max(np.abs(buffers['full']-buffers['inside_contribution']-buffers['outside_contribution'])))
    if split_error>2e-6 or digest_model(model)!=before or sha(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Partition sum or immutable state verification failed')
    source=Image.open(ds.image_filenames[ci]).convert('RGB')
    panels=[('SOURCE HC',source),('FULL MODEL',Image.open(a.output/(a.camera+'_full.png'))),
            ('INSIDE AABB ONLY',Image.open(a.output/(a.camera+'_inside.png'))),
            ('OUTSIDE: ORIGINAL VISIBLE CONTRIBUTION',Image.open(a.output/(a.camera+'_outside_contribution.png')))]
    font=ImageFont.truetype('DejaVuSans.ttf',18)
    for filename,crop in [('feet_contact.png',(870,565,1210,735)),('actors_contact.png',(735,205,1230,745))]:
        pw,ph=crop[2]-crop[0],crop[3]-crop[1]
        contact=Image.new('RGB',(pw*len(panels),ph+55),'#181818');draw=ImageDraw.Draw(contact)
        for i,(label,im) in enumerate(panels):
            contact.paste(im.crop(crop),(i*pw,55))
            draw.text((i*pw+4,6),label,font=font,fill='white')
        contact.save(a.output/filename)
    receipt=dict(kind='ACTUAL_FIELD_AABB_PARTITION_NO_TRAINING',camera=a.camera,
        aabb=bounds.tolist(),box_sha256=sha(a.box),checkpoint=str(actual),
        checkpoint_sha256=a.checkpoint_sha256,step=step,source_helper_sha256=sha(__file__),
        model_before_after_sha256=before,native_compositor_maxerror=max(parity),
        full_equals_inside_plus_outside_contributions_maxerror=split_error,
        samples=count,seconds=time.monotonic()-started,trace=trace,
        field_normalization_and_grid_unchanged=True,mask_or_source_compositing=False,updates=0,
        rule='Native sample midpoint XYZ inside exact AABB; zero other density then reintegrate. '
             'Contribution maps retain ORIGINAL full-scene transmittance and sum to full RGB/alpha.',
        boundary_discretization_world=float(model.config.render_step_size)/2,
        limitation='Historical full-scene model, not physical ground truth; cannot certify hidden anatomy or bad learned geometry.')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','box','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--camera',default='HC');p.add_argument('--chunk',type=int,default=512)
    main(p.parse_args())
