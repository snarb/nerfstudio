"""Teacher geometry SANITY: ray-interval clip only, original field/grid retained."""
import argparse,ast,copy,hashlib,json,time,types
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,PngImagePlugin
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def phase_functions(source):
    """Reuse exact reviewed slab/tests without importing its R0 path bootstrap."""
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['slab','unit_tests']]
    if len(nodes)!=2:raise ValueError('Expected reviewed phase slab and unit tests')
    namespace={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
    return namespace['slab'],namespace['unit_tests']

def png(path,rgb,alpha=None):
    info=PngImagePlugin.PngInfo();info.add(b'sRGB',b'\0')
    info.add_text('Encoding','Native encoded-PNG teacher output; NO extra tone/PQ/exposure')
    info.add_text('Alpha','Straight RGB=Cpremult/alpha; raw premult/alpha retained in NPZ'if alpha is not None else'Premultiplied volume RGB displayed over physicalblack0')
    q=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
    if alpha is not None:q=np.dstack([q,np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)])
    Image.fromarray(q).save(path,pnginfo=info)

def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(1.,16*1024**3/torch.cuda.get_device_properties(0).total_memory))
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Wrong immutable teacher')
    boxpayload=json.loads(a.boxes.read_text());box=boxpayload['scopes']['subject']['loose']
    review=json.loads(a.geometry_review.read_text());manual=json.loads(a.manual_review.read_text())
    assert review['provisional_candidates']['loose']==box and manual['all66_actual_views_inspected']
    assert manual['loose_box_obvious_limb_clip_not_observed']and manual['AABB_projection_not_actor_mask']
    slab,tests=phase_functions(a.phase_source);testresult=tests()
    from nerfstudio.utils.eval_utils import eval_setup
    from probe_marshawn_occupancy import digest_model
    from marshawn_release_common import native_bounds,trace_native
    from nerfstudio.model_components import renderers
    def configure(c):
        c.eval_checkpoint=a.checkpoint;c.load_checkpoint=c.load_dir=c.load_step=c.load_config=None
        return c
    cfg,pipe,actual,step=eval_setup(a.config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure)
    pipe.eval();m=pipe.model;m.current_train_step=step
    assert actual.resolve()==a.checkpoint.resolve()and step==20000 and cfg.pipeline.final_grid_radius==1
    assert m.config.cone_angle==0 and m.config.render_step_size==.003 and m.config.near_plane==.052
    assert renderers.BACKGROUND_COLOR_OVERRIDE is None
    background=m.renderer_rgb.background_color
    if isinstance(background,str):
        if background in ['random','black']:bg=np.zeros(3,dtype=np.float32)
        elif background=='white':bg=np.ones(3,dtype=np.float32)
        else:raise ValueError('Cannot unpremultiply unknown native background '+background)
    elif torch.is_tensor(background):bg=background.detach().cpu().float().numpy().reshape(3)
    else:raise ValueError('Unreviewed background type')
    gridrecord=json.loads(a.checkpoint.with_name('step-000020000_grid.json').read_text())
    gridarchive=a.checkpoint.with_name('step-000020000_grid.npz');archive=np.load(gridarchive)
    assert gridrecord['radius']==1 and gridrecord['checkpoint_sha256']==a.checkpoint_sha256
    assert sha(gridarchive)==gridrecord['grid_archive_sha256']
    assert np.array_equal(m.occupancy_grid.binaries.cpu().numpy(),archive['active'])
    assert np.array_equal(m.occupancy_grid.occs.cpu().numpy(),archive['ema'])
    a.output.mkdir(exist_ok=False);bounds=native_bounds(m);before=digest_model(m)
    grid={k:v.detach().cpu().clone()for k,v in m.occupancy_grid.state_dict().items()};fieldbox=m.field.aabb.clone()
    original=m.get_outputs;had_override='get_outputs'in m.__dict__;old_override=m.__dict__.get('get_outputs');active=[True];counts={'rays':0,'misses':0}
    def clipped(self,ray_bundle):
        if active[0]:
            lo,hi,ok=slab(ray_bundle.origins,ray_bundle.directions,box)
            if ray_bundle.nears is not None:lo=torch.maximum(lo,ray_bundle.nears.reshape(-1));hi=torch.maximum(lo,hi)
            if ray_bundle.fars is not None:hi=torch.minimum(hi,ray_bundle.fars.reshape(-1));lo=torch.minimum(lo,hi)
            ray_bundle.nears=lo[:,None];ray_bundle.fars=hi[:,None]
            counts['rays']+=len(lo);counts['misses']+=int((~ok).sum())
        return original(ray_bundle)
    m.get_outputs=types.MethodType(clipped,m);rows=[];phase=[]
    try:
        ds=pipe.datamanager.eval_dataset
        for i,path in enumerate(ds.image_filenames):
            name=path.stem.split('_')[-1];camera=copy.deepcopy(ds.cameras[i:i+1]).to(m.device)
            assert(int(camera.width),int(camera.height))==(1920,1012)
            started=time.monotonic()
            with torch.inference_mode():out=m.get_outputs_for_camera(camera)
            raw=out['rgb'].detach().float().cpu().numpy();alpha=out['accumulation'].detach().float().cpu().numpy().reshape(1012,1920)
            assert np.isfinite(raw).all()and np.isfinite(alpha).all()and alpha.min()>=0 and alpha.max()<1.0001
            premult=raw-bg*(1-alpha[...,None])
            if premult.min() < -2e-6:raise ValueError('Known background subtraction is inconsistent')
            straight=np.divide(premult,alpha[...,None],out=np.zeros_like(premult),where=alpha[...,None]>0)
            png(a.output/f'{name}_premult_black.png',premult);png(a.output/f'{name}_straight_RGBA.png',straight,alpha)
            png(a.output/f'{name}_alpha.png',np.repeat(alpha[...,None],3,axis=-1))
            np.savez_compressed(a.output/f'{name}_raw.npz',raw_model_rgb=raw,premultiplied_rgb=premult,alpha=alpha,
                straight_rgb=straight,depth=out['depth'].detach().float().cpu().numpy(),
                samples=out['num_samples_per_ray'].detach().cpu().numpy())
            # Project the world-box centre, then test three nearby native rays.
            centre=torch.tensor(np.mean(box,axis=0),device=m.device,dtype=torch.float32);c2w=camera.camera_to_worlds[0];local=c2w[:3,:3].T@(centre-c2w[:3,3])
            u=float(camera.fx)*float(local[0]/-local[2])+float(camera.cx);v=-float(camera.fy)*float(local[1]/-local[2])+float(camera.cy)
            for dx,dy in [(0,0),(-24,0),(0,24)]:
                x=int(np.clip(round(u)+dx,0,1919));y=int(np.clip(round(v)+dy,0,1011));coords=torch.tensor([[y+.5,x+.5]],device=m.device)
                active[0]=False;baseline=trace_native(m,camera.generate_rays(0,coords=coords,keep_shape=False))
                active[0]=True;test=trace_native(m,camera.generate_rays(0,coords=coords,keep_shape=False))
                bs,ts=baseline['starts'],test['starts'];inside=(ts>=bs.min()-1e-5)&(ts<=bs.max()+1e-5)if len(bs)and len(ts)else np.zeros(len(ts),bool)
                err=np.min(abs(ts[:,None]-bs[None,:]),axis=1)[inside]if len(bs)and len(ts)else np.array([])
                if len(err)and err.max()>2e-5:raise ValueError('Raymarch phase changed beyond floating point tolerance')
                phase.append(dict(camera=name,pixel_xy=[x,y],baseline_samples=len(bs),clipped_samples=len(ts),
                    shared_termination_range_nearest_start_maxerror=float(err.max())if len(err)else None,
                    clipped_global_lattice_maxerror=float(np.max(abs((ts-.052)/.003-np.rint((ts-.052)/.003)))*.003)if len(ts)else None))
            rows.append(dict(camera=name,seconds=time.monotonic()-started,mean_alpha=float(alpha.mean()),
                alpha_nonzero_pixels=int((alpha>0).sum()),alpha_over_half_pixels=int((alpha>.5).sum()),
                exported_straight_clip_channels=int(((straight<0)|(straight>1)).sum()),source_background=str(background),known_background_rgb=bg.tolist()))
            (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
    finally:
        if had_override:m.get_outputs=old_override
        else:delattr(m,'get_outputs')
    after=digest_model(m);assert after==before and torch.equal(fieldbox,m.field.aabb)
    assert all(torch.equal(v.detach().cpu(),grid[k])for k,v in m.occupancy_grid.state_dict().items())
    assert sha(a.checkpoint)==a.checkpoint_sha256
    report=dict(kind='TWO_ACTOR_R1_20K_LOOSE_RAYCLIP_TEACHER_SANITY_ONLY',checkpoint=str(a.checkpoint),checkpoint_sha256=a.checkpoint_sha256,
        config_sha256=sha(a.config),source_sha256=sha(__file__),phase_source=str(a.phase_source),phase_source_sha256=sha(a.phase_source),
        bounds_source_sha256=sha(a.boxes),geometry_review_sha256=sha(a.geometry_review),manual_review_sha256=sha(a.manual_review),
        clip_AABB_subject_loose_exact=box,original_field_grid_bounds=bounds,model_state_before=before,model_state_after=after,
        field_normalization_NN_EMA_original_active_grid_and_checkpoint_unchanged=True,stored_radius1_not_redilated=True,
        phase_policy='.052+n*.003; entryceil/exitfloor reused exact reviewed slab',phase_unit_tests=testresult,phase_checks=phase,
        rgba_contract='Raw native encoded-PNG premultiplied RGB + alpha; subtract known forwardBG then straight=C/alpha for PNG only; no extra transfer/tone/exposure',
        ray_counts=counts,rows=rows,metrics_computed=False,GT_RGB_or_masks_used_for_compositing=False,optimizer_objects=0,updates=0,
        limitations=['Full-scene teacher training history retained; not actor-only training or semantic isolation',
            'AABB can retain floor/phantoms inside box and cannot resolve actor instance occlusion',
            'Conservative phase-aligned whole-step clipping can remove up to.003worlddistance at box boundary',
            'PNG RGBA quantizes alpha; raw NPZ premultRGB+alpha are authoritative'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(output=str(a.output),state_unchanged=True,rows=rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','boxes','geometry-review','manual-review','phase-source','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);main(p.parse_args())
