"""Reviewed direct-LC density selectors, or a separate uncapped-path qualification.

Every selector recomputes transmittance from actual field sigma. No EVAL image
or semantic label affects a rendered pixel. Native1024 is explicitly a capped
path diagnostic; uncapped0 changes allocation and needs its own identity gate.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from prepare_marshawn_frozen_semantic_cpu import sha, write


def authorize(path, spec):
    release = json.loads(Path(spec['root_release']).read_text())
    key = 'LC_uncapped_qualification_authorized' if spec['phase'] == 'uncapped_qualification' else 'LC_physical_render_authorized'
    if release.get('reviewer') != 'root' or release.get(key) is not True or release['spec_sha256'].get(str(path)) != sha(path):
        raise ValueError('Exact separate root physical-path release required')
    for p, digest in release['bound_files'].items():
        if sha(p) != digest: raise ValueError('Bound input changed: '+p)
    return release


def capture_path(model, rays, mode):
    from marshawn_LC_semantic_physical_path import capture_lc_path
    expected_cap = {'native1024':1024,'uncapped0':0}[mode]
    if model.config.max_steps_per_ray != 1024: raise ValueError('Native config must begin and end at1024')
    try:
        model.config.max_steps_per_ray = expected_cap
        return capture_lc_path(model,rays,expected_cap=expected_cap)
    finally:
        model.config.max_steps_per_ray = 1024


def run(path, execute):
    spec = json.loads(path.read_text())
    qualification = spec['phase'] == 'uncapped_qualification'
    if spec['path_mode'] not in ['native1024','uncapped0'] or (qualification and spec['path_mode'] != 'uncapped0'):
        raise ValueError('Explicit separately reviewed traversal required')
    if not execute:
        print(json.dumps(dict(phase=spec['phase'],path_mode=spec['path_mode'],cameras=['DC','HC','KA'],
            source_model_updates=0,CUDA_started=False,optimizer_created=False,heldout_fit_labels=False,
            exact_contributor_trace_pixels={'HC':[[875,602]],'DC':[[1097,572]],'KA':[[1075,270],[961,485]]},
            full_visibility_certified=False)))
        return
    release=authorize(path,spec)
    from marshawn_gpu_isolation import require_controller_lease
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    import nerfacc
    from PIL import Image
    from marshawn_LC_frozen_source import load_source
    from marshawn_frozen_semantic_head import FrozenActorGrid,tensor_digest,assert_exact_restoration
    from marshawn_frozen_semantic_selectors import selector_memberships
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Reviewed LC render time exceeded')))
    signal.alarm(spec['max_seconds'])
    source=load_source(spec,out,lease,release);model=source['model'];m=source['manifest']
    from nerfstudio.model_components import renderers
    if renderers.BACKGROUND_COLOR_OVERRIDE is not None or model.renderer_rgb.background_color != 'black':
        raise ValueError('Actual native black background with no global override required')
    head=None;head_digest=None;completion=None;initial_probability=None;initial_nodes=None
    if not qualification:
        completion=json.loads(Path(spec['head_completion']).read_text());hp=Path(completion['checkpoint'])
        if sha(hp) != completion['checkpoint_sha256']:raise ValueError('Bound semantic head changed')
        trained=torch.load(hp,map_location='cpu',weights_only=False)
        if (trained['source_kind'] != 'DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or trained['model_digest'] != source['pipeline_digest'] or
                trained['iteration'] != 2064 or trained['initial_probability'] != spec['expected_initial_probability'] or
                trained['LC_capture_receipt_sha256'] != spec['LC_capture_receipt_sha256'] or
                trained['label_qualification_sha256'] != sha(spec['label_qualification']) or
                trained['spec_sha256'] != spec['head_training_spec_sha256'] or not trained['world_to_box_is_identity']):
            raise ValueError('Head must be separately fitted on this exact LC visibility and prior')
        initial_probability=trained['initial_probability']
        head=FrozenActorGrid(m['bounds'],np.eye(4),resolution=128,initial_probability=initial_probability).cuda()
        head.load_state_dict(trained['head'],strict=True);assert_exact_restoration(head,trained['head'])
        head.eval();head.requires_grad_(False);head_digest=tensor_digest(head)
        if not torch.equal(head.world_to_box,torch.eye(4,device=model.device)):raise ValueError('Floor transform must be identity')
        initial_logit=torch.tensor(np.log(initial_probability/(1-initial_probability)),dtype=head.logits.dtype,device=model.device)
        initial_nodes=int((head.logits==initial_logit).sum())
    ds=source['pipeline'].datamanager.eval_dataset
    cameras=[Path(p).stem.split('_')[-1]for p in ds.image_filenames]
    if set(cameras) != {'DC','HC','KA'} or len(cameras) != 3:raise ValueError('Exact three heldout model cameras required')
    bounds=torch.tensor(m['bounds'],device=model.device)
    selectors=['identity_inside'] if qualification else ['identity_inside','soft_probability','hard_theta01','hard_theta05','exactzero_smoothstep01_05']
    traces={'HC':[[875,602]],'DC':[[1097,572]],'KA':[[1075,270],[961,485]]}
    summaries=[]
    with torch.inference_mode():
        for name in ['DC','HC','KA']:
            camera=ds.cameras[cameras.index(name):cameras.index(name)+1].to(model.device)
            rays=camera.generate_rays(0,keep_shape=True).flatten()
            height,width=int(camera.height.item()),int(camera.width.item())
            variants=['native_full',*selectors]+(['native1024_reference'] if qualification else ['native_weight_semantic_partition'])
            buffers={k:np.zeros((len(rays),4),np.float32)for k in variants}
            newly_visible={k:np.zeros(len(rays),np.float32)for k in selectors}
            counts=np.zeros(len(rays),np.int32);native_counts=np.zeros_like(counts)
            last_ends=np.zeros(len(rays),np.float32);native_last_ends=np.zeros_like(last_ends)
            hidden_count=0;hidden_p_sum=0.;samples=0;parity=0.;maximum_sigma=0.;trace_records=[]
            selected_sigma_max={k:0. for k in selectors};hidden_selected_tau={k:0. for k in selectors}
            for first in range(0,len(rays),spec['chunk']):
                rb=rays[first:first+spec['chunk']]
                native=capture_path(model,rb,'native1024') if qualification else None
                row=capture_path(model,rb,spec['path_mode'])
                ids=row['ray_indices'].long();n=len(rb);sigma=row['sigma'];rgb=row['rgb'];w=row['weight']
                if not torch.isfinite(sigma).all():raise ValueError('Nonfinite sigma')
                inside=((row['xyz']>=bounds[0])&(row['xyz']<=bounds[1])).all(-1)
                p=head(row['xyz']) if head is not None else inside.to(sigma.dtype)
                memberships=dict(identity_inside=inside.to(sigma.dtype))
                if head is not None:memberships.update(selector_memberships(p))
                info=nerfacc.pack_info(ids,n)
                buffers['native_full'][first:first+n]=torch.cat((row['native_rgb'],row['native_alpha'][:,None]),-1).cpu().numpy()
                if qualification:
                    buffers['native1024_reference'][first:first+n]=torch.cat((native['native_rgb'],native['native_alpha'][:,None]),-1).cpu().numpy()
                    native_counts[first:first+n]=native['num_samples_per_ray'].cpu().numpy()
                    tail=sigma.new_zeros(n).scatter_reduce_(0,native['ray_indices'].long(),native['ends'],reduce='amax',include_self=True)
                    native_last_ends[first:first+n]=tail.cpu().numpy()
                else:
                    partition=w*p
                    comp=nerfacc.accumulate_along_rays(partition,values=rgb,ray_indices=ids,n_rays=n)
                    alpha=nerfacc.accumulate_along_rays(partition,values=None,ray_indices=ids,n_rays=n)
                    buffers['native_weight_semantic_partition'][first:first+n]=torch.cat((comp,alpha),-1).cpu().numpy()
                counts[first:first+n]=row['num_samples_per_ray'].cpu().numpy()
                last_ends[first:first+n]=sigma.new_zeros(n).scatter_reduce_(0,ids,row['ends'],reduce='amax',include_self=True).cpu().numpy()
                zero=(w==0)&inside
                hidden_count+=int(zero.sum());hidden_p_sum+=float(p[zero].sum())
                selected={}
                for key,membership in memberships.items():
                    weights=nerfacc.render_weight_from_density(t_starts=row['starts'],t_ends=row['ends'],sigmas=sigma*membership,packed_info=info)[0]
                    selected_sigma_max[key]=max(selected_sigma_max[key],float((sigma*membership).max()) if len(sigma) else 0.)
                    hidden_selected_tau[key]+=float((sigma*membership*(row['ends']-row['starts']))[zero].sum())
                    comp=nerfacc.accumulate_along_rays(weights,values=rgb,ray_indices=ids,n_rays=n)
                    alpha=nerfacc.accumulate_along_rays(weights,values=None,ray_indices=ids,n_rays=n)
                    buffers[key][first:first+n]=torch.cat((comp,alpha),-1).cpu().numpy()
                    newly_visible[key][first:first+n]=nerfacc.accumulate_along_rays(weights*zero,values=None,ray_indices=ids,n_rays=n).reshape(-1).cpu().numpy()
                    selected[key]=weights
                for x,y in traces[name]:
                    index=y*width+x
                    if not first<=index<first+n:continue
                    choose=ids==(index-first);destination=out/(name+'_trace_%d_%d.npz'%(x,y))
                    arrays={k:row[k][choose].cpu().numpy()for k in ['xyz','starts','ends','sigma','rgb','weight']}
                    arrays['p']=p[choose].cpu().numpy()
                    for key in selectors:
                        arrays[key+'_membership']=memberships[key][choose].cpu().numpy()
                        arrays[key+'_selected_sigma']=(sigma*memberships[key])[choose].cpu().numpy()
                        arrays[key+'_selected_weight']=selected[key][choose].cpu().numpy()
                    np.savez_compressed(destination,**arrays)
                    rank_key='identity_inside' if qualification else 'hard_theta05'
                    top=np.argsort(arrays[rank_key+'_selected_weight'])[-20:][::-1]
                    trace_records.append(dict(pixel_xy=[x,y],archive=str(destination),archive_sha256=sha(destination),
                        same_physical_render_batch=True,top20_selected_sample_indices=top.tolist(),
                        source_checkpoint_sha256=m['source_archive_sha256']['checkpoint.ckpt'],source_pipeline_digest=source['pipeline_digest'],
                        selected_head_sha256=None if qualification else completion['checkpoint_sha256'],
                        path_mode=spec['path_mode'],native_RGBA=buffers['native_full'][index].tolist(),
                        selected_RGBA={k:buffers[k][index].tolist()for k in selectors},
                        interpretation='Exact model sample/contribution trace. Source photo is not alphaGT; no EVAL labels or ray-empty claims.'))
                samples+=len(ids);parity=max(parity,max(row['proof'].values()))
                maximum_sigma=max(maximum_sigma,float(sigma.max()) if len(sigma) else 0.)
                if first//spec['chunk']%128==0:
                    write(out/'progress.json',dict(camera=name,rendered_pixels=first+n,total_pixels=len(rays),
                        path_mode=spec['path_mode'],native_updates=0,health=source['source_runner'].health(torch,lease['expected_physical_uuid'],m)))
            view=out/name;view.mkdir()
            buffers={k:v.reshape(height,width,4)for k,v in buffers.items()}
            for key,rgba in buffers.items():
                Image.fromarray(np.rint(np.clip(rgba[...,:3],0,1)*255).astype(np.uint8)).save(view/(key+'.png'))
                Image.fromarray(np.rint(np.clip(rgba[...,3],0,1)*65535).astype(np.uint16)).save(view/(key+'_alpha16.png'))
            np.savez_compressed(view/'physical_and_partition.npz',**buffers)
            np.savez_compressed(view/'path_coverage.npz',sample_counts=counts,last_end=last_ends,native1024_counts=native_counts,native1024_last_end=native_last_ends)
            np.savez_compressed(view/'newly_visible_native_zero_weight_mass.npz',**{k:v.reshape(height,width)for k,v in newly_visible.items()})
            native_difference=None
            if qualification:
                delta=np.abs(buffers['native_full']-buffers['native1024_reference'])
                native_difference=dict(max_RGB=float(delta[...,:3].max()),mean_RGB=float(delta[...,:3].mean()),
                    max_alpha=float(delta[...,3].max()),pixels_RGB_gt2e5=int((delta[...,:3].max(-1)>2e-5).sum()),
                    strict_float_identity_pass=bool(delta.max()<=2e-5),
                    uncapped_reaches_beyond_native_last_end_pixels=int((last_ends>native_last_ends+1e-5).sum()),
                    native_cap_hit_pixels=int((native_counts>=1024).sum()),
                    note='Uncapping changes allocation and step phase. Greater last end proves some extra tail reach, not exact discarded interval count or complete geometry.')
            checks=source['assert_unchanged']()
            summaries.append(dict(camera=name,samples=samples,native_parity_max=parity,max_sigma=maximum_sigma,
                native_zero_weight_inside_samples=hidden_count,native_zero_weight_mean_p=hidden_p_sum/max(1,hidden_count),
                selected_sigma_max=selected_sigma_max,native_zero_weight_retained_tau=hidden_selected_tau,
                native_cap_hit_pixels=int((counts>=1024).sum()) if spec['path_mode']=='native1024' else 0,
                native1024_vs_uncapped=native_difference,source_unchanged=checks,exact_traces=trace_records,
                newly_visible_mass_pixels_gt001={k:int((v>.01).sum())for k,v in newly_visible.items()},
                physical_archive_sha256=sha(view/'physical_and_partition.npz')))
    if head is not None and tensor_digest(head)!=head_digest:raise ValueError('Head changed during physical rendering')
    checks=source['assert_unchanged']();signal.alarm(0);source['guard_done'].set()
    write(out/'complete.json',dict(records=summaries,path_mode=spec['path_mode'],phase=spec['phase'],source_unchanged=checks,
        native_and_head_optimizers_created=False,source_RGB_or_masks_for_fit=False,head_initial_probability=initial_probability,
        source_pipeline_digest=source['pipeline_digest'],source_checkpoint_sha256=m['source_archive_sha256']['checkpoint.ckpt'],
        frozen_head_digest=head_digest,world_to_box_is_identity=True,sigma_callback_is_None=True,
        nodes_exactly_at_explicit_initial_logit=initial_nodes,grid_node_count=None if head is None else head.logits.numel(),
        initial_node_caveat='Exact initial values reflect the declared prior, not a perfect no-evidence certificate; shared interpolation/cancellations can preserve or restore values.',
        original_density_opacity_earlystop_disabled=True,selectors=selectors,actual_transmittance_recomputed=True,
        semantic_partition_is_nonphysical_diagnostic=True,no_hidden_support_gate=True,
        strict_uncapped_source_identity_pass=all(r['native1024_vs_uncapped']['strict_float_identity_pass']for r in summaries) if qualification else None,
        full_visibility_certified=False,geometry_limitation='Unchanged original occupancy/admission can omit geometry. Native1024 also coarsens/tail-clips; uncapped0 changes allocation/phase and requires its own identity review.',
        mandatory_manual_visual_gate=True,automatically_promoted_selector=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.spec,args.execute)
