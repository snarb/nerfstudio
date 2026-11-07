"""Frozen p01/theta.1 SOURCE-grid vs semantic R0/R1 experiment; dry-run default."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def absolute(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('Actual import binding differs')
    return module


def validate(plan_path,execute):
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('kind')!='FROZEN_LC_P01_THETA01_GRID_R0_R1_V1' or plan['selector']!='hard_theta01' or plan['phases']!=['source_grid','semantic_r0','semantic_r1']:
        raise ValueError('Exact fixed physical field and three grid states required')
    if plan['source_checkpoint_sha256']!='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641' or plan['head_sha256']!='bdb2cc0d8138e4c22c61b4d411227b272b95ceac9e8295d474eecc4a84aa98bc':
        raise ValueError('Wrong frozen2UNKNOWN source/head')
    if plan['native_cap']!=1024 or plan['chunk']!=128 or plan['source_label_regime']!='two_reviewed_UNKNOWN_40470_40622' or plan['uniform_samples']!=[8192,16384]:
        raise ValueError('Unreviewed sampling/source-label change')
    if plan.get('gpu',0)!=0 or plan.get('gpu_uuid','GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46')!='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46':raise ValueError('Qualified source loader is bound to physical local GPU0')
    for path,digest in plan['bound_files'].items():
        if sha(path)!=digest:raise ValueError('Bound source/helper changed: '+path)
    if execute:
        release=json.loads(Path(plan['root_release']).read_text())
        if release.get('reviewer')!='root' or release.get('readonly_grid_comparison_authorized')is not True or release.get('plan_sha256')!=sha(plan_path):raise ValueError('Exact root read-only grid release required')
    else:release=None
    return plan,release


def run(plan_path,execute=False):
    plan,release=validate(plan_path,execute)
    if not execute:
        print(json.dumps(dict(CPU_binding_PASS=True,files=len(plan['bound_files']),GPU_started=False,training_authorized=False,phases=plan['phases'])));return
    helpers=Path(plan['helpers']);sys.path.insert(0,str(helpers))
    isolation=absolute('marshawn_gpu_isolation',helpers/'marshawn_gpu_isolation.py')
    lease=isolation.require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    import nerfacc
    from PIL import Image
    helper=absolute('marshawn_semantic_grid_replay',helpers/'marshawn_semantic_grid_replay.py')
    stable=absolute('marshawn_stable_packed_density_reference',helpers/'marshawn_stable_packed_density_reference.py')
    observer=absolute('marshawn_LC_semantic_physical_path',helpers/'marshawn_LC_semantic_physical_path.py')
    frozen=absolute('marshawn_LC_frozen_source',helpers/'marshawn_LC_frozen_source.py')
    head_module=absolute('marshawn_frozen_semantic_head',helpers/'marshawn_frozen_semantic_head.py')
    selectors=absolute('marshawn_frozen_semantic_selectors',helpers/'marshawn_frozen_semantic_selectors.py')
    # Candidate export API does not install a field hook.
    absolute('marshawn_semantic_node_cell_bridge',helpers/'marshawn_semantic_node_cell_bridge.py')
    candidates=absolute('export_marshawn_semantic_cell_candidates_cpu',helpers/'export_marshawn_semantic_cell_candidates_cpu.py')
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded frozen-grid comparison deadline')))
    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,lambda sig,_:(_ for _ in ()).throw(RuntimeError('Interrupted owned read-only worker '+str(sig))))
    signal.alarm(plan['max_seconds'])
    # The qualified P3 source loader requires a union of original and new
    # explicitly bound imports. No P3 script or source checkpoint is rewritten.
    loader_release=dict(release,bound_files=plan['bound_files'])
    source=frozen.load_source(plan['source_spec'],out,lease,loader_release)
    model=source['model'];pipe=source['pipeline'];manifest=source['manifest']
    late=absolute('_semantic_grid_replay_digest',Path(manifest['helpers'])/'train_marshawn_semantic_late_distortion.py')
    trained=torch.load(plan['head'],map_location='cpu',weights_only=False)
    if sha(plan['head'])!=plan['head_sha256'] or trained['model_digest']!=source['pipeline_digest'] or trained['iteration']!=2064 or trained['initial_probability']!=.01:
        raise ValueError('Wrong actual frozen physical head')
    head=head_module.FrozenActorGrid(manifest['bounds'],np.eye(4),resolution=128,initial_probability=.01).to(model.device)
    head.load_state_dict(trained['head'],strict=True);head_module.assert_exact_restoration(head,trained['head']);head.eval();head.requires_grad_(False)
    head_digest=head_module.tensor_digest(head)
    if head_digest!=trained['head_digest']:raise ValueError('Actual head tensor digest differs')
    loaded={}
    for radius in (0,1):
        arr,meta=candidates.load_candidate(plan['candidate'],expected_metadata_sha256=plan['candidate_metadata_sha256'],
            expected_head_sha256=plan['head_sha256'],expected_source_grid_sha256=plan['source_grid_sha256'],expected_selector=plan['selector'],radius=radius)
        loaded[radius]=torch.from_numpy(arr['active_r'+str(radius)].copy()).to(model.device)
    with np.load(Path(plan['candidate'])/'candidate.npz',allow_pickle=False) as z:
        original=torch.from_numpy(z['original_active'].copy()).to(model.device)
        envelope=torch.from_numpy(z['semantic_envelope'].copy()).to(model.device)
        boxes=torch.from_numpy(z['cascade_aabbs'].copy()).to(model.device)
    if not torch.equal(original,model.occupancy_grid.binaries) or not torch.equal(boxes.to(model.occupancy_grid.aabbs),model.occupancy_grid.aabbs):raise ValueError('Actual source grid bounds/binaries differ')
    phases=dict(source_grid=original,semantic_r0=loaded[0],semantic_r1=loaded[1])
    if [int(phases[k].sum()) for k in plan['phases']]!=[32046,23730,38319]:raise ValueError('Exact CPU candidate counts differ')
    imports={name:dict(path=str(Path(module.__file__).resolve()),sha256=sha(module.__file__)) for name,module in
        [('helper',helper),('stable',stable),('observer',observer),('source',frozen),('head',head_module),('selector',selectors),('candidate',candidates)]}
    for item in imports.values():
        if plan['bound_files'].get(item['path'])!=item['sha256']:raise ValueError('Imported helper not bound')
    write(out/'import_binding.json',imports)
    source['assert_unchanged']()
    def check_head():
        if head_module.tensor_digest(head)!=head_digest:raise ValueError('Frozen head changed')
    requests={name:json.loads(Path(path).read_text()) for name,path in plan['queries'].items()}
    expected_counts=dict(clothing103=103,shoe6=6,DC12=12,contour9=9)
    if set(requests)!=set(expected_counts) or any(len(requests[k]['pixels'])!=n or len(requests[k]['rays'])!=n for k,n in expected_counts.items()):raise ValueError('Exactly130 sealed physical rays required')
    witnesses=[p['witness_xyz'] for name in ('clothing103','shoe6') for p in requests[name]['pixels']]
    positions,position_proof=helper.adversarial_positions(head.bounds.cpu().numpy(),boxes.cpu().numpy(),witnesses)
    support_stats=dict(points=len(positions),selected_positive_rejected_envelope=0,head_positive_rejected_envelope=0,maximum_raw_sigma=0.,point_source=position_proof)
    with torch.inference_mode():
        for first in range(0,len(positions),8192):
            xyz=torch.from_numpy(positions[first:first+8192]).to(model.device);p=head(xyz);membership=selectors.selector_memberships(p)[plan['selector']]
            directions=torch.zeros_like(xyz);directions[:,2]=-1
            sigma,_=model.field.query_points(xyz,directions);sigma=sigma.reshape(-1)
            covered=helper.grid_membership(xyz,envelope,boxes)
            if first==0:
                support_stats['actual_witness_probability']=p[:109].cpu().tolist()
                support_stats['actual_witness_selected_sigma']=(sigma*membership)[:109].cpu().tolist()
                support_stats['actual_witness_cell_support']={name:helper.grid_membership(xyz[:109],g,boxes).cpu().tolist()for name,g in phases.items()}
            support_stats['head_positive_rejected_envelope']+=int(((membership>0)&~covered).sum())
            support_stats['selected_positive_rejected_envelope']+=int(((sigma*membership>0)&~covered).sum())
            support_stats['maximum_raw_sigma']=max(support_stats['maximum_raw_sigma'],float(sigma.max()))
        if support_stats['head_positive_rejected_envelope'] or support_stats['selected_positive_rejected_envelope']:raise ValueError('Actual CUDA selected support escapes CPU envelope')
    write(out/'actual_CUDA_support.json',support_stats);source['assert_unchanged']();check_head()
    # Full-frame field/head values are independent of EVAL labels. Metrics are
    # evaluated later in a separate CPU process after the physical GPU releases.
    dataset=pipe.datamanager.eval_dataset
    names=[Path(p).stem.split('_')[-1] for p in dataset.image_filenames]
    if set(names)!={'DC','HC','KA'} or len(names)!=3:raise ValueError('Exact EVAL3 required')
    dense_cache={};summaries={};reference_complete=json.loads(Path(plan['P3_reference_completion']).read_text())
    def compose(row):
        probability=head(row['xyz'])
        result=helper.packed_selected(row,probability,plan['selector'],selectors.selector_memberships,nerfacc,stable.stable_packed_density_weights)
        result['probability']=probability
        result['selected_positive_outside_semantic_envelope']=int(((result['selected_sigma']>0)&~helper.grid_membership(row['xyz'],envelope,boxes)).sum())
        if result['selected_positive_outside_semantic_envelope']:raise ValueError('Actual selected density outside certified node-cell support')
        result['selected_positive_outside_active_grid']=int(((result['selected_sigma']>0)&~helper.grid_membership(row['xyz'],model.occupancy_grid.binaries,boxes)).sum())
        return result
    def save_trace(path,row,selected):
        arrays={k:row[k].detach().cpu().numpy() for k in ('xyz','starts','ends','sigma','rgb','weight')}
        arrays['p']=selected['probability'].detach().cpu().numpy()
        arrays.update({k:selected[k].detach().cpu().numpy() for k in ('membership','selected_sigma','selected_weight','selected_RGBA')})
        np.savez_compressed(path,**arrays);return arrays
    def profile(arrays,pixel):
        weight=arrays['selected_weight'].astype(np.float64);mid=(arrays['starts'].astype(np.float64)+arrays['ends'])*.5
        result=dict(samples=len(mid),opacity=float(weight.sum()),RGBA=arrays['selected_RGBA'].reshape(-1).tolist(),
            first_start=float(arrays['starts'][0]) if len(mid) else None,last_end=float(arrays['ends'][-1]) if len(mid) else None,
            maximum_interval=float(np.max(arrays['ends']-arrays['starts'])) if len(mid) else None)
        if 'witness_xyz' in pixel:
            depth=float(np.dot(np.asarray(pixel['witness_xyz'])-pixel['_origin'],pixel['_direction']))
            result['mass_before_band_after_transmission']=[float(weight[mid<depth-.02].sum()),float(weight[np.abs(mid-depth)<=.02].sum()),float(weight[mid>depth+.02].sum()),max(0.,1-float(weight.sum()))]
        return result
    with torch.inference_mode():
        for phase,mask in phases.items():
            dest=out/phase;dest.mkdir();views=[];query_results=[]
            with helper.temporary_binaries(pipe,mask,late.tree_digest,source['assert_unchanged']) as assert_inside:
                for name in ['DC','HC','KA']:
                    camera=dataset.cameras[names.index(name):names.index(name)+1].to(model.device);rays=camera.generate_rays(0,keep_shape=True).flatten()
                    h,w=int(camera.height.item()),int(camera.width.item())
                    rgba=np.zeros((len(rays),4),np.float32);native_rgba=np.zeros_like(rgba);counts=np.zeros(len(rays),np.int32);last=np.zeros(len(rays),np.float32)
                    native_error=selected_error=0.;outside_active=0
                    for first in range(0,len(rays),plan['chunk']):
                        row=observer.capture_lc_path(model,rays[first:first+plan['chunk']],expected_cap=1024);selected=compose(row);n=len(row['num_samples_per_ray'])
                        rgba[first:first+n]=selected['selected_RGBA'].cpu().numpy();native_rgba[first:first+n]=torch.cat((row['native_rgb'],row['native_alpha'][:,None]),-1).cpu().numpy()
                        counts[first:first+n]=row['num_samples_per_ray'].cpu().numpy()
                        last[first:first+n]=row['sigma'].new_zeros(n).scatter_reduce_(0,row['ray_indices'].long(),row['ends'],reduce='amax',include_self=True).cpu().numpy()
                        native_error=max(native_error,max(row['proof'].values()));selected_error=max(selected_error,selected['independent_weight_error']);outside_active+=selected['selected_positive_outside_active_grid']
                        if first//plan['chunk']%128==0:write(out/'progress.json',dict(phase=phase,camera=name,pixels=first+n,total=len(rays),time=time.time(),optimizer_updates=0))
                    rgba=rgba.reshape(h,w,4);native_rgba=native_rgba.reshape(h,w,4)
                    parity=None
                    if phase=='source_grid':
                        reference_path=Path(plan['P3_reference'])/name/'physical_and_partition.npz'
                        refrow=next(x for x in reference_complete['records'] if x['camera']==name)
                        if sha(reference_path)!=refrow['physical_archive_sha256']:raise ValueError('P3 reference output changed')
                        with np.load(reference_path)as z:
                            parity=dict(selected=helper.numeric_rgba_parity(rgba,z[plan['selector']]),native=helper.numeric_rgba_parity(native_rgba,z['native_full']))
                        with np.load(Path(plan['P3_reference'])/name/'path_coverage.npz')as z:
                            if not np.array_equal(counts,z['sample_counts']) or float(np.max(abs(last-z['last_end'])))>2e-7:raise ValueError('Original P3 source sampling phase differs')
                    Image.fromarray(np.rint(np.clip(rgba[...,:3],0,1)*255).astype(np.uint8)).save(dest/(name+'.png'))
                    Image.fromarray(np.rint(np.clip(rgba[...,3],0,1)*65535).astype(np.uint16)).save(dest/(name+'_alpha.png'))
                    np.savez_compressed(dest/(name+'_physical.npz'),RGBA=rgba,native_RGBA=native_rgba,sample_counts=counts,last_end=last)
                    views.append(dict(camera=name,cap_hit_pixels=int((counts>=1024).sum()),sample_count_max=int(counts.max()),total_samples=int(counts.sum()),source_P3_parity=parity,
                        native_reference_error=native_error,selected_reference_error=selected_error,selected_positive_sample_centers_outside_active_grid=outside_active,archive_sha256=sha(dest/(name+'_physical.npz'))))
                    assert_inside();check_head()
                trace_root=dest/'traces';trace_root.mkdir()
                for label,request in requests.items():
                    ds=getattr(pipe.datamanager,request.get('split','eval')+'_dataset');order=[Path(p).stem.split('_')[-1]for p in ds.image_filenames]
                    if order!=request['camera_order']:raise ValueError('Sealed camera order differs')
                    for index,(pixel,expected_ray) in enumerate(zip(request['pixels'],request['rays'])):
                        camera=ds.cameras[pixel['camera']:pixel['camera']+1].to(model.device)
                        ray=camera.generate_rays(0,coords=torch.tensor([[pixel['y']+.5,pixel['x']+.5]],device=model.device),keep_shape=False)
                        if max(float(np.max(abs(ray.origins[0].cpu().numpy()-expected_ray['origin']))),float(np.max(abs(ray.directions[0].cpu().numpy()-expected_ray['direction']))))>1e-6:raise ValueError('Physical query chain changed')
                        ray=model.collider(ray);near,far=float(ray.nears.item()),float(ray.fars.item())
                        if not np.isfinite([near,far]).all() or far<=near:raise ValueError('Invalid bounded query interval')
                        key=label+'_'+str(index);px=dict(pixel,_origin=ray.origins[0].cpu().numpy(),_direction=ray.directions[0].cpu().numpy())
                        if key not in dense_cache:
                            dense_cache[key]={};dense_dir=out/'dense';dense_dir.mkdir(exist_ok=True)
                            for count in plan['uniform_samples']:
                                edges=torch.linspace(near,far,count+1,device=model.device);starts,ends=edges[:-1],edges[1:];xyz=ray.origins+(starts+ends)[:,None]*.5*ray.directions
                                sigma,rgb=model.field.query_points(xyz,ray.directions.expand(count,-1),camera_indices=None if ray.camera_indices is None else ray.camera_indices.expand(count,-1))
                                ids=torch.zeros(count,dtype=torch.long,device=model.device)
                                row=dict(xyz=xyz,starts=starts,ends=ends,sigma=sigma.reshape(-1),rgb=rgb,ray_indices=ids,num_samples_per_ray=torch.tensor([count],device=model.device),weight=torch.zeros(count,device=model.device))
                                row['weight']=nerfacc.render_weight_from_density(t_starts=starts,t_ends=ends,sigmas=row['sigma'],packed_info=nerfacc.pack_info(ids,1))[0]
                                selected=compose(row);arrays=save_trace(dense_dir/(key+'_'+str(count)+'.npz'),row,selected)
                                masks={name:helper.grid_membership(xyz,g,boxes).cpu().numpy()for name,g in phases.items()}
                                envelope_inside=helper.grid_membership(xyz,envelope,boxes)
                                if bool(((selected['selected_sigma']>0)&~envelope_inside).any()):raise ValueError('Dense selected density escapes conservative semantic envelope')
                                dense_cache[key][count]=dict(arrays=arrays,masks=masks,summary=profile(arrays,px))
                        row=observer.capture_lc_path(model,ray,expected_cap=1024);selected=compose(row);arrays=save_trace(trace_root/(key+'.npz'),row,selected)
                        dense=dense_cache[key];lower,higher=[dense[n]for n in plan['uniform_samples']]
                        result=dict(group=label,index=index,pixel=pixel,origin=px['_origin'].tolist(),direction=px['_direction'].tolist(),near=near,far=far,
                            native=profile(arrays,px),cap_hit=bool(row['cap_hit'].item()),selected_positive_sample_centers_outside_active_grid=selected['selected_positive_outside_active_grid'],native_trace=str(trace_root/(key+'.npz')),native_trace_sha256=sha(trace_root/(key+'.npz')),
                            dense_reference={str(n):v['summary']for n,v in dense.items()},
                            missed_selected_mass=helper.missed_reference_mass(higher['arrays'],arrays,higher['masks']),
                            convergence=dict(opacity_abs_delta=abs(lower['summary']['opacity']-higher['summary']['opacity']),
                                RGB_max_abs_delta=float(np.max(abs(np.array(lower['summary']['RGBA'][:3])-higher['summary']['RGBA'][:3])))))
                        query_results.append(result)
                        write(out/'progress.json',dict(phase=phase,query_group=label,completed=len(query_results),total=130,time=time.time(),optimizer_updates=0))
                assert_inside();check_head()
            # Context restored source binaries before its strict full-state proof.
            source['assert_unchanged']();summaries[phase]=dict(views=views,queries=query_results,source_restored=True,head_unchanged=True,
                binary_cells=int(mask.sum()),only_binaries_temporarily_overridden=True)
            write(dest/'complete.json',summaries[phase])
    checks=source['assert_unchanged']();check_head();source['guard_done'].set();signal.alarm(0)
    write(out/'trace_inventory.json',{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*.npz'))})
    validate(plan_path,True)
    write(out/'complete.json',dict(passed=True,phases=list(phases),source_unchanged=checks,head_unchanged=True,
        head_sha256=plan['head_sha256'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],physical_selector=plan['selector'],
        head_label_regime=plan['source_label_regime'],source_restored_after_each_phase=True,optimizer_updates=0,occupancy_updates=0,frequency_updates=0,
        semantic_envelope_sampled_CUDA_support=support_stats,uniform_references_reused_only_after_field_immutability_proof=True,
        full_visibility_or_physical_free_space_certified=False,quality_promoted=False,mandatory_manual_visual_gate=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();run(a.plan,a.execute)
