"""Read-only native LC contribution capture:21 TRAIN rays and7 point densities."""
import argparse
import json
import os
from pathlib import Path
import signal
from prepare_marshawn_frozen_semantic_cpu import sha, write


def run(path, execute):
    spec=json.loads(path.read_text())
    if not execute:
        print(json.dumps(dict(TRAIN_rays=21,point_density_queries=7,source_updates=0,
            optimizer_created=False,head_loaded=False,selectors_rendered=False,
            original_cache_mutated=False,all_query_fit_weights=0,CUDA_started=False)))
        return
    release=json.loads(Path(spec['root_release']).read_text())
    if release.get('reviewer')!='root' or release.get('LC_standalone21_capture_authorized') is not True or release['spec_sha256'].get(str(path))!=sha(path):
        raise ValueError('Exact independent root21+7 capture release required')
    for file,digest in release['bound_files'].items():
        if sha(file)!=digest:raise ValueError('Bound input changed: '+file)
    from marshawn_gpu_isolation import require_controller_lease
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Standalone capture deadline')))
    signal.alarm(spec['max_seconds'])
    from marshawn_LC_frozen_source import load_source
    source=load_source(spec,out,lease,release)
    import numpy as np
    import torch
    from marshawn_LC_semantic_physical_path import capture_lc_path
    model=source['model'];pipeline=source['pipeline'];m=source['manifest']
    manifest=json.loads(Path(spec['query_manifest']).read_text())
    if source['pipeline_digest']!=manifest['source_pipeline_digest'] or m['world_to_box']!=np.eye(4).tolist():
        raise ValueError('Exact source pipeline and already-floor identity required')
    ds=pipeline.datamanager.train_dataset
    names=[Path(f).stem.split('_')[-1]for f in ds.image_filenames]
    if set(names)!=set(m['train_cameras']) or len(names)!=63:raise ValueError('TRAIN camera membership differs')
    frames={f['physical_camera']:f for f in json.loads(Path(spec['transforms']).read_text())['frames']}
    records=[];total_rays=0;total_samples=0
    bounds=torch.tensor(m['bounds'],device=model.device)
    with torch.inference_mode():
        for record in manifest['records']:
            cam=record['camera'];file=Path(record['archive'])
            if sha(file)!=record['archive_sha256']:raise ValueError('Standalone query bytes changed')
            with np.load(file,allow_pickle=False)as z:q={k:z[k].copy()for k in z.files}
            if np.any(q['label_weight']!=0) or not np.array_equal(q['world_to_box'],np.eye(4)):
                raise ValueError('All new fit weights remain0 and floor transform identity')
            camera=ds.cameras[names.index(cam):names.index(cam)+1].to(model.device)
            expected_pose=torch.tensor(frames[cam]['transform_matrix'],dtype=torch.float32)[:3]
            if not torch.equal(camera.camera_to_worlds[0].cpu(),expected_pose):
                raise ValueError('Camera pose is not unchanged floor pose')
            coords=torch.tensor(q['coords_yx_COLMAP'],dtype=torch.float32,device=model.device)
            rays=camera.generate_rays(0,coords=coords,keep_shape=False)
            errors={}
            for key,actual,tolerance in [('origins',rays.origins,2e-7),('directions',rays.directions,5e-7),
                    ('pixel_area',rays.pixel_area,1e-10),('directions_norm',rays.metadata['directions_norm'],5e-7)]:
                error=float((actual.cpu()-torch.from_numpy(q[key])).abs().max())
                errors[key]=error
                if error>tolerance:raise ValueError('Native floor ray/footprint proof differs: '+key)
            # Exact unchanged legacy ARM1024. One deterministic camera packet is
            # declared per archive. Recaptured old rays can have a different
            # allocator context from their original128-ray cache packets.
            row=capture_lc_path(model,rays,expected_cap=1024)
            n=len(rays);ids=row['ray_indices'].long()
            inside=((row['xyz']>=bounds[0])&(row['xyz']<=bounds[1])).all(-1)
            capacity=row['weight'].new_zeros(n).scatter_add_(0,ids,row['weight']*inside)
            arrays={k:row[k].cpu().numpy()for k in ['xyz','starts','ends','sigma','rgb','weight',
                'spacing_starts','spacing_ends','ray_indices','native_rgb','native_alpha','num_samples_per_ray','cap_hit']}
            arrays.update(q)
            arrays.update(actual_origins=rays.origins.cpu().numpy(),actual_directions=rays.directions.cpu().numpy(),
                actual_pixel_area=rays.pixel_area.cpu().numpy(),actual_directions_norm=rays.metadata['directions_norm'].cpu().numpy(),
                accepted_box_actor_cap=capacity.cpu().numpy(),model_digest=np.array(source['pipeline_digest']))
            destination=out/(cam+'.npz');np.savez_compressed(destination,**arrays)
            queries=[]
            for i,query_id in enumerate(q['query_id']):
                selected=ids==i;w=row['weight'][selected];starts=row['starts'][selected];ends=row['ends'][selected]
                cumulative=torch.cumsum(w.double(),0);A=float(w.sum())
                def depth(fraction):
                    if not len(w) or A<=0:return None
                    index=min(int(torch.searchsorted(cumulative,cumulative.new_tensor(A*fraction))),len(w)-1)
                    return float((starts[index]+ends[index])*.5)
                queries.append(dict(query_id=str(query_id),local_index=i,pixel_xy=q['pixel_xy'][i].tolist(),
                    original_global_query_index=int(q['original_global_query_index'][i]),
                    current_fit_weight=0,semantic_target=float(q['semantic_target'][i]),
                    native_alpha=A,accepted_box_capacity=float(capacity[i]),samples=int(selected.sum()),
                    cap_hit=bool(row['cap_hit'][i]),max_sigma=float(row['sigma'][selected].max()) if selected.any() else 0,
                    depth_native_mass50=depth(.5),depth_native_mass95=depth(.95),depth_native_mass99=depth(.99),
                    actual_visible_source_samples=int((w>0).sum()),
                    label_caveat='Diagnostic semantic provenance only; no opticalalphaGT or ray-empty claim.'))
            source['assert_unchanged']()
            records.append(dict(camera=cam,rays=n,samples=len(ids),archive=str(destination),archive_sha256=sha(destination),
                ray_proof_errors=errors,queries=queries,native_proof=row['proof']))
            total_rays+=n;total_samples+=len(ids)
            write(out/'progress.json',dict(cameras_done=len(records),TRAIN_rays=total_rays,samples=total_samples,source_updates=0))
        cells=json.loads(Path(spec['low_cells']).read_text())
        xyz64=np.array([cell['center']for cell in cells],np.float64)
        if xyz64.shape!=(7,3):raise ValueError('Exactly seven predeclared low cell centers required')
        xyz=torch.tensor(xyz64,dtype=torch.float32,device=model.device)
        # Field.density_fn delegates to native get_density at these exactXYZ.
        # There is no ray, view RGB, optical alpha or visibility interpretation.
        sigma=model.field.density_fn(xyz).reshape(-1)
        if not torch.isfinite(sigma).all()or(sigma<0).any():raise ValueError('Nonfinite or negative point density')
        np.savez_compressed(out/'seven_low_cell_densities.npz',xyz_declared_float64=xyz64,
            xyz_actual_float32=xyz.cpu().numpy(),sigma=sigma.cpu().numpy(),
            cell=np.array([c['cell']for c in cells],np.int32),world_to_box=np.eye(4),model_digest=np.array(source['pipeline_digest']))
    checks=source['assert_unchanged']()
    if any(sha(file)!=digest for file,digest in spec['original_cache_files'].items()):
        raise ValueError('Original frozen visibility cache changed')
    source['guard_done'].set()
    if total_rays!=21:raise ValueError('Exactly21 diagnostic TRAINrays required')
    write(out/'complete.json',dict(passed=True,records=records,TRAIN_rays=total_rays,samples=total_samples,
        point_density_queries=7,low_cell_archive_sha256=sha(out/'seven_low_cell_densities.npz'),
        source_checks=checks,source_FHD_proof=source['FHD_proof'],pipeline_digest=source['pipeline_digest'],
        source_checkpoint_sha256=sha(source['checkpoint']),world_to_box=np.eye(4).tolist(),
        source_updates=0,optimizer_created=False,head_loaded=False,selectors_rendered=False,
        original_cache_mutated=False,all_query_fit_weights=0,FD_native_center_only_diagnostic=True,
        original_cached_ray_allocator_context_not_claimed_identical=True,
        point_density_is_not_visibility_or_optical_alpha=True,full_visibility_certified=False,
        query_manifest_sha256=sha(spec['query_manifest']),spec_sha256=sha(path)))
    signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();run(args.spec,args.execute)
