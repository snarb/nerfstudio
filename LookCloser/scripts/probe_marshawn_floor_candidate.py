"""Exact new-model floor rays with one witness-union guard, no persistent edits."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from marshawn_release_common import trace_native,authoritative_grid,point_cells,native_bounds
    from build_marshawn_release_grid import witness_guard
    from probe_marshawn_occupancy import file_sha,digest_model
    from audit_marshawn_floor_rays import metrics
    torch.set_num_threads(2);source=json.loads(a.reference.read_text());selection=json.loads((a.reference.parent.parent/'selection.json').read_text())
    checkpoint=Path(source['checkpoint']);config=ROOT/'native_png_linear_distortion_STAGED/run/marshawn_001660_png_distortion_linear001/instant-ngp-bounded/cold_seed42/config.yml'
    def configure(cfg):cfg.eval_checkpoint=checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure)
    model=pipe.model;pipe.eval();native_bounds(model);before=digest_model(model);saved=model.occupancy_grid.binaries.detach().clone()
    gridroot=ROOT/'native_png_linear_distortion_STAGED/grids';stem=f'step-{step:09d}_grid'
    core,active,ema,boxes,policy=authoritative_grid(gridroot/(stem+'.npz'),gridroot/(stem+'.json'),checkpoint)
    if not np.array_equal(saved.cpu().numpy(),active):raise ValueError('Defaultsourcegridmismatch')
    record=json.loads((a.candidate/'receipt.json').read_text());archive=a.candidate/'grid.npz'
    if file_sha(archive)!=record['grid_archive_sha256']or file_sha(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Candidatebinding mismatch')
    z=np.load(archive);points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];seed,protected,expected,ordinary=witness_guard(core,boxes,points)
    if not np.array_equal(z['core'],core)or not np.array_equal(z['ema'],ema)or not np.array_equal(z['active'],expected):raise ValueError('NotexactpointunionthenONEhalo')
    _,inside,covered=point_cells(points,expected,boxes);body=np.load(ROOT/'final_pair/release_audit/radius0/body/all_witnesses.npz')['subject'].astype(bool)
    rows=[];arrays={};a.output.mkdir(exist_ok=False)
    try:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(expected,device=model.device))
        candidate_digest=digest_model(model)
        for ref in source['rows']:
            q=ref['query'];dataset=getattr(pipe.datamanager,q['split']+'_dataset');ci=[Path(p).stem.split('_')[-1]for p in dataset.image_filenames].index(q['camera']);camera=dataset.cameras[ci:ci+1].to(model.device)
            ray=camera.generate_rays(0,coords=torch.tensor([[q['y']+.5,q['x']+.5]],device=model.device),keep_shape=False)
            origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy()
            if not np.array_equal(origin,ref['origin'])or not np.array_equal(direction,ref['direction']):raise ValueError('Actualraychanged')
            colors=[]
            def hook(_m,_args,out):colors.append(out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy())
            handle=model.field.register_forward_hook(hook)
            try:trace=trace_native(model,ray)
            finally:handle.remove()
            if len(colors)!=1:raise ValueError('Expectedoneactualfieldcall')
            rgb=(trace['weights'][:,None]*colors[0]).sum(0);stat=metrics(trace,origin,direction,np.array(selection['normal']),selection['offset'],.03);stat['actual_rgb']=rgb.tolist()
            row=dict(query=q,candidate=stat,default=ref['profiles']['native'],all_cells=ref['profiles']['all_cells'],
                     candidate_default_RGB_delta=float(abs(rgb-np.array(ref['profiles']['native']['actual_rgb'])).max()),
                     candidate_allcells_RGB_delta=float(abs(rgb-np.array(ref['profiles']['all_cells']['actual_rgb'])).max()))
            rows.append(row)
            for k,v in dict(trace,rgb_samples=colors[0]).items():
                if isinstance(v,np.ndarray):arrays[f'q{q["id"]:02d}_{k}']=v
        if digest_model(model)!=candidate_digest:raise ValueError('Auditchangedcandidate/model')
    finally:
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before:raise ValueError('Originalstaterecoveryfailed')
    np.savez_compressed(a.output/'traces.npz',**arrays)
    result=dict(checkpoint_sha256=file_sha(checkpoint),candidate_receipt_sha256=file_sha(a.candidate/'receipt.json'),candidate_archive_sha256=file_sha(archive),source_reference_sha256=file_sha(a.reference),
        same9selection=True,core_and_EMA_unchanged=True,checkpoint_unchanged=True,full_original_state_restored=True,training_updates=0,
        points=int(inside.sum()),candidate_point_FN=int((inside&~covered).sum()),body_points=int(body.sum()),candidate_body_FN=int((body&inside&~covered).sum()),
        max_candidate_default_RGB_delta=max(r['candidate_default_RGB_delta']for r in rows),max_candidate_allcells_RGB_delta=max(r['candidate_allcells_RGB_delta']for r in rows),rows=rows,
        limitation='Sparsepointcoverage/admission isnotphysicaloccupancytruth;9queryRGBchangescannotcertifyallfloorpixels')
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
