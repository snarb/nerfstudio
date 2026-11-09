"""Frozen-field, same-sample causal near-prefix RGB/opacity diagnostic."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def suffix_profiles(starts,ends,sigma,rgb,cuts):
    """Exact Beer-Lambert integration of piecewise-constant sigma/RGB cells."""
    starts,ends,sigma,rgb=[np.asarray(x,np.float64)for x in [starts,ends,sigma,rgb]]
    if starts.ndim!=1 or ends.shape!=starts.shape or sigma.shape!=starts.shape or rgb.shape!=(len(starts),3):raise ValueError('Invalid shapes')
    if not all(np.isfinite(x).all()for x in [starts,ends,sigma,rgb])or np.any(ends<=starts)or np.any(sigma<0):raise ValueError('Invalid samples')
    dt=ends-starts;tau=sigma*dt;trans=np.exp(-np.r_[0,np.cumsum(tau[:-1])]);w=trans*(-np.expm1(-tau));full=(w[:,None]*rgb).sum(0);alpha=float(w.sum());out=[]
    for cut in cuts:
        length=np.clip(cut-starts,0,dt);optical=float((sigma*length).sum());T=float(np.exp(-optical));wp=trans*(-np.expm1(-sigma*length));Cp=(wp[:,None]*rgb).sum(0);ap=float(wp.sum())
        if T<=1e-12:raise ValueError('Near-prefix fully opaque: stable suffix needs direct log-domain recomposition')
        # Same sampled cells; partial cut cell keeps its sigma/RGB. No shifted grid.
        suffix=(full-Cp)/T;As=(alpha-ap)/T
        rem=np.clip(ends-np.maximum(starts,cut),0,dt);rt=sigma*rem;Tw=np.exp(-np.r_[0,np.cumsum(rt[:-1])]);wr=Tw*(-np.expm1(-rt));direct=(wr[:,None]*rgb).sum(0)
        if np.max(np.abs(direct-suffix))>1e-10 or abs(wr.sum()-As)>1e-10:raise ValueError('Same-sample suffix identity failed')
        out.append(dict(cutoff=float(cut),prefix_optical_thickness=optical,prefix_mass=ap,prefix_premult_RGB=Cp.tolist(),prefix_conditional_RGB=(Cp/ap).tolist()if ap>1e-12 else None,T_prefix=T,suffix_RGB=suffix.tolist(),suffix_opacity=float(As),delta_RGB=(suffix-full).tolist()))
    return dict(RGB=full.tolist(),opacity=alpha,cuts=out),dict(T=trans,weights=w)


def main(a):
    sys.path.insert(0,str(ROOT/'final_pair/candidate_inference_code'))
    from verify_marshawn_release import runtime
    from marshawn_release_common import ray_box_interval,trace_native
    from probe_marshawn_occupancy import file_sha,digest_model
    import torch
    from nerfstudio.cameras.rays import RaySamples,Frustums
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from types import SimpleNamespace
    stage=a.stage;plan=json.loads((stage/'launch_plan.json').read_text());grid=stage/'grids/step-000014000_grid.npz';receipt=grid.with_suffix('.json');binding=json.loads(receipt.read_text());checkpoint=Path(binding['checkpoint']);assert binding['checkpoint_sha256']=='e92e525fe7451ef9cc2959ecca57a09e68925b20143714bf463c54a6ed998ae9'
    args=SimpleNamespace(config=Path(plan['expected_run_dir'])/'config.yml',checkpoint=checkpoint,grid=grid,grid_receipt=receipt,data=None,chunk_size=512,memory_gib=8,code_root=[ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',stage/'code'])
    pipe,core,active,boxes,report=runtime(args);model=pipe.model;before=digest_model(model);ds=pipe.datamanager.eval_dataset;ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index('KA');camera=ds.cameras[ci:ci+1].to(model.device);gt=ds.get_image_float32(ci).cpu().numpy()
    selection=ROOT/'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/selection.json';queries=[dict(x=q['x'],y=q['y'],group='floor9',label='floor9_'+str(i))for i,q in enumerate(json.loads(selection.read_text())['queries'])]
    radiometry=stage/'radiometry14k_DIAGNOSTIC/audit.json'
    for row in json.loads(radiometry.read_text())['rows']:
        if row['camera']=='KA'and row['label']in ['cardboard_A','cardboard_B','paper_marker_A']:
            x0,y0,x1,y1=row['bbox'];queries.append(dict(x=(x0+x1)//2,y=(y0+y1)//2,group='background_control',label=row['label'],source_bbox=row['bbox']))
    a.output.mkdir(exist_ok=False);rows=[];arrays={};cuts=[.01,.02,.03,.05,.1]
    with torch.inference_mode():
        for i,q in enumerate(queries):
            x,y=q['x'],q['y'];ray=camera.generate_rays(0,coords=torch.tensor([[y+.5,x+.5]],device=model.device),keep_shape=False);origin=ray.origins[0].cpu().numpy();direction=ray.directions[0].cpu().numpy();near,far=ray_box_interval(origin,direction,boxes[0],model.config.near_plane,model.config.far_plane);row=dict(q,origin=origin.tolist(),direction=direction.tolist(),near=near,far=far,GT_RGB=gt[y,x].tolist(),profiles={})
            for count in [16384,32768]:
                edges=torch.linspace(near,far,count+1,device=model.device,dtype=torch.float32);samples=RaySamples(frustums=Frustums(origins=ray.origins.expand(count,3),directions=ray.directions.expand(count,3),starts=edges[:-1,None],ends=edges[1:,None],pixel_area=torch.ones((count,1),device=model.device)),camera_indices=ray.camera_indices.expand(count,1));field=model.field(samples);sigma=field[FieldHeadNames.DENSITY].float().cpu().numpy().reshape(-1);rgb=field[FieldHeadNames.RGB].float().cpu().numpy().reshape(-1,3);starts=edges[:-1].cpu().numpy();ends=edges[1:].cpu().numpy();profile,derived=suffix_profiles(starts,ends,sigma,rgb,cuts);row['profiles'][str(count)]=profile
                for key,v in dict(starts=starts,ends=ends,sigma=sigma,rgb=rgb,**derived).items():arrays[f'r{i:02d}_n{count}_{key}']=v
            colors=[]
            handle=model.field.register_forward_hook(lambda _m,_a,out:colors.append(out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy()))
            try:native=trace_native(model,ray)
            finally:handle.remove()
            assert len(colors)==1
            profile,_=suffix_profiles(native['starts'],native['ends'],native['sigma'],colors[0],cuts)
            assert np.max(abs(np.array(profile['RGB'])-(native['weights'][:,None]*colors[0]).sum(0)))<1e-4
            profile['actual_packed_weight_error']=native['weight_max_error'];row['profiles']['native']=profile
            for key,value in dict(native,rgb=colors[0]).items():
                if isinstance(value,np.ndarray):arrays[f'r{i:02d}_native_{key}']=value
            row['convergence']=dict(full_RGB_max_error=float(np.max(np.abs(np.array(row['profiles']['16384']['RGB'])-row['profiles']['32768']['RGB']))),full_alpha_error=abs(row['profiles']['16384']['opacity']-row['profiles']['32768']['opacity']),suffix_RGB_max_error=max(float(np.max(np.abs(np.array(p['suffix_RGB'])-r['suffix_RGB'])))for p,r in zip(row['profiles']['16384']['cuts'],row['profiles']['32768']['cuts'])))
            rows.append(row);(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(queries)))+'\n')
    assert digest_model(model)==before
    np.savez_compressed(a.output/'traces.npz',**arrays);report.update(script_sha256=file_sha(__file__),selection_sha256=file_sha(selection),radiometry_sha256=file_sha(radiometry),rows=rows,cuts=cuts,uniform_samples=[16384,32768],field_query='NativefieldFP32query/output without AMP, nativeTCNN internals unchanged',integration='NumPyFP64 piecewise-constant BeerLambert; SAMEsamples partial-cell prefix/suffix; independent suffix identity checked1e-10',near_scope='Native near .01: no density below original near is evaluated. Cutoff.01 is identity.',frozen_state_unchanged=True,state_sha256=before,optimizer_steps=0,no_GT_fit_or_compositing=True,no_production_camera_grid_NN_changes=True,limitations=['Raycolour is encoded scalarSDR nativefield output, not physical scene-linear energy.','Backgroundcontrol ROIcenters are predeclared diagnostics, not calibrated diffuse references.','Analytic prefix removal is a ray counterfactual, not sourcegeometry approval of a production near-plane change.']);(a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(rays=len(rows),output=str(a.output),state_unchanged=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
