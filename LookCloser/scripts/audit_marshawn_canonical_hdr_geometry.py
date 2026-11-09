"""Read-only canonical physical-AP1 LC BODY128 and three DC prefix rays.

Reuses frozen BODY/ray mathematics while binding this positive HDR field and
its actual exported grid. No optimizer, callback, objective or training resume.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,inspect,json,sys
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    stage=a.stage
    config_path=a.config or stage/'config.yml'
    for p in reversed([ROOT/'code_sfm',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',stage/'code']):sys.path.insert(0,str(p))
    sys.path.insert(0,str(ROOT/'final_pair/candidate_inference_code'))
    import torch
    from PIL import Image,ImageDraw
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    import verify_marshawn_release as verifier
    from marshawn_release_common import native_bounds,trace_native
    from probe_marshawn_occupancy import file_sha,digest_model
    from marshawn_canonical_physical_hdr import common_sdr
    from marshawn_canonical_hdr_preflight import canonical_pq_from_unbaked_ap1
    from audit_marshawn_near_prefix import suffix_profiles
    torch.set_num_threads(2)
    source_grid=stage/'grids'/('step%09d_core0.npz'%a.step);source_receipt=stage/'grids'/('step%09d_receipt.json'%a.step);policy=json.loads(source_receipt.read_text());checkpoint=Path(policy['checkpoint']);spec=json.loads((stage/'run_spec.json').read_text());context={}
    assert file_sha(checkpoint)==policy['checkpoint_sha256'] and policy['step']==a.step and file_sha(source_grid)==policy['grid_sha256']
    def runtime(_args):
        def configure(c):c.eval_checkpoint=checkpoint;c.pipeline.model.eval_num_rays_per_chunk=512;return c
        cfg,pipe,loaded,step=eval_setup(config_path,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);assert Path(loaded).resolve()==checkpoint.resolve()and step==a.step;pipe.eval();m=pipe.model
        assert type(m).__name__=='CanonicalAP1HDRModel' and m.config.rgb_output_parameterization=='linear_softplus' and not m.renderer_rgb.clamp_output
        assert m.config.ray_sampling_mode=='occupancy'and m.config.render_step_size==.003 and not m.field.enable_feature_reweighting and not m.freq_grid.enabled
        assert m.config.background_color=='black' and len(pipe.datamanager.train_dataset)==63 and len(pipe.datamanager.eval_dataset)==3
        assert pipe.datamanager.train_dataset.metadata['image_color_space']=='linear-ACES-AP1-D60'
        normal=pipe.datamanager.train_dataparser_outputs;assert normal.dataparser_scale==1 and np.allclose(normal.dataparser_transform.cpu().numpy(),np.eye(4)[:3],atol=1e-7,rtol=0)
        with np.load(source_grid)as z:active=z['active'].copy();ema=z['occs'].copy();boxes=z['aabb'].copy()
        assert active.dtype==np.bool_ and active.shape==(1,128,128,128) and boxes.shape==(1,6)
        assert np.array_equal(active,m.occupancy_grid.binaries.cpu().numpy())and np.array_equal(ema,m.occupancy_grid.occs.cpu().numpy())
        sources={}
        for name,module in [('pipeline',pipe),('model',m),('field',m.field),('occupancy_estimator',m.occupancy_grid)]:
            path=inspect.getsourcefile(type(module));sources[name]=dict(path=path,sha256=file_sha(path))
            if path in spec['sources_sha256']:assert file_sha(path)==spec['sources_sha256'][path]
        bounds=native_bounds(m);assert np.allclose(boxes[0].reshape(2,3),bounds['field.aabb'],atol=1e-6,rtol=0)
        receipt=dict(config=str(config_path),config_sha256=file_sha(config_path),checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),step=step,live_bounds=bounds,runtime_sources=sources,normalization=dict(scale=1.,transform=normal.dataparser_transform.tolist()),source_policy=dict(policy,radius=0,core_cells=int(active.sum()),active_cells=int(active.sum()),grid_archive=str(source_grid),grid_archive_sha256=file_sha(source_grid),grid_archive_format='CanonicalHDR actual native active/occs/aabb; aliases only in reader',core_update_step='Not separately exported; mask/EMA verified bitwise against checkpoint',authority='Exported actual native GPU binary mask, never CPU EMA threshold reconstruction'),physical_color_contract='Unbaked relative AP1D60 linear, signedgain52.13360318135631/floor.005/peak4000loss only',script_sha256=file_sha(__file__))
        context.update(pipe=pipe,boxes=boxes,receipt=receipt);return pipe,active,active,boxes,receipt
    def save_crop(model,dataset,index,uv,output,title):
        name=Path(dataset.image_filenames[index]).stem.split('_')[-1];ev=model.config.canonical_old_scalar_EV[name];image=dataset.get_image_float32(index).cpu().numpy();h,w=image.shape[:2];x,y=np.clip(np.floor(np.asarray(uv)-64).astype(int),[0,0],[w-128,h-128]);cam=copy.deepcopy(dataset.cameras[index:index+1]).to(model.device);cam.cx-=int(x);cam.cy-=int(y);cam.width.fill_(128);cam.height.fill_(128)
        with torch.inference_mode():pred=model.get_outputs_for_camera(cam)['rgb'];gt=torch.tensor(image[y:y+128,x:x+128,:3],device=model.device);shown=[common_sdr(v,model.config.canonical_ap1_to_709,ev).cpu().numpy()for v in [gt,pred]]
        canvas=Image.new('RGB',(512,280),'#202020');draw=ImageDraw.Draw(canvas);draw.text((3,3),title+' GT|HDRfield; commonAP1->709display',fill='white')
        for col,value in enumerate(shown):
            canvas.paste(Image.fromarray(np.rint(value*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST),(col*256,24));u,v=(np.asarray(uv)-[x+.5,y+.5])*2;draw.ellipse((col*256+u-6,24+v-6,col*256+u+6,24+v+6),outline='red',width=2)
        canvas.save(output,quality=95);return dict(path=Path(output).name,bbox_xyxy=[int(x),int(y),int(x+128),int(y+128)],camera_index=index,uv=list(uv),display='AP1D60→709,gain4×oldfixedcameraEV,Reinhard+sRGB identicalGT/pred; nofit')
    verifier.runtime=runtime;verifier.save_crop=save_crop
    args=SimpleNamespace(output=a.output/'body128',data=ROOT/'color_variants/canonical_signed_ap1/linear_ap1_exr',selection=ROOT/'staged_hdr_occupancy4/gate8k_body128/selection.npz',observations=ROOT/'frequency_seed/observations.npz',witnesses=ROOT/'geometry/train_witnesses.npz',subject=ROOT/'geometry/subject_witnesses.npz',uniform_samples=8192)
    a.output.mkdir(exist_ok=False)
    if a.dc_only:runtime(args)
    else:verifier.body(args)
    original_trace=trace_native
    if a.mask:
        from compare_marshawn_release_grids import applied_mask
        with np.load(a.mask) as z:override=z['active'].copy()
        def trace_native(model,ray):
            with applied_mask(model,override):return original_trace(model,ray)
    pipe=context['pipe'];m=pipe.model;before=digest_model(m);ds=pipe.datamanager.eval_dataset;ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index('DC');cam=ds.cameras[ci:ci+1].to(m.device);gt=ds.get_image_float32(ci).cpu().numpy();rows=[];arrays={}
    for i,(x,y)in enumerate([(1600,250),(1700,450),(1400,300)]):
        ray=cam.generate_rays(0,coords=torch.tensor([[y+.5,x+.5]],device=m.device),keep_shape=False);colors=[];hook=m.field.register_forward_hook(lambda _m,_a,out:colors.append(out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy()))
        try:trace=trace_native(m,ray)
        finally:hook.remove()
        assert len(colors)==1;rgb=colors[0];full,derived=suffix_profiles(trace['starts'],trace['ends'],trace['sigma'],rgb,a.cuts);mid=(trace['starts']+trace['ends'])/2;w=trace['weights'];rank=int(np.argmax(w));q50=int(np.searchsorted(np.cumsum(w),w.sum()*.5));o=ray.origins[0].cpu().numpy();d=ray.directions[0].cpu().numpy()
        for cut in full['cuts']:
            with torch.inference_mode():cut['suffix_signed_PQ']=canonical_pq_from_unbaked_ap1(torch.tensor(cut['suffix_RGB'],device=m.device)).cpu().tolist()
        rows.append(dict(pixel=[x,y],origin=o.tolist(),direction=d.tolist(),GT_linear_AP1=gt[y,x].tolist(),profile=full,actual_packed_weight_max_error=trace['weight_max_error'],dominant_weight_depth=float(mid[rank]),dominant_weight_XYZ=(o+mid[rank]*d).tolist(),conditional_median_depth=float(mid[q50]),conditional_median_XYZ=(o+mid[q50]*d).tolist(),prefix_is_from_native_near=float(m.config.near_plane),mask_override=str(a.mask) if a.mask else None))
        for key,value in dict(trace,rgb_samples=rgb).items():
            if isinstance(value,np.ndarray):arrays[f'r{i}_{key}']=value
    assert digest_model(m)==before;np.savez_compressed(a.output/'DC_three_native_prefix_traces.npz',**arrays);report=dict(context['receipt'],rows=rows,predeclared_pixels=[[1600,250],[1700,450],[1400,300]],cuts=a.cuts,mask_override=str(a.mask) if a.mask else None,mask_override_sha256=file_sha(a.mask) if a.mask else None,physical_AP1_linear_integration=True,same_sample_partial_cell_analytic_suffix=True,GT_diagnostic_only_no_fit=True,frozen_state_unchanged=True,state_sha256=before,optimizer_steps=0,no_model_or_grid_or_nearplane_changes=True);(a.output/'DC_three_native_prefix.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(step=a.step,output=str(a.output),DC_prefix_rows=rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--dc-only',action='store_true');p.add_argument('--mask',type=Path);p.add_argument('--cuts',type=float,nargs='+',default=[.02,.05,.1]);main(p.parse_args())
