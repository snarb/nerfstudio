"""Bounded native-only artifact ray capture; never produces training labels."""
import argparse, hashlib, importlib, inspect, json, os, sys
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main(plan_path):
    p=json.loads(plan_path.read_text());out=plan_path.parent/'trace';out.mkdir(exist_ok=False)
    for path,digest in p['bound_files'].items():assert sha(path)==digest,path
    os.environ['CUDA_VISIBLE_DEVICES']=p['gpu_uuid']
    sys.path.insert(0,p['code_root']);sys.path.insert(0,str(Path(p['code_root'])/'LookCloser/scripts'))
    import torch
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from marshawn_semantic_occupancy import load_snapshot
    from probe_marshawn_occupancy import digest_model,assert_consistent_bounds
    assert torch.cuda.device_count()==1
    device_uuid=str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
    assert device_uuid==p['gpu_uuid'].lower().removeprefix('gpu-'),device_uuid
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(.9,8*2**30/torch.cuda.get_device_properties(0).total_memory))
    def configure(cfg):
        cfg.eval_checkpoint=Path(p['checkpoint']);cfg.pipeline.model.occupancy_eval_dilation_radius=0
        cfg.pipeline.datamanager.data=Path(p['data']);cfg.pipeline.datamanager.dataparser.data=Path(p['data']);return cfg
    cfg,pipe,cp,step=eval_setup(Path(p['config']),test_mode='val',update_config_callback=configure)
    m=pipe.model;m.current_train_step=step;m.eval()
    m.field.feature_reweighting_strength=.3;m.config.feature_reweighting_strength=.3
    assert step==111316 and sha(cp)==p['checkpoint_sha256']
    assert m.config.adaptive_coarse_step_size==.0015625 and m.config.max_steps_per_ray==1024 and not m.config.corrected_arm_allocator
    assert m.config.occupancy_dilation_radius==0 and m._resolved_ray_sampling_mode()=='adaptive'
    assert m.field._semantic_admission_installed and m.field.feature_reweighting_strength==.3
    core,active,ema,boxes,receipt=load_snapshot(Path(p['receipt']),cp)
    assert receipt['semantic_receipt']==m.semantic_receipt
    np.testing.assert_array_equal(active,m.occupancy_grid.binaries.detach().cpu().numpy())
    np.testing.assert_array_equal(core,m.semantic_learned_core.detach().cpu().numpy())
    np.testing.assert_array_equal(ema,m.occupancy_grid.occs.detach().cpu().numpy().reshape(ema.shape))
    bounds=assert_consistent_bounds(m)
    parser=pipe.datamanager.train_dataparser_outputs
    assert parser.dataparser_scale==1 and np.array_equal(parser.dataparser_transform.cpu().numpy(),np.eye(4)[:3])
    ds=pipe.datamanager.eval_dataset;names=[Path(f).stem.split('_')[-1] for f in ds.image_filenames];assert names==['DC','HC','KA']
    bindings={name:dict(path=str(Path(inspect.getfile(importlib.import_module(name))).resolve()),sha256=sha(inspect.getfile(importlib.import_module(name)))) for name in p['import_modules']}
    bound_resolved={str(Path(path).resolve()):digest for path,digest in p['bound_files'].items()}
    for record in bindings.values():assert bound_resolved.get(record['path'])==record['sha256'],record
    cache_priming=dict(initial_shape=list(m.field._feature_weight_lut.shape),initial_strength=m.field._feature_weight_lut_strength,dtype=str(m.field.freq_grid.grid.dtype),analytical_only=True,extra_rays=0)
    # Prime only the exact analytical integer-level LUT; no field/encoding/density query.
    with torch.inference_mode():m.field._get_discrete_feature_weights(torch.arange(m.field.num_levels,device=m.device,dtype=m.field.freq_grid.grid.dtype).unsqueeze(-1))
    cache_priming.update(primed_shape=list(m.field._feature_weight_lut.shape),primed_strength=m.field._feature_weight_lut_strength)
    assert m.field._feature_weight_lut_strength==.3
    def states():
        # Capture every tensor buffer (including nonpersistent/private caches), plus registered field state.
        data={k:v.detach().cpu().numpy().copy() for k,v in m.named_buffers()}
        def extras(obj,prefix,depth=0):
            if depth>2:return
            for k,v in vars(obj).items():
                if isinstance(v,torch.Tensor):data[prefix+'.'+k]=v.detach().cpu().numpy().copy()
                elif isinstance(v,dict) and ('cache' in k or 'capture' in k):
                    for kk,vv in v.items():
                        if isinstance(vv,torch.Tensor):data[prefix+'.'+k+'.'+kk]=vv.detach().cpu().numpy().copy()
        for n,obj in [('model',m),('occupancy',m.occupancy_grid),('field',m.field)]:extras(obj,n)
        return data
    before=states();digest=digest_model(m);np.savez_compressed(out/'state_before.npz',**before)
    rows=[]
    for i,pixel in enumerate(p['pixels']):
        camera=ds.cameras[pixel['camera']:pixel['camera']+1].to(m.device)
        ray=camera.generate_rays(0,coords=torch.tensor([[pixel['y']+.5,pixel['x']+.5]],device=m.device),keep_shape=False);ray=m.collider(ray)
        captured=[]
        def capture(module,inputs,outputs):
            samples=inputs[0];captured.append([samples.frustums.starts,samples.frustums.ends,outputs[FieldHeadNames.DENSITY],outputs[FieldHeadNames.RGB]])
        hook=m.field.register_forward_hook(capture)
        try:
            with torch.inference_mode():result=m(ray)
        finally:hook.remove()
        assert len(captured)<=1
        if captured:start,end,sigma,rgb=[v.detach().float().cpu().numpy() for v in captured[0]]
        else:start=end=sigma=np.empty((0,1),np.float32);rgb=np.empty((0,3),np.float32)
        w=result['packed_weights'].detach().float().cpu().numpy().reshape(-1);o=ray.origins[0].cpu().numpy();d=ray.directions[0].cpu().numpy()
        xyz=o+((start.reshape(-1)+end.reshape(-1))*.5)[:,None]*d
        render=result['rgb'].detach().float().cpu().numpy().reshape(3);alpha=float(result['accumulation'].item())
        target=np.asarray(Image.open(p['reference_PNGs'][pixel['camera_name']]).convert('RGB'))[pixel['y'],pixel['x']]
        assert np.max(abs(render*255-target))<=1.01
        assert abs(float(w.sum())-alpha)<2e-6 and np.max(abs(w@rgb.reshape(-1,3)-render))<2e-6
        file=f'ray{i:02d}.npz';np.savez_compressed(out/file,t_start=start,t_end=end,sigma=sigma,rgb=rgb,actual_weights=w,xyz=xyz)
        rows.append(dict(pixel=pixel,origin=o.tolist(),direction=d.tolist(),near=float(ray.nears.item()),far=float(ray.fars.item()),trace=file,rgb=render.tolist(),alpha=alpha,PNG_max_byte_error=float(np.max(abs(render*255-target))),saturation=float(result['adaptive_saturation_rate'])))
    after=states();assert before.keys()==after.keys()
    for k in before:np.testing.assert_array_equal(before[k],after[k],err_msg=k)
    assert digest_model(m)==digest
    np.savez_compressed(out/'state_after.npz',**after)
    for path,digest in p['bound_files'].items():assert sha(path)==digest,path
    (out/'audit.json').write_text(json.dumps(dict(plan_sha256=sha(plan_path),worker_sha256=sha(__file__),checkpoint_sha256=sha(cp),imports=bindings,actual_gpu_uuid=device_uuid,CUDA_device_count=1,analytical_cache_priming=cache_priming,all_tensor_states_unchanged=True,state_keys=list(before),bounds=bounds,FR=.3,coarse_step=.0015625,radius=0,cap=1024,rows=rows,labels_generated=False),indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);main(ap.parse_args().plan)
