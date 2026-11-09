"""Actual bounded linear709 pipeline, two AMP packets, no optimizer object or updates."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
ROOT = Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_sfm/LookCloser/scripts', ROOT/'code_native/LookCloser/scripts',
          ROOT/'code_native', Path(__file__).parent]:
    sys.path.insert(0, str(d))
import numpy as np
import torch
import yaml
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.scripts.train import _set_random_seed
from probe_marshawn_occupancy import digest_model
from run_marshawn_linear709_bounded_cold import load
from staged_native_hash import file_sha


def gradient(model, value, scale):
    model.zero_grad(set_to_none=True)
    (value*scale).backward(retain_graph=True)
    groups = {}
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        g = parameter.grad.detach().double()/scale
        finite = finite and bool(torch.isfinite(g).all())
        group = 'geometry' if name.startswith('field.mlp_base.') else 'color'
        groups[group] = groups.get(group, 0.)+float(g.square().sum())
    return dict(finite=finite, group_l2={k:v**.5 for k,v in groups.items()})


def main(a):
    a.output.mkdir(exist_ok=False)
    cfg, spec = load(a.config, a.spec)
    if not spec['release_checks']['data_sealed']:
        raise ValueError('No model construction before data seal')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(1.,35*1024**3/torch.cuda.get_device_properties(0).total_memory))
    again = yaml.load(yaml.dump(cfg), Loader=yaml.Loader)
    assert again.pipeline._target is cfg.pipeline._target and '<locals>' not in yaml.dump(cfg)
    from marshawn_radial_native_candidate import NativeRadialColdPipeline
    from nerfstudio.models.instant_ngp import NGPModel
    from marshawn_linear709_bounded_hdr import bounded_linear,display_sdr,pq203,MAX_LINEAR
    from nerfstudio.field_components.field_heads import FieldHeadNames
    import nerfacc,cv2
    import gc
    reference_cfg = copy.deepcopy(cfg.pipeline)
    reference_cfg._target = NativeRadialColdPipeline
    reference_cfg.model._target = NGPModel
    _set_random_seed(42)
    reference = reference_cfg.setup(device='cuda', test_mode='val')
    reference_hash = digest_model(reference.model)
    reference_schema = [(n,list(p.shape),str(p.dtype)) for n,p in reference.model.named_parameters()]
    reference_keys = set(reference.state_dict())
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    _set_random_seed(42)
    pipe = again.pipeline.setup(device='cuda', test_mode='val')
    pipe.train()
    model = pipe.model
    initial = digest_model(model)
    assert not model.renderer_rgb.clamp_output
    assert pipe.datamanager.config.cache_images_type == 'float32'
    loader_proof=[]
    for i,path in enumerate(pipe.datamanager.eval_dataset.image_filenames):
        expected=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1].copy()
        actual=pipe.datamanager.eval_dataset.get_image_float32(i).numpy()
        assert np.array_equal(expected,actual) and actual.dtype==np.float32
        loader_proof.append(dict(camera=path.stem,bitexact=True,max=float(actual.max()),count_above1=int((actual>1).sum())))
    assert all(q['count_above1']>0 for q in loader_proof)
    synthetic=torch.tensor([0.,.5,1.],device='cuda')
    v=bounded_linear(synthetic,model.config.linear709_median_y)
    torch.testing.assert_close(v,torch.tensor([0.,model.config.linear709_median_y,MAX_LINEAR],device='cuda'),rtol=1e-6,atol=1e-7)
    model.renderer_rgb.eval()
    test=model.renderer_rgb(torch.tensor([[[2.,3.,4.]]],device='cuda'),torch.ones((1,1,1),device='cuda'))
    assert float(test.max())==4
    model.renderer_rgb.train()
    field_calls=[]
    def before_physical(m,args,out):
        field_calls.append(dict(encoded=out[FieldHeadNames.RGB].detach(),density=out[FieldHeadNames.DENSITY].detach().clone()))
    def after_physical(m,args,out):
        old=field_calls[-1]
        assert torch.equal(old['density'],out[FieldHeadNames.DENSITY])
        assert out[FieldHeadNames.RGB].dtype==torch.float32
        torch.testing.assert_close(out[FieldHeadNames.RGB],bounded_linear(old['encoded'],model.config.linear709_median_y),rtol=0,atol=0)
    handles=[model.field.register_forward_hook(before_physical,prepend=True),model.field.register_forward_hook(after_physical)]
    packet_checks=[]
    def depth(m,args,kw):
        w=kw['weights'].detach();samples=kw['ray_samples'];indices=kw['ray_indices'];sigma=field_calls[-1]['density']
        ref=nerfacc.render_weight_from_density(samples.frustums.starts[...,0],samples.frustums.ends[...,0],sigma[...,0],ray_indices=indices,n_rays=kw['num_rays'])[0][:,None]
        err=float((w-ref).abs().max()) if w.numel() else 0.
        assert err<1e-6
        packet_checks.append(dict(rays=int(kw['num_rays']),samples=len(w),weights_density_reconstruction_maxerror=err,field_density_bytes_unchanged=True))
    handles.append(model.renderer_depth.register_forward_pre_hook(depth,with_kwargs=True))
    captured_batches=[]
    original_next=pipe.datamanager.next_train
    def next_train(step):
        rb,b=original_next(step);captured_batches.append(b);return rb,b
    pipe.datamanager.next_train=next_train
    assert initial == reference_hash
    assert [(n,list(p.shape),str(p.dtype)) for n,p in model.named_parameters()] == reference_schema
    assert set(pipe.state_dict()) == reference_keys
    assert model.field.appearance_embedding_dim == 0
    assert model.config.near_plane == .052 and model.renderer_rgb.background_color == 'random'
    assert pipe.dynamic_num_rays_per_batch == pipe.datamanager.train_pixel_sampler.num_rays_per_batch == 64
    assert len(pipe.datamanager.train_dataset) == 63 and len(pipe.datamanager.eval_dataset) == 3
    assert not hasattr(pipe, '_floor_band_prior')
    callbacks = pipe.get_training_callbacks(TrainingCallbackAttributes(None,None,pipe,None))
    names = [c.func.__name__ for c in callbacks]
    assert names == ['update_occupancy_grid','guard_after_native_core']
    original_grid = {k:v.clone() for k,v in model.occupancy_grid.state_dict().items()}
    rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    results = []
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for callback in callbacks:
                callback.run_callback_at_location(0,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            after_callback = digest_model(model)
            for step in [0,1]:
                rays = pipe.dynamic_num_rays_per_batch
                with torch.autocast(device_type='cuda',enabled=cfg.mixed_precision):
                    outputs,losses,metrics = pipe.get_train_loss_dict(step)
                batch=captured_batches[-1]
                indices=batch['indices'].cpu().numpy();expected_RGB=[];train_cache={}
                for image_index,y,xpos in indices:
                    path=pipe.datamanager.train_dataset.image_filenames[int(image_index)]
                    if int(image_index) not in train_cache:train_cache[int(image_index)]=pipe.datamanager.train_dataset.get_image_float32(int(image_index)).numpy()
                    expected_RGB.append(train_cache[int(image_index)][int(y),int(xpos)])
                assert np.array_equal(np.asarray(expected_RGB),batch['image'].detach().cpu().numpy())
                expected = {'rgb_loss','observed_sfm_free_space','observed_sfm_surface_band',
                            'packed_distortion','MVS_distribution_depth'}
                assert set(losses) == expected
                assert all(bool(torch.isfinite(v).all()) for v in outputs.values() if torch.is_tensor(v))
                packet = dict(step=step,actual_main_rays=rays,
                    main_packed_samples=int(metrics['num_samples_per_batch']),
                    next_dynamic_rays=pipe.dynamic_num_rays_per_batch,
                    MVS_extra_samples=int(metrics['MVS_depth_samples']),
                    MVS_min_opacity=float(metrics['MVS_depth_opacity_min']),
                    all_terms_finite=all(bool(torch.isfinite(v)) for v in losses.values()),
                    loss_keys=sorted(losses),actual_train_GT_bitexact=True,physical_rgb_dtype=str(outputs['rgb'].dtype),prediction_max=float(outputs['rgb'].max()),floor_fraction=float((outputs['rgb']*203<.005).float().mean()),AMP=[])
                for scale in [cfg.grad_scaler_init_scale,cfg.grad_scaler_init_scale/2,
                              cfg.grad_scaler_init_scale/4,cfg.grad_scaler_init_scale/8]:
                    values = {k:gradient(model,v,scale) for k,v in losses.items()}
                    values['combined'] = gradient(model,sum(losses.values()),scale)
                    rgb = values['rgb_loss']['group_l2'].get('geometry',0.)
                    for k in expected-{'rgb_loss'}:
                        values[k]['geometry_to_RGB_norm_ratio'] = values[k]['group_l2'].get('geometry',0.)/rgb if rgb else None
                    packet['AMP'].append(dict(scale=scale,gradients=values))
                    if all(v['finite'] for v in values.values()):
                        break
                else:
                    raise FloatingPointError('No finite complete cold AMP backward within8x native diagnostic backoff')
                assert digest_model(model) == after_callback
                model.zero_grad(set_to_none=True)
                results.append(packet)
                (a.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
            for callback in callbacks:
                callback.run_callback_at_location(16,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            assert torch.equal(model.occupancy_grid.binaries,pipe._actual_core)
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1])
        model.occupancy_grid.load_state_dict(original_grid)
        model.zero_grad(set_to_none=True)
        for h in handles:h.remove()
    assert digest_model(model) == initial
    assert torch.equal(torch.get_rng_state(),rng[2])
    assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),rng[3]))
    result = dict(passed=True,time=time.time(),config_sha256=file_sha(a.config),
        spec_sha256=file_sha(a.spec),source_sha256=file_sha(__file__),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        optimizer_objects_created=0,optimizer_updates=0,no_checkpoint_loaded=True,
        initial_NN_equals_same_seed_native_baseline=True,identical_parameter_schema_state_keys=True,
        YAML_target_roundtrip=True,callback_order=names,callback_steps=[0,16],
        model_original_grid_and_global_RNG_restored=True,production_scaler_unchanged=cfg.grad_scaler_init_scale,
        diagnostic_backoff_only=True,packets=results,loader_all3=loader_proof,field_API_physical_FP32=True,renderer_eval_does_not_clip_HDR=True,actual_density_weights_checks=packet_checks,linear_black0_objective_floor_only=True,primary_PQ_MSE_multiplier=float(model.config.linear709_pq_mse_multiplier))
    (a.output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
