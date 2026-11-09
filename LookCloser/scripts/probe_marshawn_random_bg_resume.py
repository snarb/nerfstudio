"""Native full-state/random-background proof; forward/backward only, zero steps."""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:sys.path.insert(0,str(d))
import numpy as np
import torch
from PIL import Image
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.field_components.field_heads import FieldHeadNames
from marshawn_packed_distortion_candidate import PackedDistortionCapture
from probe_marshawn_occupancy import digest_model
from run_marshawn_random_bg_resume import load,validate,atomic_json
from staged_native_hash import file_sha


def equal(a,b):
    if torch.is_tensor(b):assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(b,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:equal(a[k],b[k])
    elif isinstance(b,(list,tuple)):
        assert type(a)==type(b)and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def param_grad(term,model,scale):
    model.zero_grad(set_to_none=True)
    values=torch.autograd.grad(term*scale,[p for p in model.parameters()if p.requires_grad],retain_graph=True,allow_unused=True)
    finite=all(bool(torch.isfinite(g).all())for g in values if g is not None)
    norm=sum(float((g.detach().double()/scale).square().sum())for g in values if g is not None)**.5 if finite else None
    return dict(finite=finite,unscaled_l2=norm,scale=scale)


def backoff_diagnostic(black,random,model,parent_scale):
    """Reuse one graph, change only multiplication scale; never update scaler."""
    rows=[];scale=parent_scale
    for _ in range(12):
        baseline=param_grad(black,model,scale);candidate=param_grad(random,model,scale)
        rows.append(dict(scale=scale,black=baseline,random=candidate))
        if baseline['finite']and candidate['finite']:break
        scale*=.5
    model.zero_grad(set_to_none=True)
    return rows


def packet_probe(pipe,step,scale,output,label):
    model=pipe.model;captures={};original_next=pipe.datamanager.next_train
    original_blend=model.renderer_rgb.blend_background_for_loss_computation
    def next_train(n):
        ray,batch=original_next(n);captures['rays']=ray;captures['batch']=batch;return ray,batch
    def blend(*args,**kwargs):
        value=original_blend(*args,**kwargs)
        captures['GT_before']=kwargs['gt_image'];captures['GT_after']=value[1]
        captures['blended_prediction']=value[0]
        return value
    def field_hook(module,args,out):captures['sigma']=out[FieldHeadNames.DENSITY]
    had_next='next_train'in pipe.datamanager.__dict__;had_blend='blend_background_for_loss_computation'in model.renderer_rgb.__dict__
    pipe.datamanager.next_train=next_train;model.renderer_rgb.blend_background_for_loss_computation=blend
    hook=model.field.register_forward_hook(field_hook)
    try:
        with PackedDistortionCapture(model)as packet:
            with torch.autocast(device_type='cuda',enabled=True):outputs,losses,metrics=pipe.get_train_loss_dict(step)
    finally:
        hook.remove()
        if had_next:pipe.datamanager.next_train=original_next
        else:del pipe.datamanager.next_train
        if had_blend:model.renderer_rgb.blend_background_for_loss_computation=original_blend
        else:del model.renderer_rgb.blend_background_for_loss_computation
    assert torch.equal(captures['GT_before'],captures['GT_after'])and captures['GT_after'].shape[-1]==3
    assert set(losses)=={'rgb_loss','observed_sfm_free_space','observed_sfm_surface_band','packed_distortion'}
    try:
        model.renderer_rgb.background_color='black'
        black=model.get_loss_dict(outputs,captures['batch'])['rgb_loss']
    finally:model.renderer_rgb.background_color='random'
    other=sum(v for k,v in losses.items()if k!='rgb_loss')
    backoff=backoff_diagnostic(black+other,sum(losses.values()),model,scale)
    full_black=backoff[0]['black'];full_random=backoff[0]['random']
    sigma=captures['sigma'];indices=packet.packet['ray_indices'];n=packet.packet['num_rays']
    gb=torch.autograd.grad(black,sigma,retain_graph=True)[0].detach().flatten()
    gr=torch.autograd.grad(losses['rgb_loss'],sigma,retain_graph=True)[0].detach().flatten()
    ga=torch.autograd.grad(losses['rgb_loss'],outputs['accumulation'],retain_graph=True)[0].detach().flatten()
    rb=torch.zeros(n,device=sigma.device);rr=rb.clone()
    rb.index_add_(0,indices,gb.float()*sigma.detach().float().flatten());rr.index_add_(0,indices,gr.float()*sigma.detach().float().flatten())
    alpha=outputs['accumulation'].detach().flatten();counts=torch.zeros(n,device=sigma.device);counts.index_add_(0,indices,torch.ones_like(indices,dtype=torch.float32))
    low=(alpha<.95)&(counts>0)
    conditional=outputs['rgb'].detach()/alpha[:,None].clamp_min(1e-8)
    color_error=(conditional-captures['GT_after']).abs().mean(-1)
    matched=low&(color_error<.1)
    def summary(mask):
        return dict(rays=int(mask.sum()),black_median_dL_dlog_sigma=float(rb[mask].median())if mask.any()else None,
            random_median_dL_dlog_sigma=float(rr[mask].median())if mask.any()else None,
            random_fraction_negative=float((rr[mask]<0).float().mean())if mask.any()else None,
            random_direct_alpha_gradient_median=float(ga[mask].median())if mask.any()else None,
            random_direct_alpha_gradient_negative_fraction=float((ga[mask]<0).float().mean())if mask.any()else None)
    np.savez_compressed(output/(label+'_matched_TRAIN_packet.npz'),indices=captures['batch']['indices'].detach().cpu().numpy(),
        GT=captures['GT_after'].detach().cpu().numpy(),premult_RGB=outputs['rgb'].detach().cpu().numpy(),alpha=alpha.cpu().numpy(),
        conditional_RGB_error=color_error.cpu().numpy(),dL_dlog_sigma_black=rb.cpu().numpy(),dL_dlog_sigma_random=rr.cpu().numpy(),
        direct_dL_dalpha_random=ga.cpu().numpy())
    reference=output.parent/'actual_zeroopt_probe_LOCAL_v2'/(label+'_matched_TRAIN_packet.npz')
    with np.load(reference)as saved_packet:
        assert np.array_equal(saved_packet['indices'],captures['batch']['indices'].detach().cpu().numpy())
        assert np.array_equal(saved_packet['GT'],captures['GT_after'].detach().cpu().numpy())
    # Rendering parity is evaluated on the same physical rays without stochastic marching.
    model.eval();rng=_capture_rng_state()
    with torch.inference_mode():
        model.renderer_rgb.background_color='black';black_eval=model(captures['rays'])
        black_repeat=model(captures['rays'])
        model.renderer_rgb.background_color='random';random_eval=model(captures['rays'])
    differences={k:float((black_eval[k]-random_eval[k]).abs().max())for k in['rgb','accumulation','depth']}
    repeat_noise={k:float((black_eval[k]-black_repeat[k]).abs().max())for k in differences}
    tolerances={k:max(1e-6,4*repeat_noise[k])for k in differences}
    for k in differences:torch.testing.assert_close(black_eval[k],random_eval[k],rtol=0,atol=tolerances[k])
    equal(_capture_rng_state(),rng);model.train()
    return dict(actual_rays=n,packed_samples=int(metrics['num_samples_per_batch']),physical_train_cameras=int(captures['batch']['indices'][:,0].unique().numel()),
        GT_RGB3_exact_unchanged=True,full_AMP_black=full_black,full_AMP_random=full_random,
        readonly_same_graph_backoff_diagnostic=backoff,
        finite_at_scale=backoff[-1]['scale']if backoff[-1]['random']['finite']and backoff[-1]['black']['finite']else None,
        production_scaler_modified=False,replayed_v2_TRAIN_indices_and_GT_exact=True,
        opacity=dict(min=float(alpha.min()),median=float(alpha.median()),max=float(alpha.max()),
            fraction_below095=float((alpha<.95).float().mean()),fraction_below099=float((alpha<.99).float().mean()),
            fraction_empty_native_packet=float((counts==0).float().mean())),
        random_BG_added_RGB_mean_abs=float((captures['blended_prediction']-outputs['rgb']).detach().abs().mean()),
        random_BG_added_RGB_max_abs=float((captures['blended_prediction']-outputs['rgb']).detach().abs().max()),
        black_RGB_MSE=float(black.detach()),random_RGB_MSE=float(losses['rgb_loss'].detach()),
        sigma_gradient_definition='d RGB-loss / d log(global per-ray density scale), actual packed graph; negative favors increasing density',
        direct_alpha_gradient_definition='Partial dRGB-loss/dalpha holding premultipliedRGB fixed; total density-path direction reported separately',
        all_low_alpha_nonempty_TRAIN=summary(low),low_alpha_TRAIN_conditional_RGB_MAE_below_point1=summary(matched),
        eval_black_vs_random_max_error=differences,eval_black_repeat_max_error=repeat_noise,
        eval_FP32_repeat_control_tolerance=tolerances,eval_forward_RNG_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);cfg,spec=load(a.config,a.spec);saved=torch.load(cfg.load_checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==8000;original_rng=_capture_rng_state();cfg=copy.deepcopy(cfg);cfg.output_dir=a.output/'setup_outputs'
    metadata=json.loads((cfg.pipeline.datamanager.dataparser.data/'transforms.json').read_text())
    for frame in metadata['frames']:
        with Image.open(cfg.pipeline.datamanager.dataparser.data/frame['file_path'])as im:assert im.mode=='RGB'
    assert len(metadata['frames'])==66
    try:
        _set_random_seed(42);trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();validate(trainer,spec)
        pipe=trainer.pipeline;model=pipe.model;before=digest_model(model)
        equal(pipe.state_dict(),saved['pipeline'])
        for k,opt in trainer.optimizers.optimizers.items():equal(opt.state_dict(),saved['optimizers'][k])
        for k,s in trainer.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
        equal(trainer.grad_scaler.state_dict(),saved['scalers']);equal(trainer._loaded_rng_state,saved['rng_state'])
        assert trainer._start_step==8001 and cfg.max_num_iterations-trainer._start_step==12999
        _restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None;equal(_capture_rng_state(),saved['rng_state'])
        pipe.train();scale=float(trainer.grad_scaler.get_scale())
        first=packet_probe(pipe,8001,scale,a.output,'initial64')
        budget=packet_probe(pipe,8002,scale,a.output,'actual_adapted_budget')
        equal(pipe.state_dict(),saved['pipeline'])
        for k,opt in trainer.optimizers.optimizers.items():equal(opt.state_dict(),saved['optimizers'][k])
        for k,s in trainer.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
        equal(trainer.grad_scaler.state_dict(),saved['scalers']);assert digest_model(model)==before
        report=dict(passed=all(x['finite_at_scale']is not None for x in[first,budget]),
            time=time.time(),config_sha256=file_sha(a.config),spec_sha256=file_sha(a.spec),source_sha256=file_sha(__file__),parent_checkpoint_sha256=file_sha(cfg.load_checkpoint),
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(0)),
            actual_native_Trainer_setup=True,model_schema_and_all_state_exact=True,Adam_scheduler_scaler_exact=True,parent_global_RNG_exact_at_native_last_restore_point=True,
            LR_override=False,parent_LR=trainer.optimizers.optimizers['fields'].param_groups[0]['lr'],parent_scaler=scale,
            all66_source_images_RGB3=True,optimizer_updates=0,training_iterations_called=0,no_eval_GT_used_in_gradient_probe=True,
            finite_at_loaded_scale=all(x['full_AMP_random']['finite']and x['full_AMP_black']['finite']for x in[first,budget]),
            finite_after_readonly_backoff=all(x['finite_at_scale']is not None for x in[first,budget]),
            production_must_keep_loaded_scaler_and_native_automatic_backoff=True,
            no_callbacks_or_grid_updates=True,field_grid_hash_unchanged=True,first_actual_packet=first,adapted_budget_packet=budget,
            total_attempt_upper_bound=21999,phase_start8001_end_exclusive21000=True)
    finally:_restore_rng_state(original_rng)
    equal(_capture_rng_state(),original_rng);report['probe_process_Python_NumPy_Torch_RNG_restored']=True
    atomic_json(a.output/'zeroopt_probe.json',report);print(json.dumps(dict(passed=report['passed'],optimizer_updates=0,output=str(a.output))),flush=True)


if __name__=='__main__':main()
