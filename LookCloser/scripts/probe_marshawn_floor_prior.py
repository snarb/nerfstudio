"""Actual random10k full-state/floor-prior gradient proof; zero optimizer steps."""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:sys.path.insert(0,str(d))
import numpy as np
import torch
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.scripts.train import _set_random_seed
from probe_marshawn_occupancy import digest_model
from staged_native_hash import file_sha
from run_marshawn_floor_prior_resume import load,validate,atomic_json


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


def gradient(term,named,scale):
    values=torch.autograd.grad(term*scale,[p for _,p in named],retain_graph=True,allow_unused=True)
    finite=all(bool(torch.isfinite(g).all())for g in values if g is not None);groups={}
    if finite:
        for (name,_),g in zip(named,values):
            if g is None:continue
            group='geometry'if'mlp_base'in name else'color'if'mlp_head'in name else'other'
            groups[group]=groups.get(group,0.)+float((g.detach().double()/scale).square().sum())
    return dict(finite=finite,scale=scale,l2=sum(groups.values())**.5 if finite else None,
                group_l2={k:v**.5 for k,v in groups.items()})


def packet(pipe,step,scale,output,label,weights):
    rng=_capture_rng_state();prior=pipe._floor_band_scene_prior
    a=prior.sample_batch(step);b=prior.sample_batch(step)
    assert torch.equal(a['points'],b['points']);equal(_capture_rng_state(),rng)
    with torch.autocast(device_type='cuda',enabled=True):out,losses,metrics=pipe.get_train_loss_dict(step)
    assert set(losses)=={'rgb_loss','observed_sfm_free_space','observed_sfm_surface_band','packed_distortion','finite_floor_scene_prior'}
    raw=pipe.model.field._density_before_activation.detach().float()
    unit=losses['finite_floor_scene_prior']/prior.weight
    baseline=sum(v for k,v in losses.items()if k!='finite_floor_scene_prior')
    named=[(n,p)for n,p in pipe.model.named_parameters()if p.requires_grad]
    base=gradient(baseline,named,scale);rgb=gradient(losses['rgb_loss'],named,scale);rows=[]
    for weight in weights:
        term=losses['finite_floor_scene_prior'] if weight==prior.weight else weight*unit
        extra=gradient(term,named,scale);combined=gradient(baseline+term,named,scale)
        rows.append(dict(weight=weight,prior_gradient=extra,combined_gradient=combined,
            prior_norm_over_baseline=extra['l2']/base['l2']if extra['finite']and base['finite']and base['l2']else None,
            prior_norm_over_RGB=extra['l2']/rgb['l2']if extra['finite']and rgb['finite']and rgb['l2']else None,
            prior_geometry_over_base_geometry=extra['group_l2'].get('geometry',0)/base['group_l2']['geometry']if extra['finite']and base['finite']and base['group_l2'].get('geometry')else None))
    alpha=-torch.expm1(-torch.exp(torch.logsumexp(raw.reshape(64,32)+np.log(float(pipe.model.field.average_init_density)),1)-np.log(32)+np.log(.04)))
    np.savez_compressed(output/(label+'_prior_points.npz'),points=a['points'].cpu().numpy(),centers=a['centers'].cpu().numpy(),triangle_indices=a['triangle_indices'].cpu().numpy(),raw_logits=raw.cpu().numpy(),normal_band_opacity=alpha.cpu().numpy())
    return dict(actual_rays=len(out['rgb']),packed_samples=int(metrics['num_samples_per_batch']),
        baseline_gradient=base,RGB_gradient=rgb,weights=rows,raw_logit_min=float(raw.min()),raw_logit_max=float(raw.max()),
        prior_opacity_mean=float(alpha.mean()),prior_opacity_p10=float(torch.quantile(alpha,.1)),prior_opacity_min=float(alpha.min()),
        prior_unweighted_NLL=float(unit.detach()),stateless_floor_points_repeat_exact=True,floor_sampling_global_RNG_unchanged=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=float,nargs='+',default=[.0001,.0003]);args=p.parse_args();args.output.mkdir(exist_ok=False);torch.set_num_threads(2)
    cfg,spec=load(args.config,args.spec);saved=torch.load(cfg.load_checkpoint,map_location='cpu',weights_only=False);original=_capture_rng_state();assert saved['step']==10000
    cfg=copy.deepcopy(cfg);cfg.output_dir=args.output/'setup_outputs'
    try:
        _set_random_seed(42);trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();validate(trainer,spec)
        pipe=trainer.pipeline;before=digest_model(pipe.model);equal(pipe.state_dict(),saved['pipeline'])
        for k,opt in trainer.optimizers.optimizers.items():equal(opt.state_dict(),saved['optimizers'][k])
        for k,s in trainer.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
        equal(trainer.grad_scaler.state_dict(),saved['scalers']);equal(trainer._loaded_rng_state,saved['rng_state'])
        assert trainer._start_step==10001 and cfg.max_num_iterations-trainer._start_step==10999
        assert pipe.dynamic_num_rays_per_batch==64 and not isinstance(pipe._floor_band_scene_prior,torch.nn.Module)
        _restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None;equal(_capture_rng_state(),saved['rng_state'])
        pipe.train();scale=float(trainer.grad_scaler.get_scale());assert scale==65536
        first=packet(pipe,10001,scale,args.output,'initial64',args.weights);budget=packet(pipe,10002,scale,args.output,'adapted_budget',args.weights)
        equal(pipe.state_dict(),saved['pipeline'])
        for k,opt in trainer.optimizers.optimizers.items():equal(opt.state_dict(),saved['optimizers'][k])
        for k,s in trainer.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
        equal(trainer.grad_scaler.state_dict(),saved['scalers']);assert digest_model(pipe.model)==before
        statuses=[row['combined_gradient']['finite']for x in[first,budget]for row in x['weights']]
        selected_statuses=[row['combined_gradient']['finite'] and row['prior_gradient']['finite'] for x in[first,budget]for row in x['weights']if row['weight']==pipe._floor_band_scene_prior.weight]
        assert len(selected_statuses)==2
        report=dict(passed=all(selected_statuses),all_diagnostic_weights_finite=all(statuses),selected_production_weight=pipe._floor_band_scene_prior.weight,selected_term_is_exact_production_loss=True,time=time.time(),config_sha256=file_sha(args.config),spec_sha256=file_sha(args.spec),source_sha256=file_sha(__file__),parent_checkpoint_sha256=file_sha(cfg.load_checkpoint),
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(0)),
            actual_native_Trainer_setup=True,full_pipeline_keys_values_exact=True,Adam_scheduler_scaler_exact=True,parent_global_RNG_exact_before_probe=True,
            no_new_NN_params_or_buffers=True,plain_stateless_external_prior=True,LR_override=False,parent_LR=trainer.optimizers.optimizers['fields'].param_groups[0]['lr'],parent_scaler=scale,
            optimizer_updates=0,training_iterations_called=0,field_and_EMA_grid_unchanged=True,no_negative_constraints_or_carving=True,no_eval_RGB_used=True,
            first_actual_packet=first,adapted_budget_packet=budget,prior_receipt=pipe._floor_band_scene_prior.receipt,
            total_attempt_upper_bound=21999,prior_is_scene_assumption_not_depth_GT=True)
    finally:_restore_rng_state(original)
    equal(_capture_rng_state(),original);report['process_Python_NumPy_Torch_RNG_restored']=True
    atomic_json(args.output/'floor_prior_probe.json',report);print(json.dumps(dict(passed=report['passed'],optimizer_updates=0,output=str(args.output))),flush=True)


if __name__=='__main__':main()
