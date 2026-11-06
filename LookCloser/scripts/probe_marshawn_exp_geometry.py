"""Full4096 native SH/exp(h+1) geometry AMP probe; zero optimizer updates."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from probe_marshawn_prefix_candidate import digest_state
from marshawn_softplus_prefix_candidate import sha256
from train_marshawn_exp_geometry import build_config,validate_spec
from marshawn_tuning_sampling import install_sampling
from marshawn_exp_geometry import install_exp_geometry
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state


def main(args):
    total=torch.cuda.get_device_properties(0).total_memory
    if not 0<args.memory_cap_gib<=12:raise ValueError('Probe cap must be <=12GiB')
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*2**30/total)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    spec=validate_spec(json.loads(args.spec.read_text()));out=args.output;out.mkdir(exist_ok=False,parents=True)
    cfg=build_config(dict(spec,output=str(out/'setup')));trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    pipe,model=trainer.pipeline,trainer.pipeline.model;model._postprocess_occupancy_grid(step=0);model.train()
    ready=json.loads((Path(spec['data'])/'ready.json').read_text());sampler=install_sampling(pipe,Path(spec['data']),'full','body80',spec['width'])
    adapter=install_exp_geometry(pipe,sampler,spec,ready);obj=adapter.objective;sampler.set_step(0);model.current_train_step=0
    before=digest_state(model);params=list(model.field.encoding.parameters())+list(model.field.mlp_geo.parameters())
    state=_capture_rng_state();sampler_state=sampler.state_dict()
    try:
        prefix_packet=obj.sample(0);band_packet=obj.sample_band(0);prefix_again=obj.sample(0)
        assert all(torch.equal(prefix_packet[k],prefix_again[k])for k in prefix_packet)
        assert torch.equal(prefix_packet['indices'],band_packet['indices'])
        rb,batch=pipe.datamanager.next_train(0)
        assert len(rb.origins)==4096
        with torch.autocast(device_type='cuda',enabled=trainer.mixed_precision):
            outputs=model(rb);losses=model.get_loss_dict(outputs,batch);rgb=losses['rgb_loss'];base=sum(losses.values())
            obj.weight=1.;prefix,diag=obj(model.field,0);obj.weight=.003
            band,bdiag=obj.band(model.field,0,weight=1.)
        scale=float(trainer.grad_scaler.get_scale());initial=scale;attempts=[]
        # Probe actual candidate-weighted band gradients, so an unweighted
        # diagnostic does not create overflow absent from the real objective.
        tensors=[rgb,base,.01*prefix,.003*band]
        while True:
            grads=[torch.autograd.grad(v*scale,params,retain_graph=True,allow_unused=True)for v in tensors]
            finite=all(g is None or bool(torch.isfinite(g).all())for group in grads for g in group)
            attempts.append(dict(scale=scale,finite=finite))
            if finite or scale<=1024:break
            del grads;scale/=2
        if not finite:raise FloatingPointError('Native weighted AMP gradients remain nonfinite at1024')
        def norm(group):return float(sum(g.detach().double().square().sum()for g in group if g is not None).sqrt()/scale)
        rgbnorm,basenorm,prefix01norm,band003norm=map(norm,grads)
        prefixnorm=.3*prefix01norm
        scanned=[]
        for prefix_weight in [.001,.003,.01]:
            for band_weight in [0.,.0003,.001,.003]:
                combined=base+prefix_weight*prefix+band_weight*band
                actual=torch.autograd.grad(combined*scale,params,retain_graph=True,allow_unused=True)
                finite=all(g is None or bool(torch.isfinite(g).all())for g in actual)
                scanned.append(dict(prefix_weight=prefix_weight,band_weight=band_weight,
                    prefix_to_rgb_geometry_norm=(prefix_weight/.01)*prefix01norm/max(rgbnorm,1e-30),
                    band_to_rgb_geometry_norm=(band_weight/.003)*band003norm/max(rgbnorm,1e-30),
                    combined_to_rgb_geometry_norm=norm(actual)/max(rgbnorm,1e-30),actual_combined_gradients_finite=finite))
                if not finite:raise FloatingPointError('Actual combined native-exp geometry gradient nonfinite')
        after=digest_state(model)
        assert before==after and all(p.grad is None for p in model.parameters())
        report=dict(kind='SH_NATIVE_EXP_HPLUS1_GEOMETRY_4096_AMP_ZERO_UPDATE',passed=True,
            spec_sha256=sha256(args.spec),manifest_sha256=spec['prefix_candidate']['manifest_sha256'],
            probe_sha256=sha256(__file__),helper_sha256=sha256(Path(__file__).with_name('marshawn_exp_geometry.py')),
            actual_correct_sh_directions=model.field.correct_sh_directions,config_correct_sh_directions=model.config.correct_sh_directions,
            density_activation=model.field.density_activation,actual_density_formula='native trunc_exp(h+1)',frequency_projection_mode=pipe.config.frequency_projection_mode,actual_rgb_rays=4096,prior_rays=128,prefix_samples=64,band_samples=16,
            gradient_parameters='hash encoding + geometry MLP; color head excluded',
            rgb_geometry_norm=rgbnorm,base_geometry_norm=basenorm,prefix003_geometry_norm=prefixnorm,
            prefix_to_rgb_geometry_norm=prefixnorm/max(rgbnorm,1e-30),band003_geometry_norm=band003norm,
            scanned=scanned,scale_attempts=attempts,production_initial_scale=initial,
            prefix_tau_mean=float(diag['prefix_tau_mean']),band_tau_mean=float(bdiag['band_tau_mean']),
            model_before_after_sha256=before,parameter_grads_unset=True,optimizer_updates=0,
            prefix_samples_unchanged_by_band=True,production_modified=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            memory_cap_gib=args.memory_cap_gib,quality_claim=False)
        (out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        sampler.load_state_dict(sampler_state);_restore_rng_state(state);trainer.shutdown()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--memory-cap-gib',type=float,default=12.);main(p.parse_args())
