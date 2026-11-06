"""Actual native4096-ray warmup/ARM admission and all-ones identity probe; zero updates."""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from train_marshawn_semantic_admission import build_config,validate_spec
from marshawn_tuning_sampling import install_sampling
from marshawn_semantic_admission import sha
from marshawn_semantic_probe_checks import gradient_max_difference
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main(args):
    if not 0<args.memory_cap_gib<=12:raise ValueError('Probe maximum12GiB')
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
    spec=validate_spec(json.loads(args.spec.read_text()));out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=build_config(dict(spec,output=str(out/'setup')));trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();model=trainer.pipeline.model;pipe=trainer.pipeline
    sampler=install_sampling(pipe,Path(spec['data']),'actor','body80',64);params=list(model.field.parameters());before=parameter_hash(model)
    original=model.field._semantic_original_normalize;wrapped=model.field._normalize_positions;support=model.semantic_admission.clone();lookup_support=model.semantic_lookup.mask.clone()
    if not torch.equal(support,lookup_support):raise AssertionError('Device-moved density/traversal supports differ')
    global_rng=_capture_rng_state();reports=[]
    def render(rb,batch,state):
        _restore_rng_state(state)
        with torch.autocast('cuda',enabled=trainer.mixed_precision):
            outputs=model(rb);loss=sum(model.get_loss_dict(outputs,batch).values())
        # Unscaled diagnostic gradient avoids conflating loss-scale adaptation
        # with all-ones identity; actual GradScaler+Adam is checked by smoke.
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        result={k:outputs[k].detach().cpu()for k in ('rgb','accumulation')}
        if not torch.isfinite(loss) or any(g is not None and not torch.isfinite(g).all()for g in grads):raise FloatingPointError('Nonfinite native gradient')
        result['grad']=[None if g is None else g.detach().cpu()for g in grads];return result
    try:
        for step in (0,4096):
            model.train();model.current_train_step=step;sampler.set_step(step)
            model.semantic_admission.fill_(True);model.semantic_lookup.mask.fill_(True)
            for callback in trainer.callbacks:callback.run_callback_at_location(step,location=Location.BEFORE_TRAIN_ITERATION)
            rb,batch=pipe.datamanager.next_train(step);assert len(rb.origins)==4096;state=_capture_rng_state()
            model.field._normalize_positions=original;baseline=render(rb,batch,state)
            model.field._normalize_positions=wrapped;identity=render(rb,batch,state)
            identity_report={k:float((baseline[k]-identity[k]).abs().max())for k in ('rgb','accumulation')}
            for k in ('rgb','accumulation'):torch.testing.assert_close(baseline[k],identity[k],rtol=1e-5,atol=1e-6)
            errors=[]
            for x,y in zip(baseline['grad'],identity['grad']):
                if x is None or y is None:assert x is None and y is None;continue
                torch.testing.assert_close(x,y,rtol=1e-3,atol=3e-6)
                errors.append(gradient_max_difference(x,y))
            identity_report['max_parameter_gradient_abs_difference']=max(errors,default=0.)
            del baseline,identity
            model.semantic_admission.copy_(support);model.semantic_lookup.mask.copy_(lookup_support)
            # Exercise original callback update and final admission at both phases.
            for callback in trainer.callbacks:callback.run_callback_at_location(step,location=Location.BEFORE_TRAIN_ITERATION)
            assert not bool((model.occupancy_grid.binaries&~support[None]).any())
            actual=render(rb,batch,_capture_rng_state());del actual
            generator=torch.Generator(device='cuda').manual_seed(9382+step)
            box=model.field.aabb;points=box[0]+torch.rand((8192,3),generator=generator,device=box.device)*(box[1]-box[0]);admitted=model.semantic_lookup(points)
            assert bool(admitted.any())and bool((~admitted).any())
            with torch.no_grad(),torch.autocast('cuda',enabled=trainer.mixed_precision):
                sigma=model.field.density_fn(points).flatten();query_sigma,_=model.field.query_points(points,torch.ones_like(points)/np.sqrt(3))
            assert bool(torch.isfinite(sigma).all())and bool(torch.isfinite(query_sigma).all())
            assert bool((sigma[~admitted]==0).all())and bool((query_sigma.flatten()[~admitted]==0).all())
            reports.append(dict(step=step,path='fixed_uniform256'if step==0 else'adaptive_ARM',rays=4096,identity=identity_report,all_actual_outputs_gradients_finite=True,admitted_probe_points=int(admitted.sum()),forbidden_probe_points=int((~admitted).sum()),forbidden_density_fn_max=float(sigma[~admitted].abs().max()),forbidden_query_points_max=float(query_sigma.flatten()[~admitted].abs().max()),learned_core_cells=int(model.semantic_learned_core.sum()),actual_active_cells=int(model.occupancy_grid.binaries.sum()),active_subset_support=True))
        after=parameter_hash(model);assert before==after and all(p.grad is None for p in model.parameters())
        result=dict(passed=True,kind='ACTUAL_NATIVE_SEMANTIC_ADMISSION_4096_IDENTITY_AND_PATH_GATE',spec_sha256=sha(args.spec),probe_sha256=sha(__file__),semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256'],actual_density=model.field.density_activation,actual_correct_sh=model.field.correct_sh_directions,frequency_projection=pipe.config.frequency_projection_mode,paths=reports,optimizer_updates=0,parameters_unchanged_sha256=before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),quality_claim=False)
        (out/'probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        model.field._normalize_positions=wrapped;model.semantic_admission.copy_(support);model.semantic_lookup.mask.copy_(lookup_support);_restore_rng_state(global_rng);trainer.shutdown()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--memory-cap-gib',type=float,default=12.);main(p.parse_args())
