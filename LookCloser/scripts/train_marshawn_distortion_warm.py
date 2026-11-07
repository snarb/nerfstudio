"""Matched full-state .1/.3 distortion forks; original scientific loop and plateau criterion."""
import argparse,copy,fcntl,json,os,random,signal,time
from pathlib import Path
from train_marshawn_semantic_late_distortion import (sha,read,atomic,tree_digest,adam_steps,verify_state_equality,verify_cache_ids,verify_semantic_model,verify_numeric_render_parity)
from marshawn_distortion_warm_factory import SOURCE_STEP,SOURCE_SHA
from marshawn_distortion_warm_factory import ARMS


def validate_spec(s):
    expected={'output','parent_checkpoint','parent_checkpoint_sha256','parent_spec','parent_spec_sha256','anchor_audit','anchor_audit_sha256','mode','arm','release','worker_sha256','frozen_source_hashes','source_render_sha256','confidence_manifest','confidence_manifest_sha256','stopping','density_selector'}
    if set(s)!=expected or s['mode']not in ('zero','technical','production')or s['arm']not in ARMS:raise ValueError('Exact two-arm packed-distortion warm specification required')
    if s['parent_checkpoint_sha256']!=SOURCE_SHA or s['density_selector']!='original_mapped':raise ValueError('Exact raw126316 and common original mapped selector required')
    if s['stopping']!=dict(minimum_attempts=4000,anchor_step=130316,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001,early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None):raise ValueError('Original criterion with separate +4000 fork anchor required')
    if set(s['source_render_sha256'])!={c+x for c in ('DC','HC','KA')for x in ('.png','_alpha.png')}:raise ValueError('Exact source three FHD RGB/alpha pairs required')
    return s


def run(spec_path,stop_after_attempts=None):
    spec_path=Path(spec_path).resolve();s=validate_spec(read(spec_path));release=read(s['release']);auth='production_authorized'if s['mode']=='production'else'technical_authorized'
    if release.get('reviewer')!='root'or release.get(auth)is not True or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Exact root phase release required')
    for p,d in s['frozen_source_hashes'].items():
        if sha(p)!=d:raise ValueError('Frozen scientific/helper source changed: '+p)
    for k in ('parent_checkpoint','parent_spec','anchor_audit','confidence_manifest'):
        if sha(s[k])!=s[k+'_sha256']:raise ValueError('Bound source changed: '+k)
    if sha(__file__)!=s['worker_sha256']:raise ValueError('Fork worker changed')
    from marshawn_gpu_isolation import require_controller_lease,verify_cuda
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    isolation=verify_cuda(lease['expected_physical_uuid'],torch)
    allocator_total_bytes=int(torch.cuda.get_device_properties(0).total_memory)
    allocator_limit_bytes=10*2**30
    allocator_fraction=allocator_limit_bytes/allocator_total_bytes
    torch.cuda.set_per_process_memory_fraction(allocator_fraction,0)
    isolation.update(torch_allocator_limit_bytes=allocator_limit_bytes,torch_allocator_fraction=allocator_fraction,
                     torch_allocator_total_bytes=allocator_total_bytes,torch_allocator_cap_call_succeeded=True)
    import train_marshawn_semantic_admission as base
    from marshawn_distortion_warm_factory import build_config,validate_anchor
    from marshawn_distortion_warm_rgb import ConfidencePolicy
    # Preserve the original worker's class-import order before frozen seed42
    # cache reconstruction, then restore checkpoint RNG after setup as usual.
    from marshawn_semantic_admission import SemanticAdmissionModel
    from nerfstudio.models.lookcloser import LookCloserModel
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    from marshawn_tuning_sampling import install_sampling
    from marshawn_tuning_metrics import PlateauTracker
    from marshawn_semantic_occupancy import install_capture_hook,export_snapshot
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    from nerfstudio.utils import writer
    import inspect
    bound={Path(p).resolve()for p in s['frozen_source_hashes']}
    for obj in (base,build_config,ConfidencePolicy,SemanticAdmissionModel,LookCloserModel,LookCloserPipeline,install_sampling,PlateauTracker,install_capture_hook,_capture_rng_state,require_controller_lease):
        if Path(inspect.getfile(obj)).resolve()not in bound:raise ValueError('Actual runtime import escaped frozen source')
    parent=read(s['parent_spec']);anchor=read(s['anchor_audit']);validate_anchor(parent,anchor)
    for p,d in anchor['source_hashes'].items():
        if sha(p)!=d:raise ValueError('Anchor evidence changed: '+p)
    out=Path(s['output']).resolve();out.mkdir(parents=True,exist_ok=True)
    if out==Path(parent['output']).resolve()or out in Path(s['parent_checkpoint']).resolve().parents:raise ValueError('Source output cannot be reused')
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(out/'physical_gpu_isolation.json',dict(**isolation,lease=lease))
    if stop_after_attempts is not None and(s['mode']!='technical'or stop_after_attempts!=32):raise ValueError('Only declared technical first32 interruption allowed')
    latest=out/'latest.json';checkpoint=Path(s['parent_checkpoint']);child=latest.exists()
    if child:
        receipt=read(latest);checkpoint=out/receipt['checkpoint']
        if sha(checkpoint)!=receipt['sha256']:raise ValueError('Child checkpoint changed')
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);lineage=loaded.get('distortion_fork_runtime')
    if child:
        if not lineage or lineage['spec_sha256']!=sha(spec_path)or lineage['source_sha256']!=SOURCE_SHA:raise ValueError('Strict child source/arm/phase mismatch')
    elif loaded['step']!=SOURCE_STEP:raise ValueError('Wrong immutable source step')
    if not child:
        for k,d in anchor['fullstate_hashes'].items():
            if tree_digest(loaded[k])!=d:raise ValueError('Source serialized seven-state mismatch: '+k)
    for name,d in loaded['tuning_runtime']['input_sha256'].items():
        p=Path(parent['semantic_admission']['manifest']).parent/name[9:]if name.startswith('semantic/')else Path(parent['semantic_admission']['manifest'])if name=='semantic_manifest'else Path(parent['data'])/name
        if sha(p)!=d:raise ValueError('Original source input changed: '+name)
    for name,d in loaded['tuning_runtime']['code_sha256'].items():
        matches=[p for p in bound if p.name==name]
        if len(matches)!=1 or sha(matches[0])!=d:raise ValueError('Original runtime code changed: '+name)
    ready=base.validate_data(Path(parent['data']))
    random.seed(parent['seed']);np.random.seed(parent['seed']);torch.manual_seed(parent['seed']);torch.cuda.manual_seed_all(parent['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(False)
    cfg=build_config(parent,anchor,checkpoint,out);cfg.save_config();trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();pipe=trainer.pipeline;m=pipe.model;m.current_train_step=loaded['step']
    if len(pipe.datamanager.train_dataset)!=63 or len(pipe.datamanager.eval_dataset)!=3:raise ValueError('TRAIN/EVAL membership changed')
    if m.field.feature_reweighting_strength!=.3 or not pipe._feature_reweighting_switch_applied or m.config.distortion_loss_mult!=.1:raise ValueError('Actual source FR/distortion runtime differs')
    sampler=install_sampling(pipe,Path(parent['data']),parent['objective'],parent['sampling'],parent['width']);sampler.load_state_dict(loaded['tuning_runtime']['sampling_state']);install_capture_hook(m)
    m._tuning_occupancy_capture={k:v.to(m.device)if torch.is_tensor(v)else v for k,v in loaded['actual_occupancy_capture'].items()}
    cache=pipe.datamanager.train_image_dataloader.cached_collated_batch;ids=cache['image_idx'].cpu().tolist();cache_hash=verify_cache_ids(ids,parent['seed']);cache_sources=[]
    for position,camera in enumerate(ids):
        row=pipe.datamanager.train_dataset[camera]
        for key in ('image','mask'):
            if key in cache and not torch.equal(cache[key][position].cpu(),row[key].cpu()):raise ValueError('Source complete-cache association differs')
        cache_sources.append(dict(camera=camera,image=tree_digest(cache['image'][position]),mask=tree_digest(cache['mask'][position])if'mask'in cache else None))
    pipe.datamanager.train_count=int(lineage['cache_state']['train_count']if child else sampler.state_dict()['sample_count'])
    if trainer._loaded_rng_state is None:raise ValueError('Source global RNG absent')
    _restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None;runtime=copy.deepcopy(loaded['tuning_runtime'])
    def actual_state():
        r=copy.deepcopy(runtime);r['sampling_state']=sampler.state_dict()
        return dict(pipeline=pipe.state_dict(),optimizers={k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict()for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),tuning_runtime=r,actual_occupancy_capture=m._tuning_occupancy_capture)
    # Parent/child full scientific baseline is verified BEFORE confidence changes.
    baseline=verify_state_equality(loaded,actual_state());baseline['semantic_support']=verify_semantic_model(m,loaded,anchor)
    if trainer._start_step!=loaded['step']+1:raise ValueError('Absolute resume step differs')
    if child:
        if lineage['cache_state']['ids']!=ids or lineage['cache_state']['sources']!=cache_sources:raise ValueError('Child cache reconstruction differs')
        lut=lineage['model_cache'];m.field._feature_weight_lut=lut['feature_weight_LUT'].to(m.device);m.field._feature_weight_lut_strength=lut['strength']
        if lut['FR']!=m.field.feature_reweighting_strength or lut['switch_applied']!=pipe._feature_reweighting_switch_applied:raise ValueError('Child nonpersistent FR cache differs')
    else:
        with torch.inference_mode():m.field._get_discrete_feature_weights(torch.arange(m.field.num_levels,device=m.device,dtype=m.freq_grid.grid.dtype).unsqueeze(-1))
    def model_cache():return dict(feature_weight_LUT=m.field._feature_weight_lut.detach().clone(),strength=m.field._feature_weight_lut_strength,FR=m.field.feature_reweighting_strength,switch_applied=pipe._feature_reweighting_switch_applied)
    before_policy=tree_digest(_capture_rng_state());confidence=ConfidencePolicy(sampler,s['confidence_manifest'],s['confidence_manifest_sha256'],'original')
    if child:confidence.load_state_dict(lineage['confidence_state'])
    if tree_digest(_capture_rng_state())!=before_policy:raise ValueError('Confidence installation consumed global RNG')
    verify_state_equality(loaded,actual_state())
    baseline.update(cache_ids=ids,cache_sha256=cache_hash,cache_sources=cache_sources,historical_cache_order_observed=False,cache_evidence='Frozen seed42 full-cache reconstruction, exact RGB/mask association; child IDs/sources explicitly checkpointed',confidence_state_sha256=tree_digest(confidence.state_dict()),model_cache_sha256=tree_digest(model_cache()),source_verified_before_confidence=True,FR=.3,distortion=.1,mode=s['mode'],Adam=adam_steps(loaded['optimizers']),next_step=trainer._start_step)
    effective_distortion=ARMS[s['arm']]
    if child and lineage.get('effective_distortion')!=effective_distortion:raise ValueError('Child distortion scalar changed')
    baseline.update(source_verified_before_loss_scalar=True,source_resume_distortion=.1,effective_distortion=effective_distortion)
    # The actual .1 source/seven-state baseline above precedes the sole treatment.
    m.config.distortion_loss_mult=effective_distortion
    cfg.save_config()
    atomic(out/f"resume_proof_{loaded['step']}.json",baseline)
    if s['mode']=='zero':
        cache_before=tree_digest(model_cache());confidence_before=tree_digest(confidence.state_dict());metrics=base.evaluate(trainer,ready,Path(parent['data']),out,loaded['step']);verify_state_equality(loaded,actual_state())
        if tree_digest(model_cache())!=cache_before or tree_digest(confidence.state_dict())!=confidence_before:raise ValueError('Zero evaluation changed primed model cache/confidence state')
        parity=verify_numeric_render_parity(anchor['render_directory'],out/'renders'/f"step-{loaded['step']:09d}",s['source_render_sha256'],anchor['source_eval'],metrics)
        atomic(out/'complete.json',dict(status='zero_update_state_equal',passed=True,proof=baseline,render_parity=parity,confidence_state=confidence.state_dict(),effective_distortion=effective_distortion,render_state_unchanged=True));trainer.shutdown();return
    tracker=PlateauTracker.from_dict(lineage['plateau'])if child else PlateauTracker(anchor_step=130316)
    evaluations=copy.deepcopy(lineage['evaluations'])if child else[anchor['source_eval']];checks=copy.deepcopy(lineage['gradient_checks'])if child else[];control_checks=copy.deepcopy(lineage.get('source_control_checks',[]))if child else[]
    if s['mode']=='technical':
        native_loss=m.get_loss_dict;checked=[False]
        def source_control_loss(outputs,batch,metrics_dict=None):
            losses=native_loss(outputs,batch,metrics_dict)
            if not checked[0]:
                from marshawn_tuning_sampling import weighted_charbonnier
                index=batch['indices'].detach().cpu().numpy();weights=np.array([sampler.rows[c]['W64'][y,x]for c,y,x in index],np.float32);np.testing.assert_array_equal(weights,batch['actor_confidence'].cpu().numpy())
                reference=weighted_charbonnier(outputs['rgb'],batch['image'],torch.from_numpy(weights),confidence.normalizer(sampler.step,len(index)));torch.testing.assert_close(losses['rgb_loss'],reference,rtol=1e-12,atol=1e-12)
                actual_grad=torch.autograd.grad(losses['rgb_loss'],outputs['rgb'],retain_graph=True)[0];reference_grad=torch.autograd.grad(reference,outputs['rgb'],retain_graph=True)[0];torch.testing.assert_close(actual_grad,reference_grad,rtol=1e-6,atol=1e-10)
                if 'packed_weights'not in outputs:raise ValueError('Actual native packed distortion required')
                weights=outputs['packed_weights']
                raw_distortion=m._packed_distortion_loss(outputs['packed_spacing_starts'],outputs['packed_spacing_ends'],weights,outputs['packed_ray_indices'],outputs['rgb'].shape[0]).mean()
                reference_distortion=.1*raw_distortion;expected_distortion=(effective_distortion/.1)*reference_distortion
                torch.testing.assert_close(losses['distortion_loss'],expected_distortion,rtol=1e-5,atol=1e-10)
                actual_distortion_grad=torch.autograd.grad(losses['distortion_loss'],weights,retain_graph=True)[0]
                reference_distortion_grad=torch.autograd.grad(reference_distortion,weights,retain_graph=True)[0]*(effective_distortion/.1)
                torch.testing.assert_close(actual_distortion_grad,reference_distortion_grad,rtol=1e-5,atol=1e-8)
                control_checks.append(dict(step=sampler.step,source_W64_exact=True,packed_weight_dtype=str(weights.dtype),packed_spacing_dtype=str(outputs['packed_spacing_starts'].dtype),packed_distortion_reference_mult=.1,expected_distortion_factor=effective_distortion/.1,packed_distortion_value_error=float((losses['distortion_loss']-expected_distortion).detach().abs()),packed_distortion_gradient_max_error=float((actual_distortion_grad-reference_distortion_grad).detach().abs().max()),source_main_batch_sha256=tree_digest(batch['indices']),source_original_RGB_gradient_max_error=float((actual_grad-reference_grad).abs().max()),actual_distortion=m.config.distortion_loss_mult,actual_FR=m.field.feature_reweighting_strength,extra_field_queries=0,extra_optimizer_updates=0));checked[0]=True
            return losses
        m.get_loss_dict=source_control_loss
    if child and tracker.stopped:raise ValueError('Completed plateau child cannot silently rerun')
    target=SOURCE_STEP+64 if s['mode']=='technical'else None;start_step=trainer._start_step;del loaded
    stop=[];signal.signal(signal.SIGTERM,lambda signum,frame:stop.append(signum));signal.signal(signal.SIGINT,lambda signum,frame:stop.append(signum));ckpts=out/'checkpoints';ckpts.mkdir(exist_ok=True);started=time.monotonic();last_report=0.;attempts=0;ray_hash=__import__('hashlib').sha256();first_ray=None;packets=[];original_next=pipe.datamanager.next_train
    def observed_next(step):
        nonlocal first_ray
        ray,batch=original_next(step);raw=batch['indices'].detach().cpu().contiguous().numpy().tobytes();digest=__import__('hashlib').sha256(raw).hexdigest();first_ray=first_ray or digest;ray_hash.update(raw);expected=confidence.normalizer(step,len(batch['indices']));error=abs(batch['actor_proposal_mean_weight']-expected)
        if error>1e-12:raise ValueError('Actual RGB normalizer differs from exact confidence expectation')
        if s['mode']=='technical':packets.append(dict(step=step,indices_sha256=digest,normalizer=batch['actor_proposal_mean_weight'],normalizer_error=error,confidence_sha256=tree_digest(batch['actor_confidence'])))
        return ray,batch
    pipe.datamanager.next_train=observed_next
    def segment(step):return dict(first_step=start_step,last_step=step,attempts=attempts,first_indices_sha256=first_ray,indices_sha256=ray_hash.hexdigest(),packets=packets)
    def cache_state():return dict(ids=ids,sources=cache_sources,train_count=pipe.datamanager.train_count)
    def checkpoint_path(step):return str(Path(s['parent_checkpoint']))if step==SOURCE_STEP else str(ckpts/f'step-{step:09d}.ckpt')
    def save(step):
        payload=dict(step=step,**actual_state(),distortion_fork_runtime=dict(effective_distortion=effective_distortion,spec_sha256=sha(spec_path),source_sha256=SOURCE_SHA,confidence_state=confidence.state_dict(),model_cache=model_cache(),cache_state=cache_state(),plateau=tracker.to_dict(),evaluations=evaluations,gradient_checks=checks,source_control_checks=control_checks,sampling_segment=segment(step)))
        path=ckpts/f'step-{step:09d}.ckpt';temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path);export_snapshot(m,path,step,ckpts,sources=anchor['source_hashes']);atomic(latest,dict(step=step,checkpoint=str(path.relative_to(out)),sha256=sha(path)))
        selected=base.choose_checkpoint(evaluations);raw=min(evaluations,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr'],r['step']));atomic(out/'retention.json',dict(parent=checkpoint_path(SOURCE_STEP),selected=checkpoint_path(selected['step']),selected_step=selected['step'],raw_min=checkpoint_path(raw['step']),raw_min_step=raw['step'],latest=str(path),early_visual_steps=[SOURCE_STEP+n for n in (500,1000,2000,4000)],pruning='none; preserve all evaluated/atomic resume states in this matched campaign'))
    step=start_step
    while True:
        trainer.step=step;pipe.train();sampler.set_step(step);m.config.distortion_loss_mult=effective_distortion;trainer._apply_live_optimizer_switch(step)
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=Location.BEFORE_TRAIN_ITERATION)
        if m.config.distortion_loss_mult!=effective_distortion or m.field.feature_reweighting_strength!=.3 or m.config.feature_reweighting_strength!=.3:raise ValueError('Actual live source loss scalars changed before train_iteration')
        before=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()});rng_before=tree_digest(_capture_rng_state())if s['mode']=='technical'else None
        loss,_,_=trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()):raise FloatingPointError('Nonfinite matched confidence objective')
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=Location.AFTER_TRAIN_ITERATION)
        writer.write_out_storage();attempts+=1
        if s['mode']=='technical':
            after=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()});grads=[p.grad for p in m.parameters()if p.grad is not None];finite=bool(grads)and all(bool(torch.isfinite(g).all())for g in grads);updated=any(min(after[k])>min(v)for k,v in before.items())
            if updated and not finite:raise FloatingPointError('Adam update with nonfinite gradients')
            checks.append(dict(step=step,Adam_updated=updated,combined_gradients_finite=finite,RNG_before=rng_before,RNG_after=tree_digest(_capture_rng_state()),actual_distortion=m.config.distortion_loss_mult,actual_FR=m.field.feature_reweighting_strength,LR={k:[g['lr']for g in v.param_groups]for k,v in trainer.optimizers.optimizers.items()}))
        delta=step-SOURCE_STEP;early=s['mode']=='production'and delta in (500,1000,2000,4000);plateau_gate=s['mode']=='production'and step>=tracker.anchor_step and(step-tracker.anchor_step)%tracker.interval==0
        if early or plateau_gate:
            metrics=base.evaluate(trainer,ready,Path(parent['data']),out,step);evaluations.append(metrics)
            if plateau_gate:tracker.observe(step,metrics['metrics']['validation_loss'],metrics['metrics']['lpips'])
            atomic(out/'metrics.json',evaluations);atomic(out/'visual_gates.json',[dict(step=r['step'],attempts=r['step']-SOURCE_STEP,renders=str(out/'renders'/f"step-{r['step']:09d}"),used_for_plateau=r['step']>=tracker.anchor_step,visual_review_complete=False)for r in evaluations if r['step']!=SOURCE_STEP])
        partial=stop_after_attempts is not None and attempts>=stop_after_attempts;terminal=(target is not None and step==target)or(s['mode']=='production'and tracker.stopped)
        if early or plateau_gate or partial or terminal or stop or delta%2000==0:save(step)
        if early or plateau_gate or partial or terminal or time.monotonic()-last_report>=30:
            atomic(out/'progress.json',dict(step=step,attempts=delta,elapsed_seconds=time.monotonic()-started,plateau=tracker.to_dict(),mode=s['mode'],confidence_state_sha256=tree_digest(confidence.state_dict()),sampling_segment=segment(step),Adam=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()})));last_report=time.monotonic()
        if terminal:
            final_adam=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()})
            if not all(min(v)>min(baseline['Adam'][k])for k,v in final_adam.items()):raise ValueError('No actual resumed Adam update in this invocation')
            atomic(out/'complete.json',dict(status='technical_resume_complete'if s['mode']=='technical'else'plateau',passed=True,final_step=step,attempts=delta,plateau=tracker.to_dict(),evaluations=evaluations,gradient_checks=checks,source_control_checks=control_checks,sampling_segment=segment(step),actual_Adam=final_adam,confidence_state=confidence.state_dict(),source_sha256=SOURCE_SHA,effective_distortion=effective_distortion,visual_review_complete=False,quality_promotion=False));trainer.shutdown();return
        if partial:trainer.shutdown();return
        if stop:
            atomic(out/'interrupted.json',dict(step=step,signals=stop,checkpoint_committed=True,quality_complete=False));trainer.shutdown();raise SystemExit(75)
        step+=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--stop-after-attempts',type=int);a=p.parse_args();run(a.spec,a.stop_after_attempts)
