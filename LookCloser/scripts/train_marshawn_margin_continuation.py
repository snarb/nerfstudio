"""Matched full-state margin forks; original scientific loop and plateau criterion."""
import argparse,copy,fcntl,json,os,random,signal,time
from pathlib import Path
from train_marshawn_semantic_late_distortion import (sha,read,atomic,tree_digest,adam_steps,verify_state_equality,verify_cache_ids,verify_semantic_model,verify_numeric_render_parity)
from marshawn_margin_continuation_factory import SOURCE_STEP,SOURCE_SHA
from marshawn_margin_continuation_policy import ARMS


def validate_spec(s):
    expected={'output','parent_checkpoint','parent_checkpoint_sha256','parent_spec','parent_spec_sha256','anchor_audit','anchor_audit_sha256','mode','arm','release','worker_sha256','frozen_source_hashes','source_render_sha256','margin_manifest','margin_manifest_sha256','stopping','density_selector','zero_reference','zero_reference_sha256'}
    if set(s)!=expected or s['mode']not in ('zero','technical','production')or s['arm']not in ARMS:raise ValueError('Exact paired margin fork specification required')
    if s['parent_checkpoint_sha256']!=SOURCE_SHA or s['density_selector']!=s['arm']:raise ValueError('Exact raw126316 and declared paired selector required')
    if s['stopping']!=dict(minimum_attempts=4000,anchor_step=130316,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001,early_visual_attempts=[500,1000,2000,4000],hard_iteration_limit=None):raise ValueError('Original criterion with separate +4000 fork anchor required')
    if set(s['source_render_sha256'])!={c+x for c in ('DC','HC','KA')for x in ('.png','_alpha.png')}:raise ValueError('Exact source three FHD RGB/alpha pairs required')
    return s


def run(spec_path,stop_after_attempts=None):
    spec_path=Path(spec_path).resolve();s=validate_spec(read(spec_path));release=read(s['release']);auth='production_authorized'if s['mode']=='production'else'technical_authorized'
    if release.get('reviewer')!='root'or release.get(auth)is not True or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Exact root phase release required')
    if release.get('margin_manifest_sha256')!=s['margin_manifest_sha256'] or release.get('zero_reference_sha256')!=s['zero_reference_sha256']:raise ValueError('Exact source margin and branch-zero root release required')
    for p,d in s['frozen_source_hashes'].items():
        if sha(p)!=d:raise ValueError('Frozen scientific/helper source changed: '+p)
    for k in ('parent_checkpoint','parent_spec','anchor_audit','margin_manifest','zero_reference'):
        if sha(s[k])!=s[k+'_sha256']:raise ValueError('Bound source changed: '+k)
    if sha(__file__)!=s['worker_sha256']:raise ValueError('Fork worker changed')
    from marshawn_gpu_isolation import require_controller_lease,verify_cuda
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    isolation=verify_cuda(lease['expected_physical_uuid'],torch)
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    isolation['allocator_cap_GiB']=10
    import train_marshawn_semantic_admission as base
    from marshawn_margin_continuation_factory import build_config,validate_anchor
    from marshawn_margin_continuation_policy import MarginPolicy
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
    for obj in (base,build_config,MarginPolicy,SemanticAdmissionModel,LookCloserModel,LookCloserPipeline,install_sampling,PlateauTracker,install_capture_hook,_capture_rng_state,require_controller_lease):
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
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);lineage=loaded.get('margin_fork_runtime')
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
    # Parent/child full scientific baseline is verified BEFORE selector installation.
    baseline=verify_state_equality(loaded,actual_state());baseline['semantic_support']=verify_semantic_model(m,loaded,anchor)
    if trainer._start_step!=loaded['step']+1:raise ValueError('Absolute resume step differs')
    if child:
        if lineage['cache_state']['ids']!=ids or lineage['cache_state']['sources']!=cache_sources:raise ValueError('Child cache reconstruction differs')
        lut=lineage['model_cache'];m.field._feature_weight_lut=lut['feature_weight_LUT'].to(m.device);m.field._feature_weight_lut_strength=lut['strength']
        if lut['FR']!=m.field.feature_reweighting_strength or lut['switch_applied']!=pipe._feature_reweighting_switch_applied:raise ValueError('Child nonpersistent FR cache differs')
    else:
        with torch.inference_mode():m.field._get_discrete_feature_weights(torch.arange(m.field.num_levels,device=m.device,dtype=m.freq_grid.grid.dtype).unsqueeze(-1))
    def model_cache():return dict(feature_weight_LUT=m.field._feature_weight_lut.detach().clone(),strength=m.field._feature_weight_lut_strength,FR=m.field.feature_reweighting_strength,switch_applied=pipe._feature_reweighting_switch_applied)
    if not child:
        # Both branches must reproduce original A before any additional gate.
        cache_before=tree_digest(model_cache())
        source_metrics=base.evaluate(trainer,ready,Path(parent['data']),out/'source_baseline',loaded['step'])
        verify_state_equality(loaded,actual_state())
        if tree_digest(model_cache())!=cache_before:raise ValueError('Source baseline changed primed LUT')
        source_parity=verify_numeric_render_parity(anchor['render_directory'],out/'source_baseline/renders'/f"step-{loaded['step']:09d}",s['source_render_sha256'],anchor['source_eval'],source_metrics)
        atomic(out/'source_baseline_parity.json',dict(passed=True,source_verified_before_selector=True,render_parity=source_parity))
    density_before=None
    if not child:
        manifest=read(s['margin_manifest'])
        if sha(manifest['probe_positions'])!=manifest['probe_positions_sha256']:raise ValueError('Source-native diagnostic positions drift')
        with np.load(manifest['probe_positions'],allow_pickle=False)as positions:probe_xyz=torch.from_numpy(positions['xyz']).to(m.device)
        with torch.inference_mode():density_before=m.field.density_fn(probe_xyz).detach().clone()
        if not torch.isfinite(density_before).all():raise ValueError('Nonfinite source density probe')
    before_policy=tree_digest(_capture_rng_state());gate_policy=MarginPolicy(m,sampler,s['margin_manifest'],s['margin_manifest_sha256'],s['arm'])
    if child:gate_policy.load_state_dict(lineage['selector_state'])
    if tree_digest(_capture_rng_state())!=before_policy:raise ValueError('Margin installation consumed global RNG')
    if not child:
        with torch.inference_mode():
            density_after=m.field.density_fn(probe_xyz).detach()
            keep=gate_policy.lookup(probe_xyz)if gate_policy.lookup is not None else torch.ones(len(probe_xyz),device=m.device,dtype=torch.bool)
        if not torch.isfinite(density_after).all():raise ValueError('Nonfinite branch density probe')
        if not torch.equal(density_after[keep],density_before[keep])or torch.count_nonzero(density_after[~keep]).item():raise ValueError('Actual native density selector parity failed')
        if s['arm']=='source_margin002'and(not bool(keep.any())or not bool((~keep).any())or not bool((density_before[~keep]>0).any())):raise ValueError('Density probe lacks actual positive excluded/admitted support')
        atomic(out/'density_gate_proof.json',dict(passed=True,positions_sha256=manifest['probe_positions_sha256'],points=len(probe_xyz),admitted=int(keep.sum()),excluded=int((~keep).sum()),excluded_previously_nonzero=int((density_before[~keep]>0).sum()),admitted_density_bitwise_equal=True,excluded_density_exact_zero=True,global_RNG_unchanged=tree_digest(_capture_rng_state())==before_policy))
    verify_state_equality(loaded,actual_state())
    baseline.update(cache_ids=ids,cache_sha256=cache_hash,cache_sources=cache_sources,historical_cache_order_observed=False,cache_evidence='Frozen seed42 full-cache reconstruction, exact RGB/mask association; child IDs/sources explicitly checkpointed',selector_state_sha256=tree_digest(gate_policy.state_dict()),model_cache_sha256=tree_digest(model_cache()),source_verified_before_selector=True,FR=.3,distortion=.1,mode=s['mode'],Adam=adam_steps(loaded['optimizers']),next_step=trainer._start_step)
    atomic(out/f"resume_proof_{loaded['step']}.json",baseline)
    branch_zero=None
    if not child:
        reference=read(s['zero_reference'])
        for p,d in reference['artifacts_sha256'].items():
            if sha(p)!=d:raise ValueError('Audited branch-zero reference drift')
        cache_before=tree_digest(model_cache());selector_before=tree_digest(gate_policy.state_dict())
        branch_zero=base.evaluate(trainer,ready,Path(parent['data']),out/'branch_zero',loaded['step']);verify_state_equality(loaded,actual_state());gate_policy.verify_installed()
        if tree_digest(model_cache())!=cache_before or tree_digest(gate_policy.state_dict())!=selector_before:raise ValueError('Branch zero changed primed model cache/selector state')
        parity=verify_numeric_render_parity(reference['render_directory'],out/'branch_zero/renders'/f"step-{loaded['step']:09d}",reference['render_sha256'],reference['metrics'],branch_zero)
        atomic(out/'branch_zero_selector_replay.json',dict(kind='EXPLICIT_PARENT_PLUS_SELECTOR',checkpoint=s['parent_checkpoint'],checkpoint_sha256=SOURCE_SHA,spec=str(spec_path),spec_sha256=sha(spec_path),selector=gate_policy.binding,own_zero_reference_sha256=s['zero_reference_sha256'],loader='marshawn_margin_continuation_policy.install_zero_source_margin',actual_source_FR=.3))
        atomic(out/'branch_zero_parity.json',dict(passed=True,arm=s['arm'],render_parity=parity,selector_state=gate_policy.state_dict(),own_evaluation=branch_zero,sourceA_crossreference_only=anchor['source_eval']))
        if s['mode']=='zero':
            atomic(out/'complete.json',dict(status='zero_update_state_equal',passed=True,proof=baseline,render_parity=parity,selector_state=gate_policy.state_dict(),render_state_unchanged=True,sourceA_verified_before_gate=True,own_branch_zero=branch_zero));trainer.shutdown();return
    tracker=PlateauTracker.from_dict(lineage['plateau'])if child else PlateauTracker(anchor_step=130316)
    evaluations=copy.deepcopy(lineage['evaluations'])if child else[branch_zero];checks=copy.deepcopy(lineage['gradient_checks'])if child else[];control_checks=copy.deepcopy(lineage.get('source_control_checks',[]))if child else[]
    if s['mode']=='technical'and s['arm']=='original_mapped':
        native_loss=m.get_loss_dict;checked=[False]
        def source_control_loss(outputs,batch,metrics_dict=None):
            losses=native_loss(outputs,batch,metrics_dict)
            if not checked[0]:
                from marshawn_tuning_sampling import weighted_charbonnier
                index=batch['indices'].detach().cpu().numpy();weights=np.array([sampler.rows[c]['W64'][y,x]for c,y,x in index],np.float32);np.testing.assert_array_equal(weights,batch['actor_confidence'].cpu().numpy())
                reference=weighted_charbonnier(outputs['rgb'],batch['image'],torch.from_numpy(weights),gate_policy.normalizer(sampler.step,len(index)));torch.testing.assert_close(losses['rgb_loss'],reference,rtol=1e-12,atol=1e-12)
                actual_grad=torch.autograd.grad(losses['rgb_loss'],outputs['rgb'],retain_graph=True)[0];reference_grad=torch.autograd.grad(reference,outputs['rgb'],retain_graph=True)[0];torch.testing.assert_close(actual_grad,reference_grad,rtol=1e-6,atol=1e-10)
                control_checks.append(dict(step=sampler.step,source_W64_exact=True,source_main_batch_sha256=tree_digest(batch['indices']),source_original_RGB_gradient_max_error=float((actual_grad-reference_grad).abs().max()),actual_distortion=m.config.distortion_loss_mult,actual_FR=m.field.feature_reweighting_strength,extra_field_queries=0,extra_optimizer_updates=0));checked[0]=True
            return losses
        m.get_loss_dict=source_control_loss
    if child and (tracker.stopped or (s['mode']=='technical'and loaded['step']==SOURCE_STEP+64)):
        # Latest full-state checkpoint is committed before complete.json. Recover
        # that narrow crash window idempotently, without one extra Adam attempt.
        if s['mode']=='technical'and(len(checks)!=64 or not any(r['Adam_updated']for r in checks[:32])or not any(r['Adam_updated']for r in checks[32:])):raise ValueError('Terminal technical checkpoint lacks actual two-segment Adam evidence')
        atomic(out/'complete.json',dict(status='technical_resume_complete'if s['mode']=='technical'else'plateau',passed=True,final_step=loaded['step'],attempts=loaded['step']-SOURCE_STEP,plateau=tracker.to_dict(),evaluations=evaluations,gradient_checks=checks,source_control_checks=control_checks,sampling_segment=lineage['sampling_segment'],actual_Adam=adam_steps(loaded['optimizers']),selector_state=gate_policy.state_dict(),source_sha256=SOURCE_SHA,visual_review_complete=False,quality_promotion=False,recovered_terminal_checkpoint_without_updates=True));trainer.shutdown();return
    original_frequency_update=pipe._update_frequency_grid
    def observed_frequency_update(frequency_step):
        before_grid=tree_digest(m.freq_grid.state_dict());before_rng=tree_digest(_capture_rng_state())
        result=original_frequency_update(frequency_step)
        if not torch.isfinite(m.freq_grid.grid).all():raise FloatingPointError('Nonfinite actual original frequency refresh')
        atomic(out/f'frequency_update_{frequency_step}.json',dict(step=frequency_step,actual_original_method_called=True,unchanged_original_schedule=True,before_grid=before_grid,after_grid=tree_digest(m.freq_grid.state_dict()),before_rng=before_rng,after_rng=tree_digest(_capture_rng_state()),finite=True))
        return result
    pipe._update_frequency_grid=observed_frequency_update
    target=SOURCE_STEP+64 if s['mode']=='technical'else None;start_step=trainer._start_step;del loaded
    stop=[];signal.signal(signal.SIGTERM,lambda signum,frame:stop.append(signum));signal.signal(signal.SIGINT,lambda signum,frame:stop.append(signum));ckpts=out/'checkpoints';ckpts.mkdir(exist_ok=True);started=time.monotonic();last_report=0.;attempts=0;ray_hash=__import__('hashlib').sha256();first_ray=None;packets=[];original_next=pipe.datamanager.next_train
    def observed_next(step):
        nonlocal first_ray
        ray,batch=original_next(step);raw=batch['indices'].detach().cpu().contiguous().numpy().tobytes();digest=__import__('hashlib').sha256(raw).hexdigest();first_ray=first_ray or digest;ray_hash.update(raw);expected=gate_policy.normalizer(step,len(batch['indices']));error=abs(batch['actor_proposal_mean_weight']-expected)
        if error>1e-12:raise ValueError('Actual RGB normalizer differs from unchanged source expectation')
        if s['mode']=='technical':packets.append(dict(step=step,indices_sha256=digest,normalizer=batch['actor_proposal_mean_weight'],normalizer_error=error,confidence_sha256=tree_digest(batch['actor_confidence'])))
        return ray,batch
    pipe.datamanager.next_train=observed_next
    def segment(step):return dict(first_step=start_step,last_step=step,attempts=attempts,first_indices_sha256=first_ray,indices_sha256=ray_hash.hexdigest(),packets=packets)
    def cache_state():return dict(ids=ids,sources=cache_sources,train_count=pipe.datamanager.train_count)
    def checkpoint_path(step):return str(Path(s['parent_checkpoint']))if step==SOURCE_STEP else str(ckpts/f'step-{step:09d}.ckpt')
    def save(step):
        payload=dict(step=step,**actual_state(),margin_fork_runtime=dict(spec_sha256=sha(spec_path),source_sha256=SOURCE_SHA,selector_state=gate_policy.state_dict(),model_cache=model_cache(),cache_state=cache_state(),plateau=tracker.to_dict(),evaluations=evaluations,gradient_checks=checks,source_control_checks=control_checks,sampling_segment=segment(step)))
        path=ckpts/f'step-{step:09d}.ckpt';temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path);export_snapshot(m,path,step,ckpts,sources={**anchor['source_hashes'],'margin_manifest':s['margin_manifest_sha256'],'margin_spec':sha(spec_path)});atomic(latest,dict(step=step,checkpoint=str(path.relative_to(out)),sha256=sha(path)))
        selected=base.choose_checkpoint(evaluations);raw=min(evaluations,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr'],r['step']));atomic(out/'retention.json',dict(parent=checkpoint_path(SOURCE_STEP),selected=checkpoint_path(selected['step']),selected_step=selected['step'],selected_selector=gate_policy.binding,selected_loader='install_zero_source_margin'if selected['step']==SOURCE_STEP else'install_checkpoint_margin',margin_spec=str(spec_path),margin_spec_sha256=sha(spec_path),raw_min=checkpoint_path(raw['step']),raw_min_step=raw['step'],raw_min_selector=gate_policy.binding,raw_min_loader='install_zero_source_margin'if raw['step']==SOURCE_STEP else'install_checkpoint_margin',latest=str(path),early_visual_steps=[SOURCE_STEP+n for n in (500,1000,2000,4000)],pruning='none; preserve all evaluated/atomic resume states in this matched campaign'))
    step=start_step
    while True:
        gate_policy.verify_installed();trainer.step=step;pipe.train();sampler.set_step(step);m.config.distortion_loss_mult=.1;trainer._apply_live_optimizer_switch(step)
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=Location.BEFORE_TRAIN_ITERATION)
        if m.config.distortion_loss_mult!=.1 or m.field.feature_reweighting_strength!=.3 or m.config.feature_reweighting_strength!=.3:raise ValueError('Actual live source loss scalars changed before train_iteration')
        before=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()});rng_before=tree_digest(_capture_rng_state())if s['mode']=='technical'else None
        loss,_,_=trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()):raise FloatingPointError('Nonfinite matched margin objective')
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
            import subprocess,sys
            subprocess.run([sys.executable,str(Path(__file__).with_name('export_marshawn_core_boundary_metrics_cpu.py')),'--data',parent['data'],'--renders',str(out/'renders'/f'step-{step:09d}'),'--output',str(out/'region_diagnostics'/f'step-{step:09d}.json')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            atomic(out/'metrics.json',evaluations);atomic(out/'visual_gates.json',[dict(step=r['step'],attempts=r['step']-SOURCE_STEP,renders=str(out/'renders'/f"step-{r['step']:09d}"),used_for_plateau=r['step']>=tracker.anchor_step,visual_review_complete=False)for r in evaluations if r['step']!=SOURCE_STEP])
        partial=stop_after_attempts is not None and attempts>=stop_after_attempts;terminal=(target is not None and step==target)or(s['mode']=='production'and tracker.stopped)
        if s['mode']=='technical'and(partial or terminal):
            technical_metrics=base.evaluate(trainer,ready,Path(parent['data']),out/'technical_outputs',step);gate_policy.verify_installed()
            atomic(out/f'technical_render_{step}.json',dict(step=step,metrics=technical_metrics,selector_state=gate_policy.state_dict(),finite_fullFHD=True))
        if early or plateau_gate or partial or terminal or stop or delta%2000==0:save(step)
        if early or plateau_gate or partial or terminal or time.monotonic()-last_report>=30:
            atomic(out/'progress.json',dict(step=step,attempts=delta,elapsed_seconds=time.monotonic()-started,plateau=tracker.to_dict(),mode=s['mode'],selector_state_sha256=tree_digest(gate_policy.state_dict()),sampling_segment=segment(step),Adam=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()})));last_report=time.monotonic()
        if terminal:
            final_adam=adam_steps({k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()})
            if not all(min(v)>min(baseline['Adam'][k])for k,v in final_adam.items()):raise ValueError('No actual resumed Adam update in this invocation')
            atomic(out/'complete.json',dict(status='technical_resume_complete'if s['mode']=='technical'else'plateau',passed=True,final_step=step,attempts=delta,plateau=tracker.to_dict(),evaluations=evaluations,gradient_checks=checks,source_control_checks=control_checks,sampling_segment=segment(step),actual_Adam=final_adam,selector_state=gate_policy.state_dict(),source_sha256=SOURCE_SHA,visual_review_complete=False,quality_promotion=False));trainer.shutdown();return
        if partial:trainer.shutdown();return
        if stop:
            atomic(out/'interrupted.json',dict(step=step,signals=stop,checkpoint_committed=True,quality_complete=False));trainer.shutdown();raise SystemExit(75)
        step+=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--stop-after-attempts',type=int);a=p.parse_args();run(a.spec,a.stop_after_attempts)
