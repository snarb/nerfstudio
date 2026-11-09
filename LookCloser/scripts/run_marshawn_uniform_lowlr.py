"""Low-LR full-state independent-pixel MSE control; in-place 9k gate."""
from __future__ import annotations
import argparse,csv,json,math,os,signal,sys,threading,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for directory in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',
                  ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code',Path(__file__).parent]:
    sys.path.insert(0,str(directory))
from staged_native_hash import file_sha


def scale_loaded_lr(optimizers, target_current_lr):
    """Scale restored LR metadata once; retain the actual LambdaLR and progress.

    Adam tensors/steps, scaler and RNG are untouched. Native scheduler.step()
    advances its saved epoch only after a successful AMP update; do not replace
    that epoch with the global iteration, or rebuild its captured lambda.
    """
    opt=optimizers.optimizers['fields'];scheduler=optimizers.schedulers['fields']
    if len(opt.param_groups)!=1 or len(scheduler.base_lrs)!=1:
        raise ValueError('Reviewed one fields parameter group required')
    target=float(target_current_lr);group=opt.param_groups[0];current=float(group['lr'])
    if not math.isfinite(target) or target<=0 or not math.isfinite(current) or current<=target:
        raise ValueError('Positive strict reduction from the loaded current LR required')
    if not math.isclose(scheduler.get_last_lr()[0],current,rel_tol=1e-12):
        raise ValueError('Loaded optimizer and scheduler current LR disagree')
    factor=target/current;epoch=scheduler.last_epoch;step_count=scheduler._step_count
    functions=tuple(scheduler.lr_lambdas)
    before=dict(current_lr=current,initial_lr=float(group['initial_lr']),
        default_lr=float(opt.defaults['lr']),base_lrs=list(scheduler.base_lrs),
        last_lrs=list(scheduler.get_last_lr()),last_epoch=epoch,step_count=step_count)
    group['lr']=target
    group['initial_lr']=float(group['initial_lr'])*factor
    opt.defaults['lr']=float(opt.defaults['lr'])*factor
    scheduler.base_lrs=[float(v)*factor for v in scheduler.base_lrs]
    scheduler._last_lr=[float(v)*factor for v in scheduler._last_lr]
    assert scheduler.last_epoch==epoch and scheduler._step_count==step_count
    assert tuple(scheduler.lr_lambdas)==functions
    expected_next=scheduler.base_lrs[0]*functions[0](epoch+1)
    return dict(policy='scale all restored LR metadata; preserve captured lambda and successful-update epoch',
        factor=factor,target_current_lr=target,before=before,
        after=dict(current_lr=float(group['lr']),initial_lr=float(group['initial_lr']),
            default_lr=float(opt.defaults['lr']),base_lrs=list(scheduler.base_lrs),
            last_lrs=list(scheduler.get_last_lr()),last_epoch=epoch,step_count=step_count),
        expected_next_successful_update_lr=expected_next)


def atomic_json(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def load(path,spec_path):
    import yaml
    spec=json.loads(Path(spec_path).read_text())
    if file_sha(path)!=spec['config_sha256']:raise ValueError('Config binding changed')
    for source,digest in spec['sources_sha256'].items():
        if file_sha(source)!=digest:raise ValueError('Frozen source/resource changed '+source)
    cfg=yaml.load(Path(path).read_text(),Loader=yaml.Loader)
    if cfg.load_checkpoint is None or Path(cfg.load_checkpoint)!=Path(spec['parent_checkpoint']):
        raise ValueError('Only immutable selected8k full-state parent allowed')
    if file_sha(cfg.load_checkpoint)!=spec['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    if cfg.load_dir or cfg.load_step is not None or cfg.eval_checkpoint or cfg.load_config:
        raise ValueError('Ambiguous parent selection')
    if cfg.max_num_iterations!=16000 or cfg.checkpoint_load_mode!='resume' or not cfg.load_optimizers or not cfg.load_scheduler:
        raise ValueError('Full-component resume with absolute global end16000 required')
    if cfg.machine.seed!=42 or cfg.pipeline._target.__name__!='NativePackedDistortionPipeline':raise ValueError('Wrong pipeline/seed')
    if hasattr(cfg.pipeline,'patch_dssim_weight') or hasattr(cfg.pipeline,'patch_dssim_size'):
        raise ValueError('Uniform RGB-MSE control must have no DSSIM settings')
    if cfg.pipeline.datamanager.pixel_sampler._target.__name__!='PixelSampler':
        raise ValueError('Original independent PixelSampler required')
    if cfg.pipeline.packed_distortion_mode!='linear' or cfg.pipeline.packed_distortion_weight!=.01 or cfg.pipeline.final_grid_radius!=0:
        raise ValueError('Existing native core0/linearDST.01 must remain')
    if cfg.pipeline.target_num_samples!=262144 or cfg.pipeline.max_num_samples_per_ray!=16384:
        raise ValueError('Reviewed native dynamic initial estimate required')
    if [cfg.steps_per_save,cfg.steps_per_eval_batch,cfg.steps_per_eval_image,cfg.steps_per_eval_all_images]!=[1000]*4:
        raise ValueError('Reviewed1k evaluation/checkpoints required')
    if spec['parent_attempt_upper_bound']+cfg.max_num_iterations-8001!=21609:
        raise ValueError('Parent+phase total budget mismatch')
    if cfg.resume_fields_lr_override is not None or spec['target_current_lr']!=.0005:
        raise ValueError('Only reviewed postload current-LR scaling; native base override forbidden')
    return cfg,spec


def validate(trainer,spec):
    import torch
    p=trainer.pipeline;m=p.model
    assert trainer._start_step==8001 and trainer._loaded_rng_state is not None
    assert type(p).__name__=='NativePackedDistortionPipeline'and type(m).__name__=='NGPModel'
    assert p.dynamic_num_rays_per_batch==16 and p.datamanager.train_pixel_sampler.num_rays_per_batch==16
    assert p.config.datamanager.patch_size==1 and not p.config.datamanager.pixel_sampler.ignore_mask
    assert m.config.render_step_size==.003 and m.config.max_res==8192 and m.config.log2_hashmap_size==21
    assert m.field.appearance_embedding_dim==0 and m.config.eval_num_rays_per_chunk==512
    expected=torch.tensor(json.loads((ROOT/'geometry/boxes_approved.json').read_text())['scopes']['scene']['loose'])
    arrays=[m.scene_aabb.reshape(2,3),m.field.aabb,m.occupancy_grid.aabbs[0].reshape(2,3),
            p.datamanager.train_dataparser_outputs.scene_box.aabb,p.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb]
    for value in arrays:torch.testing.assert_close(value.cpu(),expected,rtol=0,atol=1e-6)
    out=p.datamanager.train_dataparser_outputs
    torch.testing.assert_close(out.dataparser_transform,torch.eye(4)[:3],rtol=0,atol=1e-7)
    assert out.dataparser_scale==1 and len(p.datamanager.train_dataset)==63 and len(p.datamanager.eval_dataset)==3


def choose_parent_and_phase(metrics,directory,spec):
    eligible=[dict(spec['parent_candidate'])]
    if Path(metrics).exists():
        for row in csv.DictReader(Path(metrics).open()):
            try:
                step=int(row['step']);values=[float(row[k])for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']]
            except(KeyError,ValueError,TypeError):continue
            checkpoint=Path(directory)/('step-%09d.ckpt'%step)
            if 8001<=step<16000 and checkpoint.exists()and all(map(math.isfinite,values)):
                eligible.append(dict(checkpoint=str(checkpoint),step=step,psnr=values[0],ssim=values[1],lpips=values[2],source='phase_all_view_metrics'))
    best=max(r['psnr']for r in eligible)
    winner=min((r for r in eligible if r['psnr']>=best-.07-1e-12),key=lambda r:(r['lpips'],-r['psnr']))
    atomic_json(spec['selection_receipt'],dict(candidates=eligible,winner=winner,
        rule='Highest all-view PSNR with lowest LPIPS within0.07dB',parent_checkpoint_sha256=spec['parent_checkpoint_sha256']))
    return Path(winner['checkpoint']),'parent_plus_phase_all_eval_psnr_lpips_tie_0.07'


def exact_eval_config(config,checkpoint,eval_num_rays_per_chunk=None):
    """The winner may be outside this phase; bind the actual selected file."""
    import yaml
    cfg=yaml.load(Path(config).read_text(),Loader=yaml.Loader)
    cfg.eval_checkpoint=Path(checkpoint).resolve()
    cfg.load_dir=None;cfg.load_step=None;cfg.load_checkpoint=None;cfg.load_config=None
    if eval_num_rays_per_chunk is not None:cfg.pipeline.model.eval_num_rays_per_chunk=eval_num_rays_per_chunk
    output=Path(config).with_name('eval_exact_'+Path(checkpoint).stem+'.yml')
    output.write_text(yaml.dump(cfg))
    return output


def train(path,spec_path):
    import torch
    import nerfstudio.scripts.train as native
    cfg,spec=load(path,spec_path)
    if not all(spec.get('release_checks',{}).get(k)for k in ['root_training_authorized','postload_proof_passed','peer_review_passed']):
        raise ValueError('Production gates required even for direct train mode')
    def loop(local_rank,world_size,config,global_rank=0):
        native._set_random_seed(config.machine.seed+global_rank)
        trainer=config.setup(local_rank=local_rank,world_size=world_size);trainer.setup();validate(trainer,spec)
        lr_audit=scale_loaded_lr(trainer.optimizers,spec['target_current_lr'])
        trainer.checkpoint_load_audit['postload_lr_scale']=lr_audit
        folder=Path(config.output_dir)
        initial_counts=[int(s['step'])for opt in trainer.optimizers.optimizers.values()for s in opt.state.values()if'step'in s]
        initial_count=max(initial_counts)
        atomic_json(folder/'worker.json',dict(pid=os.getpid(),time=time.time(),parent_step=8000,
            start_step=trainer._start_step,global_end=16000,total_attempt_upper_bound=21609,
            loaded_RNG_pending_native_last_point_restore=True,initial_adam_updates=initial_count,
            initial_scale=float(trainer.grad_scaler.get_scale()),postload_lr_scale=lr_audit,independent_uniform_pixel_sampler=True))
        native_save=trainer.save_checkpoint
        def save_and_bind(step):
            native_save(step)
            trainer.pipeline.export_grid_for_checkpoint(trainer.checkpoint_dir/('step-%09d.ckpt'%step),step)
        trainer.save_checkpoint=save_and_bind
        native_eval=trainer.eval_iteration
        def protected_eval(step):
            if step>8000 and step%1000==0:trainer.save_checkpoint(step)
            return native_eval(step)
        trainer.eval_iteration=protected_eval
        native_iteration=trainer.train_iteration;paused=[False]
        def observed(step):
            if step==9001 and not paused[0]:
                checkpoint=trainer.checkpoint_dir/'step-000009000.ckpt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved['step']!=9000 or not all(bool(torch.isfinite(v).all())for v in saved['pipeline'].values()if torch.is_tensor(v)):
                    raise ValueError('Invalid9k gate checkpoint')
                assert all(k in saved for k in ['optimizers','schedulers','scalers','rng_state'])
                gate=dict(step=9000,next_step=9001,phase_attempts=1000,total_attempt_upper_bound=spec['parent_attempt_upper_bound']+1000,
                    worker_pid=os.getpid(),checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
                    pause='SIGSTOP same fully restored worker before next optimizer update',time=time.time())
                del saved
                atomic_json(folder/'gate_9k.json',gate);paused[0]=True
                os.kill(os.getpid(),signal.SIGSTOP)
            lr_used=float(trainer.optimizers.optimizers['fields'].param_groups[0]['lr'])
            result=native_iteration(step)
            if step<8065 or step%500==0:
                counts=[int(s['step'])for opt in trainer.optimizers.optimizers.values()for s in opt.state.values()if'step'in s]
                count=max(counts);phase_attempts=step-8001+1
                row=dict(step=step,phase_attempts=phase_attempts,adam_updates=count,
                    phase_adam_updates=count-initial_count,phase_skipped_attempts=phase_attempts-(count-initial_count),
                    total_attempt_upper_bound=spec['parent_attempt_upper_bound']+phase_attempts,
                    lr_used_for_update=lr_used,lr_after_scheduler=float(trainer.optimizers.optimizers['fields'].param_groups[0]['lr']),scheduler_last_epoch=trainer.optimizers.schedulers['fields'].last_epoch,scale=float(trainer.grad_scaler.get_scale()),scalar_finite=bool(torch.isfinite(result[0].detach())))
                if step in(8032,8064)or step%1000==0:row['weights_finite']=all(bool(torch.isfinite(p).all())for p in trainer.pipeline.model.parameters())
                with(folder/'amp_optimizer_trace.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
            return result
        trainer.train_iteration=observed;trainer.train()
    native.train_loop=loop;native.main(cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controller','train'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--spec',type=Path,required=True)
    args,rest=parser.parse_known_args()
    if args.mode=='train':train(args.config,args.spec)
    else:
        cfg,spec=load(args.config,args.spec)
        if not all(spec.get('release_checks',{}).get(k)for k in ['root_training_authorized','postload_proof_passed','peer_review_passed']):
            raise ValueError('Root review + independent production/fullstate/strict AMP gates required')
        import run_bounded_ngp_quiet as quiet
        from run_marshawn_png_distortion_control import health_loop
        sys.argv=[sys.argv[0]]+rest
        quiet.train_command=lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(args.config),'--spec',str(args.spec)]
        quiet.best_eval_checkpoint=lambda metrics,directory:choose_parent_and_phase(metrics,directory,spec)
        quiet.eval_config_for_step=exact_eval_config
        native_final_eval=quiet.run_final_eval
        def checked_final_eval(run_path,checkpoint,*a,**kw):
            result=native_final_eval(run_path,checkpoint,*a,**kw)
            if Path(result['checkpoint']).resolve()!=Path(checkpoint).resolve():
                raise ValueError('Final eval loaded a different checkpoint than selected')
            return result
        quiet.run_final_eval=checked_final_eval
        os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
        stop=threading.Event();thread=threading.Thread(target=health_loop,args=(cfg,stop),daemon=True);thread.start()
        try:result=quiet.main()
        finally:stop.set();thread.join(timeout=2)
        raise SystemExit(result)
