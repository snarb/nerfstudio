"""Finite soft floor-band scene prior continuation; not measured depth."""
from __future__ import annotations
import argparse,csv,json,math,os,signal,sys,threading,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for directory in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',
                  ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code',Path(__file__).parent]:
    sys.path.insert(0,str(directory))
from staged_native_hash import file_sha


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
        raise ValueError('Only immutable selected10k full-state parent allowed')
    if file_sha(cfg.load_checkpoint)!=spec['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    if cfg.load_dir or cfg.load_step is not None or cfg.eval_checkpoint or cfg.load_config:
        raise ValueError('Ambiguous parent selection')
    if cfg.max_num_iterations!=21000 or cfg.checkpoint_load_mode!='resume' or not cfg.load_optimizers or not cfg.load_scheduler:
        raise ValueError('Full-component resume with absolute global end21000 required')
    if cfg.machine.seed!=42 or cfg.pipeline._target.__name__!='NativeFloorBandPriorPipeline':raise ValueError('Wrong pipeline/seed')
    if hasattr(cfg.pipeline,'patch_dssim_weight') or hasattr(cfg.pipeline,'patch_dssim_size'):
        raise ValueError('Uniform RGB-MSE control must have no DSSIM settings')
    if cfg.pipeline.datamanager.pixel_sampler._target.__name__!='PixelSampler':
        raise ValueError('Original independent PixelSampler required')
    if cfg.pipeline.packed_distortion_mode!='linear' or cfg.pipeline.packed_distortion_weight!=.01 or cfg.pipeline.final_grid_radius!=0:
        raise ValueError('Existing native core0/linearDST.01 must remain')
    if cfg.pipeline.target_num_samples!=1048576 or cfg.pipeline.max_num_samples_per_ray!=16384:
        raise ValueError('Reviewed native dynamic initial estimate required')
    if [cfg.steps_per_save,cfg.steps_per_eval_batch,cfg.steps_per_eval_image,cfg.steps_per_eval_all_images]!=[2000]*4:
        raise ValueError('Reviewed2k evaluation/checkpoints required')
    if spec['parent_attempt_upper_bound']+cfg.max_num_iterations-10001!=21999:
        raise ValueError('Parent+phase total budget mismatch')
    if cfg.resume_fields_lr_override is not None:
        raise ValueError('LR override forbidden: preserve actual parent optimizer/scheduler state')
    if cfg.pipeline.model.background_color!='random' or cfg.optimizers['fields']['scheduler'].max_steps!=22000:
        raise ValueError('Only native randomBG loss and unchanged22k scheduler allowed')
    review=json.loads(Path(spec['floor_data_root_review']).read_text())
    if review['proposal_sha256']!=cfg.pipeline.floor_band_prior_manifest_sha256 or not review['plane_is_scene_assumption'] or not review['known_depth_not_measured_here']:
        raise ValueError('Explicit ROOT finite-prior DATA approval must bind this unmeasured hypothesis')
    if cfg.pipeline.floor_band_prior_weight not in(.0001,.0003,.001,.003,.01):
        raise ValueError('Only reviewed floor-prior probe weights allowed')
    return cfg,spec


def validate(trainer,spec):
    import torch
    p=trainer.pipeline;m=p.model
    assert trainer._start_step==10001 and trainer._loaded_rng_state is not None
    assert type(p).__name__=='NativeFloorBandPriorPipeline'and type(m).__name__=='NGPModel'
    assert p.dynamic_num_rays_per_batch==64 and p.datamanager.train_pixel_sampler.num_rays_per_batch==64
    assert p.config.datamanager.patch_size==1 and not p.config.datamanager.pixel_sampler.ignore_mask
    assert m.config.render_step_size==.003 and m.config.max_res==8192 and m.config.log2_hashmap_size==21
    assert m.field.appearance_embedding_dim==0 and m.config.eval_num_rays_per_chunk==512
    assert m.config.background_color==m.renderer_rgb.background_color=='random'
    from nerfstudio.model_components import renderers
    assert renderers.BACKGROUND_COLOR_OVERRIDE is None
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
            if 10001<=step<21000 and checkpoint.exists()and all(map(math.isfinite,values)):
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
    if not all(spec.get('release_checks',{}).get(k)for k in ['root_training_authorized','actual_zeroopt_probe_passed','peer_review_passed']):
        raise ValueError('Production gates required even for direct train mode')
    def loop(local_rank,world_size,config,global_rank=0):
        native._set_random_seed(config.machine.seed+global_rank)
        trainer=config.setup(local_rank=local_rank,world_size=world_size);trainer.setup();validate(trainer,spec)
        folder=Path(config.output_dir)
        initial_counts=[int(s['step'])for opt in trainer.optimizers.optimizers.values()for s in opt.state.values()if'step'in s]
        initial_count=max(initial_counts)
        atomic_json(folder/'worker.json',dict(pid=os.getpid(),time=time.time(),parent_step=10000,
            start_step=trainer._start_step,global_end=21000,total_attempt_upper_bound=21999,
            loaded_RNG_pending_native_last_point_restore=True,initial_adam_updates=initial_count,
            initial_scale=float(trainer.grad_scaler.get_scale()),LR_override=False,background_policy='native_random_loss_black_forward',independent_uniform_pixel_sampler=True))
        native_save=trainer.save_checkpoint
        def save_and_bind(step):
            native_save(step)
            trainer.pipeline.export_grid_for_checkpoint(trainer.checkpoint_dir/('step-%09d.ckpt'%step),step)
        trainer.save_checkpoint=save_and_bind
        native_eval=trainer.eval_iteration
        def protected_eval(step):
            if step>10000 and step%2000==0:trainer.save_checkpoint(step)
            return native_eval(step)
        trainer.eval_iteration=protected_eval
        native_iteration=trainer.train_iteration;paused=[False]
        def observed(step):
            if step==12001 and not paused[0]:
                checkpoint=trainer.checkpoint_dir/'step-000012000.ckpt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved['step']!=12000 or not all(bool(torch.isfinite(v).all())for v in saved['pipeline'].values()if torch.is_tensor(v)):
                    raise ValueError('Invalid12k gate checkpoint')
                assert all(k in saved for k in ['optimizers','schedulers','scalers','rng_state'])
                gate=dict(step=12000,next_step=12001,phase_attempts=2000,total_attempt_upper_bound=spec['parent_attempt_upper_bound']+2000,
                    worker_pid=os.getpid(),checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
                    pause='SIGSTOP same fully restored worker before next optimizer update',time=time.time())
                del saved
                atomic_json(folder/'gate_12k.json',gate);paused[0]=True
                os.kill(os.getpid(),signal.SIGSTOP)
            lr_used=float(trainer.optimizers.optimizers['fields'].param_groups[0]['lr'])
            result=native_iteration(step)
            if step<10065 or step%500==0:
                counts=[int(s['step'])for opt in trainer.optimizers.optimizers.values()for s in opt.state.values()if'step'in s]
                count=max(counts);phase_attempts=step-10001+1
                row=dict(step=step,phase_attempts=phase_attempts,adam_updates=count,
                    phase_adam_updates=count-initial_count,phase_skipped_attempts=phase_attempts-(count-initial_count),
                    total_attempt_upper_bound=spec['parent_attempt_upper_bound']+phase_attempts,
                    lr_used_for_update=lr_used,lr_after_scheduler=float(trainer.optimizers.optimizers['fields'].param_groups[0]['lr']),scheduler_last_epoch=trainer.optimizers.schedulers['fields'].last_epoch,scale=float(trainer.grad_scaler.get_scale()),scalar_finite=bool(torch.isfinite(result[0].detach())))
                if step in(10032,10064)or step%2000==0:row['weights_finite']=all(bool(torch.isfinite(p).all())for p in trainer.pipeline.model.parameters())
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
        if not all(spec.get('release_checks',{}).get(k)for k in ['root_training_authorized','actual_zeroopt_probe_passed','peer_review_passed']):
            raise ValueError('Root review + independent production/fullstate/strict AMP gates required')
        import run_bounded_ngp_quiet as quiet
        from run_marshawn_png_distortion_control import health_loop
        sys.argv=[sys.argv[0]]+rest
        quiet.train_command=lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(args.config),'--spec',str(args.spec)]
        from marshawn_all_view_early_stop import configure_quiet_all_view_stop
        configure_quiet_all_view_stop(quiet,lambda metrics,directory:choose_parent_and_phase(metrics,directory,spec))
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
