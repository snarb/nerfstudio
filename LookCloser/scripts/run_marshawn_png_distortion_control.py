"""One matched cold PNG distortion control; quiet supervision and exact 2k gate."""
from __future__ import annotations
import argparse,csv,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',Path(__file__).parent]:
    sys.path.insert(0,str(d))
from staged_native_hash import file_sha


def load(config_path,spec_path):
    import yaml
    spec=json.loads(Path(spec_path).read_text())
    if file_sha(config_path)!=spec['config_sha256']:raise ValueError('Config SHA changed')
    for path,digest in spec['sources_sha256'].items():
        if file_sha(path)!=digest:raise ValueError('Source changed '+path)
    cfg=yaml.load(Path(config_path).read_text(),Loader=yaml.Loader)
    if cfg.max_num_iterations!=22000 or cfg.load_checkpoint or cfg.load_dir or cfg.load_step is not None or cfg.eval_checkpoint:
        raise ValueError('Exactly22000 hard cap and fresh cold model required')
    if cfg.machine.seed!=42 or cfg.pipeline.final_grid_radius!=0:raise ValueError('Seed42/core0 required')
    if cfg.pipeline.packed_distortion_mode!='linear' or cfg.pipeline.packed_distortion_weight!=.01:
        raise ValueError('Reviewed bounded-linear .01 only')
    if cfg.pipeline._target.__qualname__!='NativePackedDistortionPipeline':raise ValueError('Wrong YAML target')
    if [cfg.steps_per_save,cfg.steps_per_eval_batch,cfg.steps_per_eval_image,cfg.steps_per_eval_all_images]!=[2000]*4:
        raise ValueError('Matched2k checkpoint/all-view evaluation required')
    return cfg,spec


def validate(pipe):
    import torch
    m=pipe.model
    assert type(pipe).__name__=='NativePackedDistortionPipeline'and type(m).__name__=='NGPModel'
    assert m.config.render_step_size==.003 and m.config.max_res==8192 and m.config.log2_hashmap_size==21
    assert m.field.appearance_embedding_dim==0 and m.config.eval_num_rays_per_chunk==512
    assert pipe.config.datamanager.cache_images_type=='uint8'and pipe.config.datamanager.patch_size==1
    assert len(pipe.datamanager.train_dataset)==63 and len(pipe.datamanager.eval_dataset)==3
    expected=torch.tensor(json.loads((ROOT/'geometry/boxes_approved.json').read_text())['scopes']['scene']['loose'])
    boxes={'model':m.scene_aabb.reshape(2,3),'field':m.field.aabb,'grid':m.occupancy_grid.aabbs[0].reshape(2,3),
           'train':pipe.datamanager.train_dataparser_outputs.scene_box.aabb,
           'eval':pipe.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb}
    for key,value in boxes.items():torch.testing.assert_close(value.cpu(),expected,rtol=0,atol=1e-6)
    out=pipe.datamanager.train_dataparser_outputs
    torch.testing.assert_close(out.dataparser_transform,torch.eye(4)[:3],rtol=0,atol=1e-7)
    assert out.dataparser_scale==1
    return {k:v.cpu().tolist()for k,v in boxes.items()}


def train(path,spec_path):
    import torch
    import nerfstudio.scripts.train as native
    cfg,spec=load(path,spec_path)
    def loop(local_rank,world_size,config,global_rank=0):
        native._set_random_seed(config.machine.seed+global_rank)
        trainer=config.setup(local_rank=local_rank,world_size=world_size);trainer.setup()
        bounds=validate(trainer.pipeline);folder=Path(config.output_dir)
        (folder/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time(),bounds=bounds,
            cold_seed=42,max_attempts=22000,distortion_mode='bounded_linear',distortion_weight=.01),indent=2)+'\n')
        original_save=trainer.save_checkpoint
        def save_and_bind(step):
            original_save(step)
            trainer.pipeline.export_grid_for_checkpoint(trainer.checkpoint_dir/('step-%09d.ckpt'%step),step)
        trainer.save_checkpoint=save_and_bind
        original_eval=trainer.eval_iteration
        def protected_eval(step):
            if step>0 and step%2000==0:trainer.save_checkpoint(step)
            return original_eval(step)
        trainer.eval_iteration=protected_eval
        original=trainer.train_iteration;paused=[False]
        def observed(step):
            if step==2001 and not paused[0]:
                checkpoint=trainer.checkpoint_dir/'step-000002000.ckpt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved['step']!=2000 or not all(bool(torch.isfinite(v).all())for v in saved['pipeline'].values()if torch.is_tensor(v)):
                    raise ValueError('2k gate checkpoint invalid/nonfinite')
                gate=dict(step=2000,next_step=2001,spent_attempts=2001,worker_pid=os.getpid(),
                    checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
                    finite=True,pause='SIGSTOP exact worker before next optimizer update',time=time.time())
                del saved
                (folder/'gate_2k.json').write_text(json.dumps(gate,indent=2)+'\n');paused[0]=True
                os.kill(os.getpid(),signal.SIGSTOP)
            result=original(step)
            if step<64 or step%2000==0:
                counts=[int(state['step'])for optimizer in trainer.optimizers.optimizers.values()
                        for state in optimizer.state.values()if'step'in state]
                count=max(counts)if counts else 0
                row=dict(step=step,adam_updates=count,cumulative_skipped_attempts=step+1-count,
                    scale=float(trainer.grad_scaler.get_scale()),scalar_finite=bool(torch.isfinite(result[0].detach())))
                if step in(31,63)or step%2000==0:row['weights_finite']=all(bool(torch.isfinite(p).all())for p in trainer.pipeline.model.parameters())
                with(folder/'amp_optimizer_trace.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
            return result
        trainer.train_iteration=observed;trainer.train()
    native.train_loop=loop;native.main(cfg)


def health_loop(cfg,stop):
    folder=Path(cfg.output_dir);run=cfg.get_base_dir()
    while not stop.is_set():
        processes=[]
        children=Path('/proc/%d/task/%d/children'%(os.getpid(),os.getpid()))
        for value in children.read_text().split()if children.exists()else[]:
            p=Path('/proc')/value
            if p.exists():
                stat=p.joinpath('stat').read_text().split()
                processes.append(dict(pid=int(value),state=stat[2],start_ticks=stat[21],
                    command=p.joinpath('cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')))
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
        latest=None;csv_path=run/'metrics_compact.csv'
        if csv_path.exists():
            rows=list(csv.DictReader(csv_path.open()));latest=int(rows[-1]['step'])if rows else None
        log=run/'train_stdout.log';tail=log.read_text(errors='replace')[-12000:]if log.exists()else''
        row=dict(time=time.time(),controller_pid=os.getpid(),children=processes,latest_step=latest,
            checkpoints=[p.name for p in (run/'nerfstudio_models').glob('*.ckpt')],gpu_MiB_util=gpu,
            OOM_evidence=('CUDA out of memory'in tail or'OutOfMemoryError'in tail))
        with(folder/'health_30s.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        stop.wait(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controller','train'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);args,rest=p.parse_known_args()
    if args.mode=='train':train(args.config,args.spec)
    else:
        cfg,spec=load(args.config,args.spec)
        if not spec.get('release_checks',{}).get('production_peer_review_passed')or not spec.get('release_checks',{}).get('cold_probe_passed'):
            raise ValueError('Both independent production review and cold probe must pass')
        import run_bounded_ngp_quiet as quiet
        import run_marshawn_native_ngp as native
        sys.argv=[sys.argv[0]]+rest
        quiet.train_command=lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(args.config),'--spec',str(args.spec)]
        quiet.best_eval_checkpoint=native.select
        os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
        stop=threading.Event();thread=threading.Thread(target=health_loop,args=(cfg,stop),daemon=True);thread.start()
        try:result=quiet.main()
        finally:stop.set();thread.join(timeout=2)
        raise SystemExit(result)
