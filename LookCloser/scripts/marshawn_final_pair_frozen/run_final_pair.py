"""Run one cold final specification; native NN/optimizer source is unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
sys.path.insert(0,str(ROOT/'code_sfm/LookCloser/scripts'))
sys.path.insert(0,str(ROOT/'code_native/LookCloser/scripts'))
sys.path.insert(0,str(ROOT/'code_native'))
sys.path.insert(0,str(Path(__file__).parent))


def load(path):
    import yaml
    cfg=yaml.load(Path(path).read_text(),Loader=yaml.Loader)
    if not 0<cfg.max_num_iterations<=22000:raise ValueError('Absolute attempt cap<=22000 required')
    if cfg.load_checkpoint or cfg.load_dir or cfg.load_step is not None:raise ValueError('Final models are cold, not resumed')
    return cfg


def validate(trainer):
    import torch
    cfg=trainer.config;pipe=trainer.pipeline;model=pipe.model
    variant=pipe.config.final_geometry_variant;scope,size=variant.split('_')
    expected=torch.tensor(json.loads((ROOT/'geometry/boxes_approved.json').read_text())['scopes'][scope][size])
    arrays=dict(model=model.scene_aabb.reshape(2,3),field=model.field.aabb,
        occupancy=model.occupancy_grid.aabbs[0].reshape(2,3),
        train_parser=pipe.datamanager.train_dataparser_outputs.scene_box.aabb,
        eval_parser=pipe.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb)
    for name,value in arrays.items():
        if not torch.allclose(value.cpu(),expected,rtol=0,atol=1e-6):raise ValueError('Bound mismatch '+name)
    assert type(model).__name__=='NGPModel' and type(pipe).__name__=='FinalNativePipeline'
    assert model.field.appearance_embedding_dim==0 and model.config.eval_num_rays_per_chunk==512
    assert model.config.render_step_size==.003 and model.config.max_res==8192 and model.config.log2_hashmap_size==21
    assert len(pipe.datamanager.train_dataset)==63 and len(pipe.datamanager.eval_dataset)==3
    out=pipe.datamanager.train_dataparser_outputs
    torch.testing.assert_close(out.dataparser_transform,torch.eye(4)[:3],rtol=0,atol=1e-7)
    assert out.dataparser_scale==1.
    record=dict(time=time.time(),pid=os.getpid(),variant=variant,radius=pipe.config.final_grid_radius,
        bounds={name:value.cpu().tolist()for name,value in arrays.items()},
        no_new_nn_parameters=True,appearance_embedding_dim=0,train=63,eval=3,
        auxiliary_gradient='True raw-log-density likelihood; native RGB/LOS trunc_exp surrogate unchanged')
    (Path(cfg.output_dir)/'actual_bounds.json').write_text(json.dumps(record,indent=2)+'\n')


def train(path):
    import torch
    import nerfstudio.scripts.train as native
    cfg=load(path)
    def loop(local_rank,world_size,config,global_rank=0):
        native._set_random_seed(config.machine.seed+global_rank)
        trainer=config.setup(local_rank=local_rank,world_size=world_size);trainer.setup();validate(trainer)
        save=trainer.save_checkpoint
        def save_and_bind(step):
            save(step)
            trainer.pipeline.export_grid_for_checkpoint(trainer.checkpoint_dir/('step-%09d.ckpt'%step),step)
        trainer.save_checkpoint=save_and_bind
        original=trainer.train_iteration;previous=[0]
        def observed(step):
            result=original(step)
            if step<64 or step%2000==0:
                counts=[int(state['step'])for optimizer in trainer.optimizers.optimizers.values()
                        for state in optimizer.state.values()if 'step'in state]
                count=max(counts)if counts else 0
                row=dict(step=step,adam_updates=count,cumulative_skipped_attempts=step+1-count,
                    scale=float(trainer.grad_scaler.get_scale()),scalar_finite=bool(torch.isfinite(result[0].detach())))
                if step in(31,63)or step%2000==0:
                    row['weights_finite']=all(bool(torch.isfinite(p).all())for p in trainer.pipeline.model.parameters())
                folder=Path(config.output_dir)
                with(folder/'amp_optimizer_trace.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                if previous[0]==0 and count:(folder/'first_adam_update.json').write_text(json.dumps(row,indent=2)+'\n')
                previous[0]=count
            return result
        trainer.train_iteration=observed
        trainer.train()
    native.train_loop=loop;native.main(cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controller','train'])
    parser.add_argument('--config',type=Path,required=True);args,rest=parser.parse_known_args()
    if args.mode=='train':train(args.config)
    else:
        import run_bounded_ngp_quiet as quiet
        import run_marshawn_native_ngp as native
        sys.argv=[sys.argv[0]]+rest
        quiet.train_command=lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(args.config)]
        quiet.best_eval_checkpoint=native.select
        os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
        raise SystemExit(quiet.main())
