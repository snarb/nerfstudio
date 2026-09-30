"""Render the metric-selected checkpoint at a measured export integration setting."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import yaml
from blur_runtime import sha,write
from run_luster_experiment import evaluate


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--coarse-step',type=float,default=.0005);p.add_argument('--max-samples',type=int,default=4096)
    p.add_argument('--hull-margin-voxels',type=float,help='Optional conservative train-hull occupancy envelope; original field remains unchanged')
    args=p.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise ValueError('Export destination must be new')
    if args.coarse_step<=0 or args.max_samples<=0:raise ValueError('Integration settings must be positive')
    selected=json.loads((args.run/'selection.json').read_text());request=json.loads((args.run/'request.json').read_text())
    cfg=yaml.load(Path(selected['config']).read_text(),Loader=yaml.Loader)
    cfg.pipeline.model.adaptive_coarse_step_size=args.coarse_step
    cfg.pipeline.model.max_steps_per_ray=args.max_samples
    cfg.load_checkpoint=Path(selected['checkpoint']);cfg.load_dir=None
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    pipe=cfg.pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval();del state
    args.output.mkdir(parents=True)
    guard=None
    if args.hull_margin_voxels is not None:
        from luster_render_guard import build_guard
        guard=build_guard(pipe,request['data'],args.output/'occupancy_guard.npz',args.hull_margin_voxels)
    statistics=[];original=pipe.model.get_outputs_for_camera_ray_bundle
    def render(rays):
        outputs=original(rays)
        if 'num_samples_per_ray' in outputs:
            samples=outputs['num_samples_per_ray'].float();maximum=int(samples.max())
            statistics.append(dict(mean_samples=float(samples.mean()),max_samples=maximum,cap_fraction=float((samples>=args.max_samples).float().mean())))
            if maximum>=args.max_samples:raise RuntimeError('Export integration saturates its sample cap; increase the cap and use a new destination')
        return outputs
    pipe.model.get_outputs_for_camera_ray_bundle=render
    (args.output/'config.yml').write_text(yaml.dump(cfg))
    request.update(output=str(args.output),eval_stride=1,render_only=True)
    write(args.output/'request.json',request)
    result=evaluate(SimpleNamespace(pipeline=pipe),request,selected['step'])
    result.update(checkpoint=selected['checkpoint'],config=str(args.output/'config.yml'),render_dir=str(args.output/f'eval_{selected["step"]:06d}'),
                  selected_by=dict(protocol='Training all-eval PSNR, LPIPS tie-break within0.07dB',source=str(args.run/'selection.json'),
                                   **{k:selected[k] for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))
    for view,stats in zip(result['per_view'],statistics):stats['image']=view['image']
    if guard is not None:result['occupancy_guard']=guard
    write(args.output/'selection.json',result)
    write(args.output/'complete.json',dict(checkpoint_sha256=sha(Path(selected['checkpoint'])),transforms_sha256=sha(Path(request['data'])/'transforms.json'),
                                         coarse_step=args.coarse_step,max_samples=args.max_samples,marching_statistics=statistics))
    print(json.dumps({k:v for k,v in result.items() if k!='per_view'},indent=2))


if __name__=='__main__':main()
