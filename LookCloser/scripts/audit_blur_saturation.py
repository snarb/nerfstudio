"""Measure opacity-normalized learned color on fixed, valid training rays."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from blur_runtime import write, sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoints',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--color-gradient',action='store_true',help='Measure the derivative of summed RGB with respect to the color head on 64 fixed rays')
    p.add_argument('--snapshot-dir',type=Path,help='Save frozen RGB/opacity/depth tensors for implementation parity checks')
    args=p.parse_args();torch.set_num_threads(2);results=[]
    for path in args.checkpoints:
        checkpoint_sha=sha(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        if sha(path)!=checkpoint_sha:raise RuntimeError('Checkpoint changed while loading; pin a snapshot first')
        pipe=state['config'].pipeline.setup(device='cuda');pipe.load_pipeline(state['pipeline'],state['step']);pipe.eval()
        data=pipe.datamanager.train_dataset[0]
        valid=data.get('mask',torch.ones_like(data['image'][...,:1],dtype=torch.bool))[...,0].bool()
        coords=valid.nonzero();index=torch.linspace(0,len(coords)-1,1024).long();coords=coords[index].cuda().float()+.5
        rays=pipe.datamanager.train_dataset.cameras[0:1].to('cuda').generate_rays(0,coords=coords)
        out=pipe.model(rays);opacity=out['accumulation'].float()
        if args.snapshot_dir:
            args.snapshot_dir.mkdir(parents=True,exist_ok=True)
            torch.save({k:out[k].detach().cpu() for k in ('rgb','accumulation','depth')},
                       args.snapshot_dir/(path.parent.name+'.pt'))
        effective_rgb=out['rgb'].float()/opacity.clamp_min(1e-8)
        results.append(dict(checkpoint=str(path),checkpoint_sha256=checkpoint_sha,step=state['step'],
            mean_opacity=float(opacity.mean()),mean_effective_rgb=effective_rgb.mean(0).cpu().tolist(),
            channel_saturation_fraction=(effective_rgb>.99).float().mean(0).cpu().tolist(),
            white_saturation_fraction=float((effective_rgb.min(-1).values>.99).float().mean()),
            mean_effective_chroma=float((effective_rgb.max(-1).values-effective_rgb.min(-1).values).mean())))
        if args.color_gradient:
            with torch.enable_grad():
                differentiable=pipe.model(rays[:64])
                gradients=torch.autograd.grad(differentiable['rgb'].float().sum(),
                                              tuple(pipe.model.field.mlp_color.parameters()))
                flat=torch.cat([g.detach().float().reshape(-1) for g in gradients])
                if not torch.isfinite(flat).all():raise FloatingPointError('Nonfinite color-head gradient')
                results[-1]['rgb_sum_color_gradient_l2']=float(flat.norm())
                results[-1]['rgb_sum_color_gradient_nonzero_fraction']=float((flat!=0).float().mean())
            del differentiable,gradients,flat
        del pipe,state,out;torch.cuda.empty_cache()
    write(args.output,results);print(json.dumps(results,indent=2))


if __name__=='__main__':main()
