#!/usr/bin/env python3
"""Frozen-field eval throughput and batch-independence gate; no fitting."""
from pathlib import Path
import argparse,time
import numpy as np
import torch,yaml
from dec5_background_separation import attach_inference
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['frame-root','training','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--camera-scale',type=float,default=1.)
    a=p.parse_args();torch.set_num_threads(2)
    configs=list(a.training.rglob('config.yml'))
    if len(configs)!=1:raise ValueError('One frozen config required')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True)
    pipe.eval();pipe.model.current_train_step=state['step'];del state
    provider=attach_inference(pipe,a.frame_root/'ray_background/background.pt')
    camera=pipe.datamanager.eval_dataset.cameras[0:1].to('cuda')
    if not .5<=a.camera_scale<=4:raise ValueError('Bounded camera scaling required')
    camera.rescale_output_resolution(a.camera_scale);h,w=int(camera.height[0]),int(camera.width[0])
    y,x=torch.meshgrid(torch.arange(0,h,4,device='cuda'),torch.arange(0,w,4,device='cuda'),indexing='ij')
    rays=camera.generate_rays(0,coords=torch.stack((y,x),-1).float()+.5)
    measurements=[];reference=None;checkpoint_digest=sha(a.checkpoint)
    for chunk in [1024,4096,8192,1024,8192,4096]:
        pipe.model.config.eval_num_rays_per_chunk=chunk
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.monotonic()
        with torch.no_grad():outputs=pipe.model.get_outputs_for_camera_ray_bundle(rays)
        torch.cuda.synchronize();elapsed=time.monotonic()-began
        values={k:outputs[k].detach().float().cpu().numpy() for k in ['rgb','depth','accumulation']}
        if reference is None:reference=values
        differences={k:float(np.max(np.abs(values[k]-reference[k]))) for k in values}
        opacity=values['accumulation'].squeeze();reference_opacity=reference['accumulation'].squeeze()
        supported=(opacity>=.001)&(reference_opacity>=.001)
        depth_error=float(np.max(np.abs(values['depth'].squeeze()-reference['depth'].squeeze())[supported])) if supported.any() else 0.
        crossings={str(t):int(((opacity>=t)!=(reference_opacity>=t)).sum()) for t in [.001,.01,.1,.9]}
        passed=differences['rgb']<=1e-5 and differences['accumulation']<=1e-5 and depth_error<=1e-5 and not any(crossings.values())
        measurements.append(dict(chunk=chunk,seconds=elapsed,rays=y.numel(),rays_per_second=y.numel()/elapsed,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),max_absolute_differences=differences,
            supported_depth_max_absolute_difference=depth_error,opacity_gate_crossings=crossings,passes=passed))
        write(a.output,dict(measurements=measurements,checkpoint_sha256=checkpoint_digest,fitting=False,
            comparison='same frozen rays, RGB/opacity everywhere, depth only on joint opacity >=.001; tolerance1e-5; no gate crossings',finished=False))
    provider.verify()
    summaries={str(chunk):dict(median_seconds=float(np.median([r['seconds'] for r in measurements if r['chunk']==chunk])),
        passes=all(r['passes'] for r in measurements if r['chunk']==chunk)) for chunk in [1024,4096,8192]}
    write(a.output,dict(measurements=measurements,summary=summaries,checkpoint_sha256=checkpoint_digest,
        camera_raster=[w,h],probe_pixel_stride=4,depth_support_minimum_opacity=.001,zero_gate_crossings_required=True,
        fitting=False,finished=True,concurrent_training_may_affect_timing=True,reference_tolerance=1e-5))


if __name__=='__main__':main()
