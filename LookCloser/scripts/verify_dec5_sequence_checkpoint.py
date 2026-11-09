#!/usr/bin/env python3
"""Reproduce sealed late-phase RGB with and without runtime restoration."""
from pathlib import Path
import argparse
import numpy as np,torch
from dec5_sequence_checkpoint import load
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['training','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    pipe,provider,provenance=load(a.training,checkpoint=a.checkpoint)
    row=next(r for r in read(a.training/'history.json') if r['step']==provenance['step'])
    reference=Path(row['renders'])/'0.npz';saved=dict(np.load(reference))
    camera=pipe.datamanager.eval_dataset.cameras[0:1].to('cuda');h,w=int(camera.height[0]),int(camera.width[0])
    y,x=torch.meshgrid(torch.arange(0,h,4,device='cuda'),torch.arange(0,w,4,device='cuda'),indexing='ij')
    rays=camera.generate_rays(0,coords=torch.stack((y,x),-1).float()+.5)
    results={};correct=pipe.model.field.feature_reweighting_strength
    for label,strength in [('initial_config_phase',1.),('restored_phase',correct)]:
        pipe.model.config.feature_reweighting_strength=strength;pipe.model.field.feature_reweighting_strength=strength
        with torch.no_grad():result=pipe.model.get_outputs_for_camera_ray_bundle(rays)
        prediction=result['rgb'].float().cpu().numpy();target=saved['rgb'][::4,::4]
        results[label]=dict(strength=strength,rgb_max_absolute_difference=float(np.abs(prediction-target).max()),
            rgb_mean_absolute_difference=float(np.abs(prediction-target).mean()))
    provider.verify()
    write(a.output,dict(provenance=provenance,comparisons=results,
        reference_prediction_sha256=sha(reference),fitting=False,
        roundtrip_passes=results['restored_phase']['rgb_max_absolute_difference']<=1e-5))


if __name__=='__main__':main()
