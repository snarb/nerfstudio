#!/usr/bin/env python3
"""Seal a separate background-volume prediction before opening held-out RGB."""
from pathlib import Path
import argparse,time
import cv2
import numpy as np
import torch
from run_dec5_sequence import read,write,alive
from dec5_sequence_worker import sha
from train_dec5_neural_background import install_empty_subject_box


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--step',type=int,default=19999)
    a=p.parse_args();r=a.root;training=r/'neural_background';checkpoint=training/f'step-{a.step:09d}.ckpt'
    deadline=time.monotonic()+7200
    while not checkpoint.exists() or (read(training/'latest.json',{}).get('step',-1)<a.step):
        if not alive(read(training/'worker.json')):raise RuntimeError('Background worker is terminal before requested checkpoint')
        if time.monotonic()>deadline:raise TimeoutError('Background checkpoint observation timed out; training is not declared failed')
        time.sleep(10)
    output=training/f'evaluation_{a.step:06d}';output.mkdir(exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory)
    data=r/'frame/semantic_data';meta=read(data/'transforms.json')
    f=next(f for f in meta['frames'] if f['file_path'] in meta['val_filenames']);h,w=f['h'],f['w']
    fallback_path=r/'fallback4k'/(f['physical_camera']+'.npz')
    fallback=cv2.resize(np.load(fallback_path)['rgb_display'],(w,h),interpolation=cv2.INTER_AREA)
    prediction=output/'prediction.npz'
    if not (output/'prediction_complete.json').exists():
        from nerfstudio.utils.eval_utils import eval_setup
        def config(config):config.eval_checkpoint=checkpoint;return config
        cfg_path=next((training/'training').rglob('config.yml'))
        _,pipe,loaded,step=eval_setup(cfg_path,eval_num_rays_per_chunk=4096,test_mode='inference',update_config_callback=config)
        install_empty_subject_box(pipe.model.field)
        pipe.model.requires_grad_(False)
        camera=pipe.datamanager.eval_dataset.cameras[0:1].to(pipe.device)
        with torch.no_grad():values=pipe.model.get_outputs_for_camera_ray_bundle(camera.generate_rays(0))
        rgb=values['rgb'].cpu().numpy();alpha=values['accumulation'].cpu().numpy()
        if alpha.ndim==2:alpha=alpha[...,None]
        hybrid=rgb+(1-alpha).clip(0,1)*fallback
        if rgb.shape!=(h,w,3) or not np.isfinite(hybrid).all():raise ValueError('Invalid background volume RGB')
        np.savez_compressed(prediction,rgb_display=hybrid,neural_rgb=rgb,opacity=alpha,depth=values['depth'].cpu().numpy())
        write(output/'prediction_complete.json',dict(step=step,checkpoint_sha256=sha(loaded),
            prediction_sha256=sha(prediction),fallback_sha256=sha(fallback_path),
            source_request_sha256=sha(training/'request.json'),uses_validation_rgb=False,uses_test_rgb=False,
            method='separate background volume premultiplied RGB plus residual frozen ray-background RGB'))
    receipt=read(output/'prediction_complete.json')
    if receipt['prediction_sha256']!=sha(prediction) or receipt['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('Sealed background artifacts changed')
    # Target RGB and reference semantics are evaluation inputs only.
    gt=cv2.imread(str(data/f['file_path']))[...,::-1].astype(np.float32)/255
    with np.load(prediction) as saved:hybrid=saved['rgb_display'];alpha=saved['opacity'].squeeze()
    probability=np.load(r/'eval_reference_hd.npz')['probability'].astype(np.float32)/65535
    if probability.shape!=gt.shape:raise ValueError('Reference raster differs')
    known=probability[...,2]>.98;hair=probability[...,0]>.5
    masks=dict(trusted_background=cv2.erode(known.astype(np.uint8),np.ones((33,33),np.uint8))>0,
        hair_adjacent_background=known&(cv2.dilate(hair.astype(np.uint8),np.ones((65,65),np.uint8))>0))
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.functional.image import structural_similarity_index_measure
    metric=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval();scores={}
    for region,mask in masks.items():
        y,x=np.nonzero(mask)
        if len(x)<100:raise ValueError('Insufficient background metric domain')
        yy=slice(max(0,y.min()-4),min(h,y.max()+5));xx=slice(max(0,x.min()-4),min(w,x.max()+5))
        scores[region]={}
        for name,image in [('ray',fallback),('volume_hybrid',hybrid)]:
            target=gt[yy,xx];pred=np.where(mask[yy,xx,None],image[yy,xx],target)
            t,g=[torch.as_tensor(np.ascontiguousarray(v),device='cuda').permute(2,0,1)[None] for v in [pred,target]]
            with torch.no_grad():lp=float(metric(t,g));ss=float(structural_similarity_index_measure(t,g,data_range=1.))
            scores[region][name]=dict(psnr=float(-10*np.log10(max(float(np.square(image[mask]-gt[mask]).mean()),1e-12))),
                ssim=ss,lpips=lp,pixels=len(x),mean_background_volume_opacity=float(alpha[mask].mean()))
    panels=[cv2.resize(np.rot90(np.rint(np.clip(v,0,1)*255).astype(np.uint8)),(360,720),interpolation=cv2.INTER_AREA) for v in [gt,fallback,hybrid]]
    cv2.imwrite(str(output/'gt_ray_volume.jpg'),np.concatenate(panels,axis=1)[...,::-1])
    write(output/'metrics.json',dict(regions=scores,step=a.step,background_component_only=True,
        semantic_reference_for_metrics_only=True,hidden_background_has_no_gt=True,
        foreground_improvement_established=False,production_promoted=False,prediction_receipt=receipt))


if __name__=='__main__':main()
