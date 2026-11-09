#!/usr/bin/env python3
"""Held-out background-component gate; source geometry/textures remain frozen."""
from pathlib import Path
import argparse,time
import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure
from dec5_plane_background import PlaneBackground
from dec5_hair import cv_pose
from dec5_sequence_worker import sha
from run_dec5_sequence import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    geometry=r/'plane_background';deadline=time.monotonic()+1200
    while not (geometry/'complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Background geometry did not seal')
        time.sleep(10)
    output=r/'plane_background_evaluation';output.mkdir(exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory)
    data=r/'frame/semantic_data';meta=read(data/'transforms.json')
    heldout=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(heldout)!=1:raise ValueError('One excluded camera required')
    f=heldout[0];name=f['physical_camera'];h,w=f['h'],f['w']
    fallback_path=r/'fallback4k'/(name+'.npz')
    fallback=cv2.resize(np.load(fallback_path)['rgb_display'],(w,h),interpolation=cv2.INTER_AREA)
    prediction=output/'prediction.npz'
    if not (output/'prediction_complete.json').exists():
        model=PlaneBackground(data,r/'frame/semantics',geometry)
        pose=torch.as_tensor(cv_pose(f),device='cuda',dtype=torch.float32)
        colors=fallback.copy().reshape(-1,3);valid=np.zeros(h*w,bool)
        depths=np.zeros(h*w,np.float32);support=np.zeros(h*w,np.int16)
        for begin in range(0,h*w,8192):
            ids=torch.arange(begin,min(begin+8192,h*w),device='cuda')
            x=(ids%w+.5-f['cx'])/f['fl_x'];y=(ids//w+.5-f['cy'])/f['fl_y']
            directions=torch.stack([x,y,torch.ones_like(x)],-1)@pose[:3,:3].T
            rgb,good,depth,count=model.predict(pose[:3,3].expand(len(ids),3),directions)
            ii=ids[good].cpu().numpy();colors[ii]=rgb[good].cpu().numpy()
            valid[ii]=True;depths[ii]=depth[good].cpu().numpy();support[ii]=count[good].cpu().numpy()
            write(output/'progress.json',dict(phase='render',completed=begin+len(ids),total=h*w))
        colors=colors.reshape(h,w,3);valid=valid.reshape(h,w)
        np.savez_compressed(prediction,rgb_display=colors,valid=valid,depth=depths.reshape(h,w),support=support.reshape(h,w))
        write(output/'prediction_complete.json',dict(camera=name,prediction_sha256=sha(prediction),
            geometry_sha256=sha(geometry/'complete.json'),fallback_sha256=sha(fallback_path),
            source_count=len(model.frames),uses_validation_rgb=False,uses_test_rgb=False,
            method='nearest supported plane; >=3 trusted-background source colors; median absolute dispersion<=.05; original ray BG fallback',
            depth_convention='OpenCV camera-z in calibration units; background hypotheses only'))
    receipt=read(output/'prediction_complete.json')
    if sha(prediction)!=receipt['prediction_sha256'] or sha(geometry/'complete.json')!=receipt['geometry_sha256']:
        raise ValueError('Background prediction changed')
    # Evaluation RGB/semantics are first opened after candidate sealing.
    gt=cv2.imread(str(data/f['file_path']))[...,::-1].astype(np.float32)/255
    with np.load(prediction) as saved:colors=saved['rgb_display'];valid=saved['valid']
    probability=np.load(r/'eval_reference_hd.npz')['probability'].astype(np.float32)/65535
    if probability.shape!=gt.shape:raise ValueError('Metric reference raster differs')
    hair=probability[...,0]>.5;known=probability[...,2]>.98
    masks=dict(trusted_background=cv2.erode(known.astype(np.uint8),np.ones((33,33),np.uint8))>0,
        hair_adjacent_background=known&(cv2.dilate(hair.astype(np.uint8),np.ones((65,65),np.uint8))>0))
    metric=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval();scores={}
    for label,mask in masks.items():
        y,x=np.nonzero(mask)
        if len(x)<100:raise ValueError('Insufficient background ROI')
        yy=slice(max(0,y.min()-4),min(h,y.max()+5));xx=slice(max(0,x.min()-4),min(w,x.max()+5))
        scores[label]={}
        for arm,image in [('ray',fallback),('planes',colors)]:
            target=gt[yy,xx];pred=np.where(mask[yy,xx,None],image[yy,xx],target)
            t,g=[torch.as_tensor(np.ascontiguousarray(v),device='cuda').permute(2,0,1)[None] for v in [pred,target]]
            with torch.no_grad():lp=float(metric(t,g));ss=float(structural_similarity_index_measure(t,g,data_range=1.))
            scores[label][arm]=dict(psnr=float(-10*np.log10(max(float(np.square(image[mask]-gt[mask]).mean()),1e-12))),
                ssim=ss,lpips=lp,pixels=len(x),plane_coverage=float(valid[mask].mean()))
    panels=[cv2.resize(np.rot90(np.rint(v*255).astype(np.uint8)),(360,720),interpolation=cv2.INTER_AREA) for v in [gt,fallback,colors]]
    cv2.imwrite(str(output/'gt_ray_planes.jpg'),np.concatenate(panels,axis=1)[...,::-1])
    write(output/'metrics.json',dict(regions=scores,analysis_raster=[w,h],
        all_pixel_plane_coverage=float(valid.mean()),hair_proxy_plane_coverage=float(valid[hair].mean()),
        background_component_only=True,hidden_background_has_no_gt=True,
        semantic_reference_for_metrics_only=True,production_promoted=False,prediction_receipt=receipt))


if __name__=='__main__':main()
