#!/usr/bin/env python3
"""ROI metrics for sealed one-camera HD preflight/continuation predictions."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['frame-root','training','reference','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--include-person-roi',action='store_true')
    p.add_argument('--resize-evaluation-reference',action='store_true',help='Explicitly resample the metrics-only semantic reference to the prediction raster')
    a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    meta=read(a.frame_root/'data/transforms.json')
    frames=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(frames)!=1:raise ValueError('One held-out camera required')
    gt=cv2.imread(str(a.frame_root/'data'/frames[0]['file_path']))[...,::-1].astype(np.float32)/255
    probability=np.load(a.reference)['probability'].astype(np.float32)/65535
    reference_shape=list(probability.shape[:2]);scale=1.
    if gt.shape!=probability.shape:
        if not a.resize_evaluation_reference:raise ValueError('Reference raster differs from evaluation')
        scale=gt.shape[1]/probability.shape[1]
        if abs(scale-gt.shape[0]/probability.shape[0])>1e-6:raise ValueError('Reference aspect ratio differs')
        probability=cv2.resize(probability,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
    radius=max(1,round(5*scale));erosion=max(1,round(16*scale))
    hair=probability[...,0]>.5;face=probability[...,1]>.5
    boundary=cv2.dilate(hair.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)^cv2.erode(hair.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    bg=cv2.erode((probability[...,2]>.98).astype(np.uint8),np.ones((2*erosion+1,2*erosion+1),np.uint8))>0
    masks=dict(hair=hair,face_neck=face,boundary=boundary,trusted_background=bg)
    if a.include_person_roi:masks['clothing_hands_proxy']=(probability[...,2]<.5)&(probability[...,0]<.2)&(probability[...,1]<.2)
    metric=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval();rows=[]
    for row in read(a.training/'history.json',[]):
        archive=Path(row['renders'])/'0.npz';values=np.load(archive);pred=values['rgb'];alpha=values['accumulation'].squeeze()
        if pred.shape!=gt.shape or not np.isfinite(pred).all():raise ValueError('Invalid sealed RGB prediction')
        scores={}
        for name,mask in masks.items():
            y,x=np.nonzero(mask)
            if len(x)<100:raise ValueError('Insufficient reference pixels')
            yy=slice(max(0,y.min()-4),min(gt.shape[0],y.max()+5));xx=slice(max(0,x.min()-4),min(gt.shape[1],x.max()+5))
            target=gt[yy,xx];prediction=np.where(mask[yy,xx,None],pred[yy,xx],target)
            t,g=[torch.as_tensor(np.ascontiguousarray(v),device='cuda').permute(2,0,1)[None] for v in [prediction,target]]
            with torch.no_grad():lp=float(metric(t,g));ss=float(structural_similarity_index_measure(t,g,data_range=1.))
            scores[name]=dict(psnr=float(-10*np.log10(max(float(np.square(pred[mask]-gt[mask]).mean()),1e-12))),
                ssim=ss,lpips=lp,pixels=len(x),opacity_mean=float(alpha[mask].mean()),
                opacity_below_01_fraction=float((alpha[mask]<.1).mean()))
        rows.append(dict(step=row['step'],regions=scores,prediction_sha256=sha(archive)))
    write(a.output,dict(rows=rows,heldout_camera=frames[0]['physical_camera'],reference_sha256=sha(a.reference),
        metric_domain='Prediction raster; GT-composited masked tight crop for SSIM/LPIPS, PSNR on ROI pixels',
        reference_original_shape=reference_shape,evaluation_shape=list(gt.shape[:2]),reference_resized=scale!=1.,
        boundary_radius_pixels=radius,boundary_radius_hd_pixels=5,reference_used_for_metrics_only=True,fallible_semantics=True,
        checkpoint_selection_unchanged=True,no_independent_depth_ground_truth=True))


if __name__=='__main__':main()
