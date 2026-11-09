#!/usr/bin/env python3
"""Experimental confidence compositing across fixed ordered GT sources.

Plateau confidence preserves sharp first-source RGB for good correspondences.
At the depth-consistency boundary, a smooth stick-breaking transition replaces
the abrupt source/fallback switch. Geometry and source ordering stay unchanged.
This changes appearance, not physical depth or a semantic output mask.
"""
from pathlib import Path
import json,sys
import cv2,numpy as np,torch
import study_dec5_hair_geometry as geometry
from run_dec5_hair_campaign import atomic_json,sha


def confidence(residual_ratio):
    t=((residual_ratio-.5)/1.).clamp(0,1)
    return 1-t*t*(3-2*t)


@torch.inference_mode()
def warp(self,camera,depth):
    h,w=depth.shape;device=depth.device
    y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    uv_target=torch.stack((x.flatten()+.5,y.flatten()+.5),-1)
    xyz=geometry.tensor_unproject(uv_target,depth.flatten(),camera);order=self.source_order(camera)
    rgb=torch.zeros((h*w,3),device=device);pure_rgb=torch.zeros_like(rgb);pure=torch.zeros(h*w,device=device,dtype=torch.bool)
    ids=torch.full((h*w,),-1,device=device,dtype=torch.int16)
    support=torch.zeros(h*w,device=device,dtype=torch.int16);hair=torch.zeros(h*w,device=device)
    skin=torch.zeros_like(hair);error_sum=torch.zeros_like(hair);total=torch.zeros_like(hair)
    for begin in range(0,h*w,65536):
        end=min(begin+65536,h*w);size=end-begin
        remaining=torch.ones(size,device=device);largest=torch.zeros(size,device=device)
        valid_target=depth.flatten()[begin:end]>0
        for index in order:
            source=self.scene.cameras[index];uv,z=geometry.tensor_project(xyz[begin:end],source)
            sd=geometry.sample(self.depth[index],uv,mode=getattr(self,'visibility_depth_sampling','bilinear'))[:,0]
            valid=valid_target&(z>0)&(sd>0)&(uv[:,0]>=.5)&(uv[:,0]<=source['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=source['h']-.5)
            ratio=(sd-z).abs()/(.01*z+.002).clamp_min(1e-12)
            alpha=torch.where(valid,confidence(ratio),torch.zeros_like(z));weight=remaining*alpha;remaining*=1-alpha
            native_uv=uv*uv.new_tensor([self.rgb[index].shape[2]/source['w'],self.rgb[index].shape[1]/source['h']])
            color=geometry.sample(self.rgb[index],native_uv);sem=geometry.sample(self.semantic[index],uv)
            scaled=color.clamp_min(0)*10.570320292648677;bounded=scaled/(1+scaled)
            rgb[begin:end]+=weight[:,None]*bounded;hair[begin:end]+=weight*sem[:,0];skin[begin:end]+=weight*sem[:,1]
            exact=weight==1;pure_rgb[begin:end][exact]=color[exact];pure[begin:end]|=exact
            back,_=geometry.tensor_project(geometry.tensor_unproject(uv,sd,source),camera)
            error_sum[begin:end]+=weight*torch.linalg.norm(back-uv_target[begin:end],dim=-1).clamp_max(20)
            support[begin:end]+=(alpha>.001).to(torch.int16)
            take=(weight>largest)&(weight>0);ids[begin:end][take]=index;largest=torch.maximum(largest,weight)
        total[begin:end]=1-remaining
    norm=total.clamp_min(1e-12)
    # Benchmark expects unpremultiplied admitted source RGB. The wrapper below
    # applies the same explicit confidence to the common frozen fallback.
    linear=(rgb/norm[:,None]).clamp(0,1);mixed=linear/(1-linear).clamp_min(1e-6)/10.570320292648677
    mixed[pure]=pure_rgb[pure]
    return dict(rgb=mixed.reshape(h,w,3),source_id=ids.reshape(h,w),support=support.reshape(h,w),
        hair_probability=(hair/norm).reshape(h,w),skin_probability=(skin/norm).reshape(h,w),
        reprojection_error=torch.where(total>0,error_sum/norm,20.).reshape(h,w),
        gt_confidence=total.reshape(h,w))


def main():
    args=sys.argv[1:];output=Path(args[args.index('--output')+1]);fallback=Path(args[args.index('--fallback')+1])
    geometry.Renderer.warp=warp
    import render_dec5_native_head_benchmark as benchmark
    benchmark.main();receipt=json.loads((output/'complete.json').read_text())
    original_receipt=sha(output/'complete.json')
    for record in receipt['records']:
        name=record['camera'];path=output/(name+'.npz');arrays=dict(np.load(path));base=np.load(fallback/(name+'.npz'))
        alpha=arrays['gt_confidence'];roi=arrays['roi'];background=np.full((*roi.shape,3),.5,np.float32)
        background[roi]=base['rgb_display'][roi]
        background_linear=np.where(background<=.04045,background/12.92,((background+.055)/1.055)**2.4)
        original=arrays['rgb'];scaled=np.maximum(original,0)*10.570320292648677
        linear=scaled/(1+scaled)*alpha[...,None]+background_linear*(1-alpha[...,None])
        arrays['rgb']=linear/np.maximum(1-linear,1e-6)/10.570320292648677
        arrays['rgb'][alpha>=1]=original[alpha>=1]
        arrays['neural_fallback']=roi&(alpha<1)
        np.savez_compressed(path,**arrays);image=np.rint(geometry.display(arrays['rgb'])*255).astype(np.uint8)
        png=output/(name+'.png');cv2.imwrite(str(png),image[...,::-1])
        cv2.imwrite(str(output/(name+'_review.jpg')),cv2.resize(np.rot90(image),(360,640),interpolation=cv2.INTER_AREA)[...,::-1])
        record.update(npz_sha256=sha(path),png_sha256=sha(png),fractional_gt_confidence_pixels=int(((alpha>0)&(alpha<1)&roi).sum()))
    receipt.update(soft_depth_consistency=True,confidence_ratio_plateau=.5,confidence_ratio_zero=1.5,
        method='native train-JPEG GT transfer; smooth depth-confidence compositing; common frozen fallback',
        source_selection='same nearest-eight order; smooth confidence stick breaking; RGB mixing only at uncertain correspondences',
        compositing='ordered confidence stick breaking plus identical frozen fallback',
        mixing_domain='bounded linear-sRGB; plateau source RGB bit-identical',
        geometry_changed=False,source_order_changed=False,semantic_output_mask=False,
        implementation_sha256=sha(__file__),precomposite_receipt_sha256=original_receipt)
    atomic_json(output/'complete.json',receipt)


if __name__=='__main__':main()
