#!/usr/bin/env python3
"""Standalone train-consensus source-selection ablation on unchanged depth.

Retains the same eight cameras, z tests, semantic votes and missing pixels.
For the winning train-view semantic class, chooses the first geometrically
valid source of that class; no RGB averaging, fitted thresholds or target RGB.
"""
from pathlib import Path
import json,sys
import torch
import study_dec5_hair_geometry as base
from run_dec5_hair_campaign import atomic_json,sha


class SemanticRenderer(base.Renderer):
    @torch.inference_mode()
    def warp(self,camera,depth):
        h,w=depth.shape;device=depth.device
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        uv0=torch.stack((xx.flatten()+.5,yy.flatten()+.5),-1)
        xyz=base.tensor_unproject(uv0,depth.flatten(),camera)
        order=self.source_order(camera)
        rgb=torch.zeros((h*w,3),device=device);ids=torch.full((h*w,),-1,device=device,dtype=torch.int16)
        support=torch.zeros(h*w,device=device,dtype=torch.int16);votes=torch.zeros((h*w,3),device=device)
        reprojection=torch.zeros(h*w,device=device)
        for begin in range(0,h*w,65536):
            end=min(begin+65536,h*w);n=end-begin
            class_rgb=torch.zeros((n,3,3),device=device);class_ids=torch.full((n,3),-1,device=device,dtype=torch.int16)
            for index in order:
                source=self.scene.cameras[index];uv,z=base.tensor_project(xyz[begin:end],source)
                sd=base.sample(self.depth[index],uv)[:,0]
                valid=((depth.flatten()[begin:end]>0)&(z>0)&(sd>0)&(uv[:,0]>=.5)&(uv[:,0]<=source['w']-.5)
                       &(uv[:,1]>=.5)&(uv[:,1]<=source['h']-.5)&((sd-z).abs()<=.01*z+.002))
                sem=base.sample(self.semantic[index],uv)
                support[begin:end]+=valid.to(torch.int16);votes[begin:end]+=valid[:,None]*sem
                back_uv,_=base.tensor_project(base.tensor_unproject(uv,sd,source),camera)
                error=torch.linalg.norm(back_uv-uv0[begin:end],dim=-1).clamp_max(20)
                reprojection[begin:end]+=torch.where(valid,error,0.)
                native_uv=uv*uv.new_tensor([self.rgb[index].shape[2]/source['w'],self.rgb[index].shape[1]/source['h']])
                colors=base.sample(self.rgb[index],native_uv)
                take=valid&(ids[begin:end]<0);rgb[begin:end][take]=colors[take];ids[begin:end][take]=int(index)
                labels=sem.argmax(-1)
                for label in range(3):
                    take=valid&(labels==label)&(class_ids[:,label]<0)
                    class_rgb[take,label]=colors[take];class_ids[take,label]=int(index)
            desired=votes[begin:end].argmax(-1);row=torch.arange(n,device=device)
            selected=class_ids[row,desired];take=selected>=0
            rgb[begin:end][take]=class_rgb[row,desired][take];ids[begin:end][take]=selected[take]
        return dict(rgb=rgb.reshape(h,w,3),source_id=ids.reshape(h,w),support=support.reshape(h,w),
            hair_probability=(votes[:,0]/support.clamp_min(1)).reshape(h,w),
            skin_probability=(votes[:,1]/support.clamp_min(1)).reshape(h,w),
            reprojection_error=torch.where(support>0,reprojection/support.clamp_min(1),20.).reshape(h,w))


def main():
    if len(sys.argv)<2 or sys.argv[1]!='render':raise ValueError('This standalone ablation only renders')
    output=Path(sys.argv[sys.argv.index('--output')+1])
    base.Renderer=SemanticRenderer;base.main()
    receipt=json.loads((output/'complete.json').read_text())
    receipt.update(source_semantic_consensus=True,source_selector_implementation_sha256=sha(__file__),
        source_selector='first geometrically valid source matching the multiview majority semantic class',
        target_rgb_used_for_selection=False,rgb_averaging=False)
    atomic_json(output/'complete.json',receipt)

if __name__=='__main__':main()
