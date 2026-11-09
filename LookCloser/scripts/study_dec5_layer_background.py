#!/usr/bin/env python3
"""Layer compositor ablation: independent train-only background visibility.

Foreground keeps the original eight-view source selector. Background searches
all 59 train cameras using original stereo visibility, not neural head depth.
No target RGB, generated RGB, new geometry, or RGB averaging is introduced.
"""
from pathlib import Path
import json,sys
import numpy as np
import torch
import study_dec5_hair_geometry as base
from run_dec5_hair_campaign import atomic_json,sha


class LayerRenderer(base.Renderer):
    def __init__(self,scene):
        super().__init__(scene)
        self.background_observed=[torch.as_tensor(d,device='cuda')[None] for d in scene.observed_depth]

    @torch.inference_mode()
    def warp_layer(self,camera,depth,foreground=False,decontaminate=False):
        if foreground:return super().warp_layer(camera,depth,foreground,decontaminate)
        h,w=depth.shape;device=depth.device
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        xyz=base.tensor_unproject(torch.stack((xx.flatten()+.5,yy.flatten()+.5),-1),depth.flatten(),camera)
        center=getattr(self.scene,'source_reference_center',base.cv_pose(camera)[:3,3])
        order=np.argsort(np.linalg.norm(self.scene.centers-center,axis=-1),kind='stable')
        rgb=torch.zeros((h*w,3),device=device);ids=torch.full((h*w,),-1,device=device,dtype=torch.int16)
        support=torch.zeros(h*w,device=device,dtype=torch.int16)
        for begin in range(0,h*w,65536):
            end=min(begin+65536,h*w)
            for index in order:
                source=self.scene.cameras[index];uv,z=base.tensor_project(xyz[begin:end],source)
                observed=base.sample(self.background_observed[index],uv)[:,0]
                valid=((depth.flatten()[begin:end]>0)&(z>0)&(observed>0)&((observed-z).abs()<=.01*z+.002)
                       &(uv[:,0]>=.5)&(uv[:,0]<=source['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=source['h']-.5))
                sem=base.sample(self.semantic[index],uv);valid&=sem[:,2]>.8
                native_uv=uv*uv.new_tensor([self.rgb[index].shape[2]/source['w'],self.rgb[index].shape[1]/source['h']])
                colors=base.sample(self.rgb[index],native_uv)
                support[begin:end]+=valid.to(torch.int16)
                take=valid&(ids[begin:end]<0);rgb[begin:end][take]=colors[take];ids[begin:end][take]=int(index)
        return dict(rgb=rgb.reshape(h,w,3),alpha=(ids>=0).float().reshape(h,w),source_id=ids.reshape(h,w),support=support.reshape(h,w))


def main():
    if len(sys.argv)<2 or sys.argv[1]!='render':raise ValueError('Layer-only render experiment')
    if '--layer-style' not in sys.argv or sys.argv[sys.argv.index('--layer-style')+1]!='fb':raise ValueError('Use the fixed F/B decontamination ablation')
    output=Path(sys.argv[sys.argv.index('--output')+1]);parent=Path(sys.argv[sys.argv.index('--baseline')+1])
    base.Renderer=LayerRenderer;base.main()
    receipt=json.loads((output/'complete.json').read_text());audits=[]
    for row in receipt['records']:
        name=row['camera'];p=output/(name+'.npz')
        with np.load(p) as values:a={k:values[k] for k in values.files}
        with np.load(parent/(name+'.npz')) as values:b={k:values[k] for k in values.files}
        # The parent is the already-frozen neural receiver; layer compositing
        # must retain its edit domain rather than applying a second skin gate.
        a['roi']=b['roi'].copy();gate=a['layer_gate']
        for key in ['front_depth','back_depth','projected_hair_depth']:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        np.testing.assert_array_equal(a['rgb'][~gate],b['rgb'][~gate])
        np.savez_compressed(p,**a);row['npz_sha256']=sha(p)
        audits.append(dict(camera=name,layer_pixels=int(gate.sum()),soft_alpha_pixels=int((b['roi']&(a['layer_alpha']>.001)&(a['layer_alpha']<.995)).sum()),
            background_supported_pixels=int((b['roi']&(a['background_source_id']>=0)).sum())))
    receipt.update(independent_background_sources=True,background_source_pool=59,foreground_source_pool=8,
        background_visibility='original train-only stereo nearest surface; semantic background > .8',
        source_selector_implementation_sha256=sha(__file__),frozen_parent_geometry_and_roi=True,
        exact_rgb_outside_layer_gate=True,layer_audits=audits,target_rgb_used=False)
    atomic_json(output/'complete.json',receipt)

if __name__=='__main__':main()
