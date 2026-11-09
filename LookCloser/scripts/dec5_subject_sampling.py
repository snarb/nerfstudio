"""Compose train-only subject quotas with LookCloser's real FAS context sampler."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from run_dec5_hair_campaign import atomic_json,sha


class SubjectSampler:
    def __init__(self,sampler,dataset,semantic_dir,hair_fraction,person_fraction,warmup=1000,ramp=2000,boundary_width=11,expected_train_count=59,boundary_share=.3,frequency_weighted_hair=False):
        if min(hair_fraction,person_fraction)<0 or hair_fraction+person_fraction>1:raise ValueError('Invalid subject quotas')
        if boundary_width<1 or boundary_width%2!=1:raise ValueError('Boundary width must be positive and odd')
        self.boundary_width=boundary_width
        self.sampler=sampler;self.original=sampler.sample;self.hair_fraction=hair_fraction;self.person_fraction=person_fraction
        self.warmup=warmup;self.ramp=ramp;self.pools={};self.labels={};self.audit=dict(batches=0,hair=0,person_nonhair=0,background=0,uncertain=0,rays=0)
        self.audit_start_sample_count=int(getattr(sampler,'sample_count',0))
        if len(dataset.image_filenames)!=expected_train_count:raise ValueError('Subject quotas require the explicitly declared train count')
        if not 0<=boundary_share<=1:raise ValueError('Invalid hair boundary share')
        self.boundary_share=boundary_share;self.raster_scales={};self.frequency_weighted_hair=frequency_weighted_hair;self.cdfs={}
        self.records=[]
        for index,path in enumerate(dataset.image_filenames):
            # Campaign filenames include the immutable physical camera name.
            name=Path(path).stem.removeprefix('frame_train_') if hasattr(str,'removeprefix') else Path(path).stem[len('frame_train_'):]
            source=semantic_dir/(name+'.npz');data=np.load(source)['probability'].astype(np.float32)/65535
            hair=data[...,0]>.5;person=(data[...,2]<.2)&~hair
            kernel=np.ones((boundary_width,boundary_width),np.uint8)
            band=cv2.dilate(hair.astype(np.uint8),kernel).astype(bool)^cv2.erode(hair.astype(np.uint8),kernel).astype(bool)
            self.pools[index]={key:torch.from_numpy(np.argwhere(mask).astype(np.int64)) for key,mask in [('hair',hair),('boundary',band),('person',person)]}
            label=np.where(hair,0,np.where(person,1,np.where(data[...,2]>.8,2,3))).astype(np.uint8)
            self.labels[index]=torch.from_numpy(label)
            self.raster_scales[index]=(int(dataset.cameras.height[index])/label.shape[0],int(dataset.cameras.width[index])/label.shape[1])
            if frequency_weighted_hair:
                # Match each semantic cell to its native frequency-map cell;
                # rescaling labels never reassigns a map from another camera.
                frequency=torch.load(Path(path).parent.parent/'lookcloser_frequencies'/(Path(path).stem+'.pt'),map_location='cpu',weights_only=True).float()
                levels=(torch.log(frequency.clamp_min(16)/16)/np.log((8192/16)**(1/15))).clamp(0,15)
                sy,sx=self.raster_scales[index]
                for key in ['hair','boundary']:
                    pool=self.pools[index][key]
                    y=((pool[:,0].float()+.5)*sy/8).long().clamp(0,frequency.shape[0]-1)
                    x=((pool[:,1].float()+.5)*sx/8).long().clamp(0,frequency.shape[1]-1)
                    self.cdfs[index,key]=torch.cumsum((1+2*levels[y,x]/15).double(),0)
            self.records.append(dict(camera=name,semantic_sha256=sha(source),hair_pixels=int(hair.sum()),person_nonhair_pixels=int(person.sum()),boundary_pixels=int(band.sum())))

    def draw(self,key,count,available):
        eligible=[i for i in available if len(self.pools[i][key])]
        if not eligible or count==0:return torch.empty((0,3),dtype=torch.long)
        cameras=torch.tensor(eligible)[torch.randint(len(eligible),(count,))]
        result=torch.empty((count,3),dtype=torch.long);result[:,0]=cameras
        for camera in torch.unique(cameras).tolist():
            selected=cameras==camera;pool=self.pools[camera][key]
            if (camera,key) in self.cdfs:
                cdf=self.cdfs[camera,key];pick=torch.searchsorted(cdf,torch.rand(int(selected.sum()),dtype=torch.float64)*cdf[-1])
            else:pick=torch.randint(len(pool),(int(selected.sum()),))
            result[selected,1:]=pool[pick]
            sy,sx=self.raster_scales[camera]
            if sy!=1 or sx!=1:
                scale=torch.tensor([sy,sx]);result[selected,1:]=((result[selected,1:].float()+torch.rand((int(selected.sum()),2)))*scale).long()
        return result

    def sample(self,image_batch,**kwargs):
        if not isinstance(image_batch['image'],torch.Tensor) or image_batch['image'].device.type!='cpu':
            raise ValueError('Subject quota control requires homogeneous CPU image cache')
        if set(image_batch)-{'image','image_idx'}:raise ValueError('Unsupported pixel metadata; cannot silently misalign auxiliary targets')
        step=max(int(getattr(self.sampler,'sample_count',0)),self.audit_start_sample_count+self.audit['batches'])
        baseline=self.original(image_batch,**kwargs);indices=baseline['indices'].clone();n=len(indices)
        strength=min(max((step-self.warmup)/max(self.ramp,1),0),1)
        nh=int(round(n*self.hair_fraction*strength));nperson=min(n-nh,int(round(n*self.person_fraction*strength)));nb=int(round(nh*self.boundary_share))
        available=image_batch['image_idx'].tolist()
        proposed=torch.cat([self.draw('hair',nh-nb,available),self.draw('boundary',nb,available),self.draw('person',nperson,available)],0)
        replace=torch.randperm(n)[:len(proposed)];indices[replace]=proposed
        local={int(global_index):i for i,global_index in enumerate(available)}
        c=torch.tensor([local[int(i)] for i in indices[:,0]],dtype=torch.long)
        baseline['image']=image_batch['image'][c,indices[:,1],indices[:,2]]
        baseline['indices']=indices
        counts=torch.zeros(4,dtype=torch.long);classes=torch.empty(n,dtype=torch.uint8)
        for camera in torch.unique(indices[:,0]).tolist():
            select=indices[:,0]==camera;sy,sx=self.raster_scales[camera];label=self.labels[camera]
            y=((indices[select,1].float()+.5)/sy).long().clamp(0,label.shape[0]-1);x=((indices[select,2].float()+.5)/sx).long().clamp(0,label.shape[1]-1)
            classes[select]=label[y,x]
            counts+=torch.bincount(classes[select].long(),minlength=4)
        baseline['subject_sampling_class']=classes
        for name,value in zip(('hair','person_nonhair','background','uncertain'),counts.tolist()):self.audit[name]+=value
        self.audit['batches']+=1;self.audit['rays']+=n
        self.last_batch=dict(zip(('hair','person_nonhair','background','uncertain'),counts.tolist()))
        return baseline

    def install(self,model=None):
        self.sampler.sample=self.sample
        self.query_counts=None
        if model is not None:
            original_metrics=model.get_metrics_dict
            def metrics(outputs,batch):
                result=original_metrics(outputs,batch)
                if model.training and 'subject_sampling_class' in batch:
                    labels=batch['subject_sampling_class'].to(outputs['rgb'].device).long()
                    with torch.no_grad():
                        if 'packed_ray_indices' in outputs:
                            count=torch.bincount(labels[outputs['packed_ray_indices'].long()],minlength=4)
                        else:count=torch.bincount(labels,minlength=4)*outputs['loss_weights'].shape[1]
                        if self.query_counts is None:self.query_counts=torch.zeros_like(count)
                        self.query_counts+=count
                        result['hair_point_fraction']=count[0].float()/count.sum().clamp_min(1)
                return result
            model.get_metrics_dict=metrics

    def save(self,path):
        atomic_json(path,dict(policy=dict(hair_fraction=self.hair_fraction,person_nonhair_fraction=self.person_fraction,
            context_fraction=1-self.hair_fraction-self.person_fraction,hair_boundary_share=self.boundary_share,frequency_weighted_hair=self.frequency_weighted_hair,boundary_width_pixels=self.boundary_width,warmup_steps=self.warmup,ramp_steps=self.ramp),
            audit=self.audit,audit_start_sample_count=self.audit_start_sample_count,last_batch=getattr(self,'last_batch',{}),
            rgb_training_field_queries_by_class=(dict(zip(('hair','person_nonhair','background','uncertain'),self.query_counts.cpu().tolist()))
                if getattr(self,'query_counts',None) is not None else None),
            records=self.records,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))
