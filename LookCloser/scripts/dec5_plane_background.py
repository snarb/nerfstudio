"""Frozen multi-view plane texture lookup; no target image or subject geometry."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dec5_hair import cv_pose
from dec5_sequence_worker import sha
from run_dec5_sequence import read


class PlaneBackground:
    def __init__(self,data,semantics,geometry,device='cuda'):
        receipt=read(geometry/'complete.json');meta=read(data/'transforms.json')
        if receipt['data_sha256']!=sha(data/'transforms.json') or receipt['uses_validation_rgb'] or receipt['uses_test_rgb']:
            raise ValueError('Unverified train background geometry')
        names={f['physical_camera'] for f in meta['frames'] if f['file_path'] in meta['train_filenames']}
        self.frames=receipt['frames'];self.device=device;self.planes=receipt['planes'];self.receipt=receipt
        self.images=[];self.masks=[];self.poses=[];self.intrinsics=[]
        records={r['camera']:r for r in receipt['records']}
        for f in self.frames:
            name=f['physical_camera'];image_path=data/f['file_path'];mask_path=semantics/(name+'.npz')
            if name not in names or sha(image_path)!=records[name]['rgb_sha256'] or sha(mask_path)!=records[name]['semantic_sha256']:
                raise ValueError('Train background input changed')
            image=cv2.imread(str(image_path))[...,::-1].astype(np.float32)/255
            p=np.load(mask_path)['probability']
            mask=cv2.erode((p[...,2]>round(.98*65535)).astype(np.uint8),np.ones((33,33),np.uint8))
            if mask.shape!=image.shape[:2]:mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
            self.images.append(torch.as_tensor(image.copy(),device=device).permute(2,0,1)[None])
            self.masks.append(torch.as_tensor(mask,device=device,dtype=torch.float32)[None,None])
            self.poses.append(torch.as_tensor(cv_pose(f),device=device,dtype=torch.float32))
            self.intrinsics.append(torch.tensor([f['fl_x'],f['fl_y'],f['cx'],f['cy']],device=device))

    @torch.no_grad()
    def predict(self,origins,directions):
        n=len(origins);device=origins.device
        output=torch.zeros((n,3),device=device);best_t=torch.full((n,),torch.inf,device=device)
        support=torch.zeros(n,device=device,dtype=torch.int16);valid_output=torch.zeros(n,device=device,dtype=torch.bool)
        for plane in self.planes:
            normal=torch.tensor(plane['normal'],device=device,dtype=origins.dtype)
            denominator=directions@normal
            t=-(origins@normal+plane['offset'])/torch.where(denominator.abs()>1e-6,denominator,torch.ones_like(denominator))
            points=origins+t[:,None]*directions
            basis=torch.tensor(plane['basis'],device=device,dtype=origins.dtype)
            origin=torch.tensor(plane['origin'],device=device,dtype=origins.dtype)
            uv=(points-origin)@basis.T
            hull=torch.tensor(plane['hull_equations'],device=device,dtype=origins.dtype)
            domain=(denominator.abs()>1e-6)&(t>0)&(t<best_t)&((uv@hull[:,:2].T+hull[:,2])<=1e-5).all(-1)
            ids=torch.nonzero(domain).flatten()
            if not len(ids):continue
            samples=[]
            for f,image,mask,pose,k in zip(self.frames,self.images,self.masks,self.poses,self.intrinsics):
                camera=(points[ids]-pose[:3,3])@pose[:3,:3];z=camera[:,2]
                xy=camera[:,:2]/z.clamp_min(1e-6)[:,None]*k[:2]+k[2:]
                inside=(z>0)&(xy[:,0]>=.5)&(xy[:,0]<=f['w']-.5)&(xy[:,1]>=.5)&(xy[:,1]<=f['h']-.5)
                grid=(2*xy/xy.new_tensor([f['w'],f['h']])-1).reshape(1,-1,1,2)
                known=F.grid_sample(mask,grid,mode='nearest',padding_mode='zeros',align_corners=False)[0,0,:,0]>.5
                colors=F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,:,0].T
                samples.append(torch.where((inside&known)[:,None],colors,torch.full_like(colors,float('nan'))))
            samples=torch.stack(samples);count=torch.isfinite(samples[...,0]).sum(0)
            color=torch.nanmedian(samples,dim=0).values
            dispersion=torch.nanmedian((samples-color[None]).abs(),dim=0).values.mean(-1)
            accepted=(count>=3)&(dispersion<=.05)&torch.isfinite(color).all(-1)
            take=ids[accepted];output[take]=color[accepted];support[take]=count[accepted].to(torch.int16)
            best_t[take]=t[take];valid_output[take]=True
        return output,valid_output,best_t,support
