"""Optional calibrated stereo intervals; conditional depth, never alpha targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def opaque_surface_intervals(depths,sigmas,opaque):
    """Conservative HD neighborhoods also cover native subpixel RGB jitter."""
    finite=torch.isfinite(depths).all(0)&torch.isfinite(sigmas).all(0)
    good=finite&(depths>0).all(0)&(sigmas>0).all(0)&opaque.bool()
    difference=(depths[0]-depths[1]).abs()
    good &= difference<=2*sigmas.sum(0)
    depth=torch.where(good,depths.mean(0),0.)
    sigma=torch.where(good,sigmas.max(0).values+difference/2,0.)
    valid=(1-F.max_pool2d(F.pad((~good)[None,None].float(),(1,1,1,1),value=1),3,stride=1))[0,0].bool()
    high=F.max_pool2d(depth[None,None],3,stride=1,padding=1)[0,0]
    low=-F.max_pool2d(-depth[None,None],3,stride=1,padding=1)[0,0]
    scale=F.max_pool2d(sigma[None,None],3,stride=1,padding=1)[0,0]
    valid &= (high-low)<=2*scale
    # No precision gain is claimed from averaging correlated network predictions.
    return (high+low)/2,scale+(high-low)/2,valid


def conditional_depth_interval(weights,distances,target,sigma,valid):
    """Penalize density shape outside a two-scale interval, invariant to alpha scale."""
    selected=valid.reshape(-1)>0
    if not selected.any():return weights.sum()*0
    w=weights[selected].float();t=distances[selected].float()
    z=target[selected].float()[:,None];scale=sigma[selected].float()[:,None]
    excess=((t-z).abs()/scale-2).clamp_min(0)
    penalty=torch.where(excess<1,.5*excess.square(),excess-.5)
    return ((w*penalty).sum(1)/w.sum(1).clamp_min(1e-8)).mean()


class StereoDepthTargets:
    def __init__(self,receipt,dataset):
        receipt=Path(receipt);meta=json.loads(receipt.read_text());root=Path(dataset.metadata['distillation_root'])
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        stage_path=Path(meta['arguments']['stereo'])/'request.json';stage=json.loads(stage_path.read_text())
        if dataset.metadata['distillation_split']!='train' or meta['uses_eval'] or stage['heldout_used']:
            raise ValueError('Stereo depth requires train-only evidence')
        if digest(stage_path)!=meta['staged_request_sha256'] or digest(root/'transforms.json')!=stage['data_manifest_sha256']:
            raise ValueError('Stereo depth calibration/data identity changed')
        for index,expected in meta['source_image_sha256'].items():
            if digest(dataset.image_filenames[int(index)])!=expected:raise ValueError('Stereo source photograph changed')
        self.views={};records=[]
        for record in meta['disjoint_pairs']:
            index=record['index'];first,second=record['pairs']
            if set(first)&set(second):raise ValueError('Depth evidence must use disjoint camera pairs')
            if index in self.views:raise ValueError('Ambiguous multiple stereo proposals for one camera')
            path=receipt.parent/f'camera_{index:02}_disjoint_{first[0]:02}_{second[0]:02}.npz'
            if digest(path)!=meta['depth_artifacts'][path.name]:raise ValueError('Stereo depth arrays changed')
            data=np.load(path,allow_pickle=False);depth=torch.from_numpy(data['depths']);sigma=torch.from_numpy(data['sigmas'])
            row=dataset.metadata['distillation_rows'][index]
            if depth.shape!=(2,row['h'],row['w']) or sigma.shape!=depth.shape:raise ValueError('Stereo depth dimensions mismatch')
            batch=dataset[index]
            opaque=((batch['alpha_target']>=.999)&(batch['mask']>0)&(batch['confidence']>=1)&(batch['alpha_valid']>=1)&(batch['empty_mask']==0))[...,0]
            z,s,valid=opaque_surface_intervals(depth,sigma,opaque)
            self.views[index]=(z,s,valid)
            records.append(dict(index=index,pairs=record['pairs'],trusted_pixels=int(valid.sum()),
                median_interval_scale=float(s[valid].median()) if valid.any() else None))
        self.receipt=dict(evidence_sha256=digest(receipt),records=records,scope='Opaque coherent 3x3 neighborhoods; no opacity target',
            uncertainty='Propagated disparity scale plus pair/neighborhood spread; heuristic, not calibrated depth accuracy',uses_eval=False)

    def apply(self,batch):
        camera,y,x=batch['indices'].cpu().T
        depth=torch.zeros((len(camera),1));sigma=torch.ones_like(depth);valid=torch.zeros_like(depth)
        for i,(z,s,known) in self.views.items():
            select=camera==i;yy=y[select];xx=x[select]
            depth[select,0]=z[yy,xx];sigma[select,0]=s[yy,xx];valid[select,0]=known[yy,xx].float()
        return dict(batch,stereo_depth=depth,stereo_depth_sigma=sigma,stereo_depth_valid=valid)
