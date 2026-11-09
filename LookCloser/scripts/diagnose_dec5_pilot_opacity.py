#!/usr/bin/env python3
"""Post-prediction held-out opacity/frequency diagnosis; no fitting hooks."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np,torch
import torch.nn.functional as F
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser()
    for key in ['frame-root','training','sapiens-repo','sapiens-checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,default=15188);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    data=a.frame_root/'data';meta=read(data/'transforms.json');evals=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(evals)!=1:raise ValueError('One sealed held-out view required')
    row=evals[0];prediction_path=a.training/f'eval_{a.step:06d}/0.npz'
    history=read(a.training/'history.json')
    if not prediction_path.exists() or not any(r['step']==a.step for r in history):raise ValueError('Prediction must be sealed before reference segmentation')
    prediction=dict(np.load(prediction_path));image=cv2.imread(str(data/row['file_path']))
    sys.path.insert(0,str(a.sapiens_repo));from sapiens.dense.models import init_model
    config=a.sapiens_repo/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py'
    model=init_model(config,a.sapiens_checkpoint,device='cuda:0');portrait=np.ascontiguousarray(np.rot90(image))
    inputs=model.data_preprocessor(model.pipeline(dict(img=portrait)))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(inputs['inputs']);logits=F.interpolate(logits.float(),size=portrait.shape[:2],mode='bilinear',align_corners=False)
        probability=logits.softmax(1)[0,[4,3,0]].permute(1,2,0).cpu().numpy()
    probability=np.ascontiguousarray(np.rot90(probability,-1));np.savez_compressed(a.output/'reference.npz',probability=np.rint(probability*65535).astype(np.uint16))
    del model,logits;torch.cuda.empty_cache()
    poses=torch.tensor([f['transform_matrix'] for f in meta['frames']],dtype=torch.float32)
    poses,_=auto_orient_and_center_poses(poses,method='up',center_method='focus');poses[:,:3,3]/=poses[:,:3,3].abs().max()
    index=next(i for i,f in enumerate(meta['frames']) if f['physical_camera']==row['physical_camera']);pose=poses[index].numpy()
    state=torch.load(a.training/f'step-{a.step:09d}.ckpt',map_location='cpu',weights_only=False)['pipeline']
    prefix=next(k[:-len('grid')] for k in state if k.endswith('freq_grid.grid') and 'field.' not in k)
    grid=state[prefix+'grid'].numpy();low=state[prefix+'aabb_min_buf'].numpy();span=state[prefix+'aabb_size_buf'].numpy()
    alpha=prediction['accumulation'].squeeze();depth=prediction['depth'].squeeze();h,w=alpha.shape;y,x=np.mgrid[:h,:w]
    direction=np.stack(((x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-np.ones_like(x)),-1)
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True);points=pose[:3,3]+(direction@pose[:3,:3].T)*depth[...,None]
    inside=((points>=low)&(points<=low+span)).all(-1)&(depth>0)
    indices=np.clip((points-low)/span*(grid.shape[0]-1),0,grid.shape[0]-1.001).astype(np.int64)
    levels=grid[indices[...,0],indices[...,1],indices[...,2]]
    hair=probability[...,0]>.5;face=probability[...,1]>.5;background=cv2.erode((probability[...,2]>.98).astype(np.uint8),np.ones((33,33),np.uint8))>0
    band=cv2.dilate(hair.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)^cv2.erode(hair.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    results={}
    for label,mask in [('hair',hair),('face',face),('trusted_bg',background),('hair_boundary',band)]:
        valid=mask&inside&(alpha>=.01)
        results[label]=dict(pixels=int(mask.sum()),opacity_mean=float(alpha[mask].mean()),
            opacity_quantiles=np.quantile(alpha[mask],[.05,.5,.95]).tolist(),
            opacity_below_01_fraction=float((alpha[mask]<.1).mean()),
            opacity_above_09_fraction=float((alpha[mask]>=.9).mean()),
            frequency_level_histogram=np.bincount(levels[valid].astype(int),minlength=16).tolist(),
            frequency_low_level_fraction=float((levels[valid]<=3).mean()) if valid.any() else None)
    write(a.output/'diagnosis.json',dict(step=a.step,regions=results,prediction_sha256=sha(prediction_path),
        reference_used_for_fitting=False,reference_is_fallible_semantics=True,frequency_position='rendered expected depth; no independent 3D ground truth'))


if __name__=='__main__':main()
