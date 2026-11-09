#!/usr/bin/env python3
"""Frozen-foreground counterfactual: train-only higher-capacity ray background.

No NeRF fitting, altered geometry, source selection or held-out background fit.
The subtraction uses the same raw calibrated rays as the frozen background.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from dec5_hair import cv_pose
from dec5_ray_background import RayBackgroundNetwork
from train_dec5_sequence import fit_background
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


@torch.no_grad()
def predict(checkpoint,frame):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=RayBackgroundNetwork(state['origin_levels'],state['direction_levels'],state['width']).cuda().eval()
    model.load_state_dict(state['state_dict']);pose=torch.as_tensor(cv_pose(frame),device='cuda',dtype=torch.float32)
    h,w=frame['h'],frame['w'];y,x=np.mgrid[:h,:w];x,y=x.reshape(-1),y.reshape(-1)
    origin=(pose[:3,3]-torch.tensor(state['center'],device='cuda'))/state['radius'];result=np.empty((h*w,3),np.float32)
    for begin in range(0,len(x),32768):
        xx=torch.tensor(x[begin:begin+32768],device='cuda');yy=torch.tensor(y[begin:begin+32768],device='cuda')
        direction=torch.stack(((xx+.5-frame['cx'])/frame['fl_x'],(yy+.5-frame['cy'])/frame['fl_y'],torch.ones_like(xx)),1).float()@pose[:3,:3].T
        direction=torch.nn.functional.normalize(direction,dim=-1)
        result[begin:begin+len(xx)]=model(origin.expand(len(xx),3),direction).cpu().numpy()
    return result.reshape(h,w,3)


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    frame=a.pilot/'frames/000901';directory=a.output/'frame';directory.mkdir(exist_ok=True)
    for name in ['data','semantic_data','semantics']:
        target=directory/name
        if not target.exists():target.symlink_to((frame/name).resolve(),target_is_directory=True)
    new=fit_background(directory,origin_levels=2,direction_levels=9,width=128,steps=20000,batch_size=8192,seed=42)
    meta=read(frame/'data/transforms.json');heldout=[f for f in meta['frames'] if f['file_path'] in meta['val_filenames']]
    if len(heldout)!=1:raise ValueError('One held-out camera required')
    training=frame/'training_hd_preflight_v4';entry=next(r for r in read(training/'history.json') if r['step']==15188)
    original_path=Path(entry['renders'])/'0.npz';arrays=dict(np.load(original_path))
    old=frame/'ray_background/background.pt';before=predict(old,heldout[0]);after=predict(new,heldout[0])
    alpha=arrays['accumulation'];alpha=alpha[...,None] if alpha.ndim==2 else alpha
    foreground=arrays['rgb']-(1-alpha).clip(0,1)*before
    tolerance=float(np.maximum(-foreground,0).max())
    if tolerance>2e-4:raise ValueError('Raw-ray background subtraction does not match normalized inference')
    arrays['rgb']=foreground+(1-alpha).clip(0,1)*after
    for key in ['depth','accumulation']:np.testing.assert_array_equal(arrays[key],np.load(original_path)[key])
    render=a.output/'eval_015188';render.mkdir(exist_ok=True);np.savez_compressed(render/'0.npz',**arrays)
    write(a.output/'history.json',[dict(step=15188,renders=str(render))])
    write(a.output/'complete.json',dict(foreground_frozen=True,geometry_unchanged=True,
        source_prediction_sha256=sha(original_path),original_background_sha256=sha(old),new_background_sha256=sha(new),
        prediction_sha256=sha(render/'0.npz'),foreground_subtraction_negative_tolerance=tolerance,
        background_fitted_from_train_only=True,heldout_used_for_fitting=False))


if __name__=='__main__':main()
