#!/usr/bin/env python3
"""Train-only ray/bounds coverage audit; semantic people are not instance GT."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
from run_dec5_sequence import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--frame-root',type=Path,required=True);p.add_argument('--scene-scale',type=float,default=.15);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=True)
    data=a.frame_root/'semantic_data';meta=json.loads((data/'transforms.json').read_text());frames=meta['frames']
    poses=torch.tensor([f['transform_matrix'] for f in frames],dtype=torch.float32)
    normalized,_=auto_orient_and_center_poses(poses,method='up',center_method='focus')
    normalized[:,:3,3]/=normalized[:,:3,3].abs().max();poses=normalized.numpy();rows=[];tiles=[]
    for frame,pose in zip(frames,poses):
        if frame['file_path'] not in meta['train_filenames']:continue
        values=np.load(a.frame_root/'semantics'/(frame['physical_camera']+'.npz'))['probability'].astype(np.float32)/65535
        h,w=values.shape[:2];y,x=np.mgrid[0:h:8,0:w:8]
        direction=np.stack(((x+.5-frame['cx'])/frame['fl_x'],-(y+.5-frame['cy'])/frame['fl_y'],-np.ones_like(x)),-1)@pose[:3,:3].T
        safe=np.where(np.abs(direction)>1e-12,direction,np.where(direction<0,-1e-12,1e-12))
        lo=(-a.scene_scale-pose[:3,3])/safe;hi=(a.scene_scale-pose[:3,3])/safe
        hit=np.minimum(lo,hi).max(-1)<=np.maximum(lo,hi).min(-1)
        hit&=np.maximum(lo,hi).min(-1)>0
        probability=values[::8,::8];hair=probability[...,0]>.5;person=probability[...,2]<.2
        row=dict(camera=frame['physical_camera'],hair_samples=int(hair.sum()),hair_missed=int((hair&~hit).sum()),
            person_samples=int(person.sum()),person_missed=int((person&~hit).sum()))
        rows.append(row)
        if frame['physical_camera'] in ['C004_B005_1210ER','F004_A005_12103K','K004_D005_121016'] or row['hair_missed']>0:
            rgb=cv2.imread(str(data/frame['file_path']));overlay=rgb.copy();miss=cv2.resize((person&~hit).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
            overlay[miss]=(.3*overlay[miss]+.7*np.array([0,0,255])).astype(np.uint8)
            panel=cv2.resize(np.rot90(overlay),(480,960),interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(a.output/(frame['physical_camera']+'.jpg')),panel)
    write(a.output/'audit.json',dict(scene_scale=a.scene_scale,rows=rows,
        hair_missed=sum(r['hair_missed'] for r in rows),hair_samples=sum(r['hair_samples'] for r in rows),
        person_missed=sum(r['person_missed'] for r in rows),person_samples=sum(r['person_samples'] for r in rows),
        caveat='Sapiens includes background crew; misses require instance-aware visual review. Ray intersection is necessary, not sufficient for surface containment.',uses_heldout_rgb=False))


if __name__=='__main__':main()
