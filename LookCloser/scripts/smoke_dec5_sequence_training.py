#!/usr/bin/env python3
"""Two real optimizer updates of the production trainer on synthetic tiny RGB.

This checks fixed warmup, separate background, the 64+1 split, person/FAS sampler,
Jacobian projection, checkpoint save and evaluation. It measures no scene quality.
Outputs must be isolated from every real training campaign.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch
from prepare_dec5_sequence import pose
from fast_freqmap import save_metadata
from run_dec5_sequence import write
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rig',type=Path,required=True);a=p.parse_args()
    root=a.root;frame='000901';directory=root/'frames'/frame;data=directory/'data'
    if root.exists():raise FileExistsError('Synthetic structural test requires an isolated new directory')
    (data/'images').mkdir(parents=True);(data/'lookcloser_frequencies').mkdir();(directory/'semantics').mkdir()
    rig=json.loads(a.rig.read_text());heldout='D004_A005_12103C';rows=[]
    for name,camera in sorted(rig['cameras'].items()):
        prefix='eval' if name==heldout else 'train';relative=f'images/frame_{prefix}_{name}.png'
        image=np.full((64,128,3),60,np.uint8);image[16:55,45:85]=[130,90,60];cv2.imwrite(str(data/relative),image)
        fx,fy,cx,cy,*dist=camera['params_fx_fy_cx_cy_k1_k2_p1_p2']
        rows.append(dict(file_path=relative,physical_camera=name,transform_matrix=pose(camera),
            w=128,h=64,fl_x=fx*128/6144,fl_y=fy*64/3072,cx=cx*128/6144,cy=cy*64/3072,k1=0,k2=0,p1=0,p2=0))
        if name==heldout:continue
        probability=np.zeros((64,128,3),np.uint16);probability[...,2]=65535;probability[16:55,45:85,2]=0;probability[16:35,45:85,0]=65535
        np.savez_compressed(directory/'semantics'/(name+'.npz'),probability=probability)
        target=data/'lookcloser_frequencies'/(Path(relative).stem+'.pt');torch.save(torch.full((8,16),16.),target)
        save_metadata(target.with_suffix('.json'),Path(relative).name,(64,128),8,8192)
    meta=dict(camera_model='OPENCV',w=128,h=64,frame_id=frame,frames=rows,
        train_filenames=[r['file_path'] for r in rows if r['physical_camera']!=heldout],
        val_filenames=[r['file_path'] for r in rows if r['physical_camera']==heldout],
        test_filenames=[r['file_path'] for r in rows if r['physical_camera']==heldout])
    write(data/'transforms.json',meta);(directory/'semantic_data').symlink_to(data,target_is_directory=True)
    write(root/'manifest.json',dict(frames=[frame],heldout_camera=heldout,purpose='SYNTHETIC structural test; no scene-quality result'))
    write(directory/'frequency_ready.json',dict(data_sha256=sha(data/'transforms.json'),maps=64))
    recipe=dict(revision='synthetic_smoke',synthetic_smoke_test=True,scene_scale=.15,
        sampling_policy='luster_body80_fas50',frequency_projection='normalized_jacobian',
        final_distortion=1.,occupancy_dilation=1,synthetic_iteration_steps=[0,8192])
    train(root,frame,recipe,max_steps=2)
    write(root/'smoke_complete.json',dict(real_optimizer_updates=2,synthetic_rgb=True,no_quality_claim=True))


if __name__=='__main__':main()
