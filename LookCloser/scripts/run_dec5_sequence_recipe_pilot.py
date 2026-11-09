#!/usr/bin/env python3
"""HD preflight of the full new 64+1 recipe on a separate experimental machine.

Uses genuine HD fits; never substitutes resized native frequency maps. It is
not a native-quality result and does not mark a production frame complete.
"""
from pathlib import Path
import argparse
import json
import time
import torch
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha
from fast_freqmap import process_image,save_metadata
from nerfstudio.scripts.lookcloser_preprocess import load_image_as_tensor
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    frame='000901';directory=a.root/'frames'/frame;data=directory/'data';torch.set_num_threads(2)
    if not data.exists():data.symlink_to(directory/'semantic_data',target_is_directory=True)
    write(a.root/'manifest.json',dict(frames=[frame],heldout_camera='D004_A005_12103C'))
    recipe=dict(revision='hd_preflight_v4',scene_scale=.15,
        sampling_policy='luster_body80_fas50',frequency_projection='normalized_jacobian',final_distortion=1.,occupancy_dilation=1,
        purpose='HD functional/quality pilot of the new full leader recipe; independent of silver native queue')
    write(a.root/'recipe.json',recipe);meta=read(data/'transforms.json');maps=data/'lookcloser_frequencies';maps.mkdir(exist_ok=True)
    for index,name in enumerate(meta['train_filenames']):
        source=data/name;target=maps/(source.stem+'.pt');receipt=target.with_suffix('.json')
        if target.exists() and read(receipt,{}).get('source_sha256')==sha(source):continue
        torch.manual_seed(42+index);image=load_image_as_tensor(source,torch.device('cuda'))
        values=process_image(image,1000,8192,2048,8192,ps=8,ssim_thr=.95,win=7)
        temp=target.with_suffix('.pt.tmp');torch.save(values,temp);temp.replace(target)
        save_metadata(receipt,source.name,image.shape[:2],8,8192);detail=read(receipt);detail.update(source_sha256=sha(source));write(receipt,detail)
        write(directory/'progress.json',dict(phase='frequency',completed=index+1,total=64,time=time.time()))
        del image,values;torch.cuda.empty_cache()
    write(directory/'frequency_ready.json',dict(data_sha256=sha(data/'transforms.json'),maps=64))
    train(a.root,frame,recipe)
    write(a.root/'complete.json',dict(resolution='1920x960 HD preflight, not native 6K',time=time.time()))


if __name__=='__main__':main()
