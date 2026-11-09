#!/usr/bin/env python3
"""Matched, full-state 5000-update HD continuation; never a production frame."""
from pathlib import Path
import argparse
from run_dec5_sequence import read,write
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','root','parent']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=['control','front10','distortion01','observedbg_off','stable_distortion'],required=True);a=p.parse_args()
    source=a.pilot/'frames/000901';frame=a.root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        destination=frame/name
        if not destination.exists():destination.symlink_to((source/name).resolve(),target_is_directory=True)
    write(frame/'frequency_ready.json',read(source/'frequency_ready.json'))
    write(a.root/'manifest.json',read(a.pilot/'manifest.json'))
    recipe=dict(read(a.pilot/'recipe.json'),revision='frequency_depth_'+a.arm,
        experimental_continuation=True,purpose='Same-state HD frequency projection ablation, not native production')
    if a.arm=='front10':recipe['frequency_depth_quantile']=.1
    if a.arm=='distortion01':recipe['final_distortion']=.01
    if a.arm=='observedbg_off':recipe['known_background_rgb']=False
    if a.arm=='stable_distortion':recipe['packed_distortion_precision']='fp64_global_prefix'
    write(a.root/'recipe.json',recipe)
    train(a.root,'000901',recipe,max_steps=20189,experimental_parent=a.parent)


if __name__=='__main__':main()
