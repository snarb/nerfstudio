#!/usr/bin/env python3
"""Matched ablations from the corrected cold recipe's first sealed checkpoint."""
from pathlib import Path
import argparse,time,os
from run_dec5_sequence import read,write,identity
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','root']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=['control','hair_mix','frequency_gate01'],required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    write(a.root/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    source=a.pilot/'frames/000901';parent=source/'training_hd_stable_v6/step-000015188.ckpt'
    started=time.monotonic()
    while not parent.exists():
        if time.monotonic()-started>7200:raise TimeoutError('Corrected baseline parent not ready in two hours')
        write(a.root/'progress.json',dict(phase='waiting_corrected_parent',time=time.time()));time.sleep(30)
    frame=a.root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        target=frame/name
        if not target.exists():target.symlink_to((source/name).resolve(),target_is_directory=True)
    write(frame/'frequency_ready.json',read(source/'frequency_ready.json'));write(a.root/'manifest.json',read(a.pilot/'manifest.json'))
    recipe=dict(read(a.pilot/'recipe.json'),revision='corrected_'+a.arm,experimental_continuation=True,
        purpose='Matched5000-update ablation after numerical baseline repair; not native production')
    if a.arm=='hair_mix':recipe['hair_core_mix']=.5
    if a.arm=='frequency_gate01':recipe['frequency_update_minimum_opacity']=.1
    write(a.root/'recipe.json',recipe);train(a.root,'000901',recipe,max_steps=20189,experimental_parent=parent)
    write(a.root/'complete.json',dict(arm=a.arm,production_frame_completion=False,time=time.time()))


if __name__=='__main__':main()
