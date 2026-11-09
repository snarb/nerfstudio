#!/usr/bin/env python3
"""Native6K warm-background test, matched to contour control at20188."""
from pathlib import Path
import argparse,os,time
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train
from run_dec5_neural_background_trial import dependency


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    out=r/'native_warm_background';out.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    write(out/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    source=r/'native_background_control/frames/000901';frame=out/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        if not (frame/name).exists():(frame/name).symlink_to((source/name).resolve(),target_is_directory=True)
    write(frame/'frequency_ready.json',read(source/'frequency_ready.json'))
    write(out/'manifest.json',dict(frames=['000901'],heldout_camera='D004_A005_12103C'))
    parent=source/'training_native_bg_control/step-000015188.ckpt'
    recipe=dict(read(r/'native_background_control/recipe.json'),revision='native_warm_background',
        experimental_cold_probe=False,experimental_continuation=True,initial_same_frame_ancestor_sha256=sha(parent),
        background_volume=dict(dependency(r),start_step=15189),
        purpose='Native same15188 full state, no sampling change; background volume activates from15189')
    write(out/'recipe.json',recipe);train(out,'000901',recipe,max_steps=20189,experimental_parent=parent)
    write(out/'complete.json',dict(updates=5000,native6k=True,production50_complete=False))


if __name__=='__main__':main()
