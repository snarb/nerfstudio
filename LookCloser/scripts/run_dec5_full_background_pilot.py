#!/usr/bin/env python3
"""Continue the measured same-frame background ancestry through final FR.3."""
from pathlib import Path
import argparse,os,time
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    write(root/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    source=a.root/'foreground_background_volume';frame=root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        if not (frame/name).exists():(frame/name).symlink_to((source/'frames/000901'/name).resolve(),target_is_directory=True)
    write(root/'manifest.json',read(source/'manifest.json'))
    write(frame/'frequency_ready.json',read(source/'frames/000901/frequency_ready.json'))
    ancestor=source/'frames/000901/training_background_volume'
    parent=ancestor/'step-000025188.ckpt';digest=sha(parent)
    recipe=dict(read(source/'recipe.json'),revision='background_volume_full',
        purpose='HD continuation of measured same-frame ancestry through106316, not native production',
        verified_same_frame_ancestor_sha256=digest,background_switch_step=20189)
    recipe['background_volume']=dict(recipe['background_volume'],start_step=20189)
    write(root/'recipe.json',recipe);out=frame/'training_background_volume_full';out.mkdir(exist_ok=True)
    if not (out/'latest.json').exists():
        checkpoint=out/parent.name
        if not checkpoint.exists():os.link(parent,checkpoint)
        write(out/'latest.json',dict(path=str(checkpoint),sha256=digest,step=25188))
    train(root,'000901',recipe)
    write(root/'complete.json',dict(final_step=106316,native_production=False,
        full_state_ancestor_step=25188,background_switch_step=20189,
        checkpoint_selection_scope='Canonical gates after25188; native production retains all seven canonical gates'))


if __name__=='__main__':main()
