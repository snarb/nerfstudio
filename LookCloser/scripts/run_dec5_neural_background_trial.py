#!/usr/bin/env python3
"""Matched 5000-update continuations from the corrected20188 full state."""
from pathlib import Path
import argparse,os
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def dependency(root):
    background=root/'neural_background';complete=read(background/'complete.json')
    if not complete or complete['smoke'] or complete['steps']!=20000:raise ValueError('Full background trial required')
    checkpoint=Path(complete['checkpoint']['path']);config=Path(complete['configuration']);source=root/'frame/semantic_data/transforms.json'
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        config=str(config),config_sha256=sha(config),source_metadata=str(source),source_metadata_sha256=sha(source),
        frozen=True,empty_subject_half_width=.18)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['control','volume'],required=True);a=p.parse_args();r=a.root;out=a.output
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    out.mkdir(parents=True,exist_ok=True);write(out/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=__import__('time').time()))
    frame=out/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    source=r/'frame'
    for name,target in [('data','semantic_data'),('semantic_data','semantic_data'),('semantics','semantics'),('ray_background','ray_background')]:
        link=frame/name
        if not link.exists():link.symlink_to((source/target).resolve(),target_is_directory=True)
    data=frame/'data';train_meta=read(data/'transforms.json')
    write(out/'manifest.json',dict(frames=['000901'],heldout_camera='D004_A005_12103C',
        purpose='One-frame matched experimental continuation; not a production50 manifest'))
    write(frame/'frequency_ready.json',dict(data_sha256=sha(data/'transforms.json'),maps=64,
        revision='Existing genuine64-camera HD fits from corrected baseline'))
    ancestor=r/'models/corrected_control/frames/000901/training_corrected_control'
    parent=ancestor/'step-000020188.ckpt'
    recipe=dict(read(ancestor/'request.json')['recipe'],revision='background_'+a.arm,
        experimental_continuation=True,hair_core_mix=0.,
        purpose='Matched20189..25188: fixed original vs separate3D background; sampling unchanged')
    if a.arm=='volume':recipe['background_volume']=dependency(r)
    write(out/'recipe.json',recipe)
    train(out,'000901',recipe,max_steps=25189,experimental_parent=parent)
    write(out/'complete.json',dict(arm=a.arm,production_frame_completion=False,updates=5000))


if __name__=='__main__':main()
