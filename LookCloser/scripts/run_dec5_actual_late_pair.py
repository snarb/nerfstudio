#!/usr/bin/env python3
"""Actual silver V7 prefix through91128; matched late correction to106316."""
from pathlib import Path
import argparse,os,time
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['control','warm'],required=True);a=p.parse_args();r=a.root
    out=r/('late_'+a.arm);out.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    write(out/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    source=r/'frames/000901';frame=out/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        if not (frame/name).exists():(frame/name).symlink_to((source/name).resolve(),target_is_directory=True)
    write(out/'manifest.json',dict(frames=['000901'],heldout_camera='D004_A005_12103C'))
    write(frame/'frequency_ready.json',dict(data_sha256=sha(frame/'data/transforms.json'),maps=64))
    parent=r/'prefix/step-000091128.ckpt';digest=sha(parent)
    recipe=dict(revision='actual_late_'+a.arm,sampling_policy='luster_body80_fas50',scene_scale=.15,
        frequency_projection='normalized_jacobian',final_distortion=1.,occupancy_dilation=1,
        eval_rays_per_chunk=8192,hair_core_mix=0.,background_seed=42,
        actual_production_prefix_sha256=digest,background_switch_step=91129,
        purpose='Real V7 prefix0..91128, matched late continuation through106316; not production50 completion')
    if a.arm=='warm':
        bg=source/'neural_background';complete=read(bg/'complete.json')
        if not complete or complete['steps']!=20000 or complete['smoke']:raise ValueError('Full new-platform background fit required')
        checkpoint=Path(complete['checkpoint']['path']);config=Path(complete['configuration']);metadata=source/'semantic_data/transforms.json'
        recipe['background_volume']=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config=str(config),config_sha256=sha(config),
            source_metadata=str(metadata),source_metadata_sha256=sha(metadata),frozen=True,empty_subject_half_width=.18,start_step=91129)
    write(out/'recipe.json',recipe);training=frame/('training_'+recipe['revision']);training.mkdir(exist_ok=True)
    if not (training/'latest.json').exists():
        checkpoint=training/parent.name
        if not checkpoint.exists():os.link(parent,checkpoint)
        initial=next(row for row in read(r/'prefix/history.json') if row['step']==91128)
        initial=dict(initial,renders=str(r/'prefix/eval_091128'))
        write(training/'history.json',[initial]);write(training/'latest.json',dict(path=str(checkpoint),sha256=digest,step=91128))
    train(out,'000901',recipe)
    write(out/'complete.json',dict(final_step=106316,actual_prefix_last_step=91128,
        native_resolution=[6144,3072],background_activated=91129 if a.arm=='warm' else None,
        selection_scope='91128 and106316; prior canonical metrics remain in the original silver production run',
        production50_complete=False))


if __name__=='__main__':main()
