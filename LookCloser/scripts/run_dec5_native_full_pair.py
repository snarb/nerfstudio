#!/usr/bin/env python3
"""Full native64+1 matched pair sharing the exact cold0..15188 prefix."""
from pathlib import Path
import argparse,os,time
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['control','warm'],required=True)
    a=p.parse_args();r=a.root;root=r/('full_'+a.arm);root.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    write(root/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    shared=r/'shared';ancestor=read(shared/'ancestor.json');parent=shared/'parent/step-000015188.ckpt'
    if sha(parent)!=ancestor['checkpoint_sha256']:raise ValueError('Native common ancestor checksum mismatch')
    frame=root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    source=Path('/home/ubuntu/lookcloser_transfer/recipe_pilot_20261028/frames/000901')
    for name,target in [('data',r/'data'),('semantic_data',source/'semantic_data'),('semantics',source/'semantics'),('ray_background',shared/'ray_background')]:
        if not (frame/name).exists():(frame/name).symlink_to(target.resolve(),target_is_directory=True)
    write(root/'manifest.json',dict(frames=['000901'],heldout_camera='D004_A005_12103C',
        purpose='Single-frame full native matched pair, not production50 completion'))
    ready=read(shared/'frequency_ready.json')
    if sha(r/'data/transforms.json')!=ready['data_sha256'] or ready['maps']!=64:raise ValueError('Native input receipt mismatch')
    write(frame/'frequency_ready.json',ready)
    recipe=dict(read(shared/'recipe.json'),revision='native_full_'+a.arm,experimental_cold_probe=False,
        experimental_continuation=False,purpose='Full native matched pair with exact shared0..15188 prefix',
        initial_same_frame_ancestor_sha256=ancestor['checkpoint_sha256'],background_switch_step=15189)
    if a.arm=='warm':
        dependency=read(shared/'volume_dependency.json')
        dependency.update(checkpoint=str(shared/'volume/step-000019999.ckpt'),config=str(shared/'volume/config.yml'),
            source_metadata=str(source/'semantic_data/transforms.json'),start_step=15189)
        recipe['background_volume']=dependency
    write(root/'recipe.json',recipe);out=frame/('training_'+recipe['revision']);out.mkdir(exist_ok=True)
    if not (out/'latest.json').exists():
        checkpoint=out/parent.name
        if not checkpoint.exists():os.link(parent,checkpoint)
        initial=read(shared/'parent/history.json')
        if len(initial)!=1 or initial[0]['step']!=15188:raise ValueError('Expected sealed canonical first gate')
        initial[0]['renders']=str(shared/'parent/eval_015188')
        write(out/'history.json',initial);write(out/'latest.json',dict(path=str(checkpoint),sha256=sha(checkpoint),step=15188))
    train(root,'000901',recipe)
    write(root/'complete.json',dict(final_step=106316,full_native_recipe=True,production50_complete=False,
        shared_prefix_last_step=15188,background_switch_step=15189 if a.arm=='warm' else None,
        all_seven_canonical_gates_retained=True))


if __name__=='__main__':main()
