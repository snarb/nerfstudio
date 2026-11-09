#!/usr/bin/env python3
"""Full verified ancestry: corrected body80 through15188, then hair mixture.

The native50 recipe will reproduce the first stage independently per frame.
This one-frame HD pilot reuses that exact same-frame full-state checkpoint.
"""
from pathlib import Path
import argparse,os,shutil
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','root']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();source=a.pilot/'frames/000901';frame=a.root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        target=frame/name
        if not target.exists():target.symlink_to((source/name).resolve(),target_is_directory=True)
    write(frame/'frequency_ready.json',read(source/'frequency_ready.json'));write(a.root/'manifest.json',read(a.pilot/'manifest.json'))
    parent=source/'training_hd_stable_v6';checkpoint=parent/'step-000015188.ckpt';digest=sha(checkpoint)
    recipe=dict(read(a.pilot/'recipe.json'),revision='hd_hair_v7',hair_core_mix=.5,hair_mix_start=15189,
        initial_same_frame_ancestor_sha256=digest,
        purpose='Full corrected same-frame ancestry with measured hair-mixture switch; HD, not native production')
    write(a.root/'recipe.json',recipe);out=frame/'training_hd_hair_v7';out.mkdir(exist_ok=True)
    if not (out/'latest.json').exists():
        target=out/checkpoint.name
        if not target.exists():os.link(checkpoint,target)
        row=next(r for r in read(parent/'history.json') if r['step']==15188)
        destination=out/'eval_015188'
        if not destination.exists():shutil.copytree(row['renders'],destination,copy_function=os.link)
        write(out/'history.json',[dict(row,renders=str(destination))])
        write(out/'latest.json',dict(path=str(target),sha256=digest,step=15188))
    train(a.root,'000901',recipe)
    write(a.root/'complete.json',dict(full_steps=106317,same_frame_ancestry=True,native_production=False))


if __name__=='__main__':main()
