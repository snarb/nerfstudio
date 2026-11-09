#!/usr/bin/env python3
"""Cold full-recipe HD verification after the packed-distortion precision fix."""
from pathlib import Path
import argparse
from run_dec5_sequence import read,write
from train_dec5_sequence import train


def main():
    p=argparse.ArgumentParser()
    for key in ['pilot','root']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();source=a.pilot/'frames/000901';frame=a.root/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name in ['data','semantic_data','semantics','ray_background']:
        target=frame/name
        if not target.exists():target.symlink_to((source/name).resolve(),target_is_directory=True)
    write(frame/'frequency_ready.json',read(source/'frequency_ready.json'))
    write(a.root/'manifest.json',read(a.pilot/'manifest.json'))
    recipe=dict(read(a.pilot/'recipe.json'),revision='hd_stable_v6',eval_rays_per_chunk=8192,
        packed_distortion_precision='fp64_global_prefix',
        purpose='Cold full106317-update HD verification after a numerical bug fix; not native production')
    write(a.root/'recipe.json',recipe);train(a.root,'000901',recipe)
    write(a.root/'complete.json',dict(resolution='HD functional/quality pilot, not native6K',numerical_fix=True))


if __name__=='__main__':main()
