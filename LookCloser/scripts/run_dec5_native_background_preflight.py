#!/usr/bin/env python3
"""Real native6K cold A/B at the canonical15188 review gate; not production50."""
from pathlib import Path
import argparse,os,time
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha
from train_dec5_sequence import train
from run_dec5_neural_background_trial import dependency


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['control','volume'],required=True);a=p.parse_args();r=a.root;out=a.output
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    out.mkdir(parents=True,exist_ok=True);write(out/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    source=r/'native_frame901/data';meta=read(source/'transforms.json')
    if meta['frame_id']!='000901' or len(meta['train_filenames'])!=64 or len(meta['val_filenames'])!=1:
        raise ValueError('Native90164+1 input required')
    verification=[]
    for name in meta['train_filenames']:
        image=source/name;path=source/'lookcloser_frequencies'/(Path(name).stem+'.pt');record=read(path.with_suffix('.json'))
        if record['image_shape']!=[3072,6144] or record['crop_coords_xywh'] is not None or record['fit_steps_per_level']!=1000:
            raise ValueError('Expected genuine native frequency fit')
        if sha(image)!=record['source_sha256'] or sha(path)!=record['map_sha256']:
            raise ValueError('Native image/frequency provenance mismatch')
        verification.append(dict(image=name,map_sha256=record['map_sha256'],source_sha256=record['source_sha256']))
    write(out/'input_verification.json',dict(images=verification,native_shape=[3072,6144],train_only=True))
    frame=out/'frames/000901';frame.mkdir(parents=True,exist_ok=True)
    for name,target in [('data',source),('semantic_data',r/'frame/semantic_data'),('semantics',r/'frame/semantics')]:
        if not (frame/name).exists():(frame/name).symlink_to(target.resolve(),target_is_directory=True)
    write(out/'manifest.json',dict(frames=['000901'],heldout_camera='D004_A005_12103C',
        purpose='Native single-frame canonical review gate; not production50'))
    write(frame/'frequency_ready.json',dict(data_sha256=sha(source/'transforms.json'),maps=64,
        revision='Verified native6K1000-steps-per-level source fits'))
    recipe=dict(revision='native_bg_'+a.arm,sampling_policy='luster_body80_fas50',scene_scale=.15,
        frequency_projection='normalized_jacobian',final_distortion=1.,occupancy_dilation=1,
        eval_rays_per_chunk=8192,hair_core_mix=0.,background_seed=42,experimental_cold_probe=True,
        purpose='Real native cold-start15188 A/B after explicit background-seed fix')
    if a.arm=='volume':recipe['background_volume']=dict(dependency(r),start_step=0)
    write(out/'recipe.json',recipe);train(out,'000901',recipe,max_steps=15189)
    write(out/'complete.json',dict(arm=a.arm,updates=15189,native_resolution=[6144,3072],
        production50_complete=False,canonical_review_gate=15188))


if __name__=='__main__':main()
