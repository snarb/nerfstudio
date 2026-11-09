#!/usr/bin/env python3
"""Read-only full-checkpoint derivative for inference transfer; never resume SGD."""
from pathlib import Path
import argparse
import torch
from run_dec5_sequence import write
from dec5_sequence_worker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Do not overwrite an inference artifact')
    torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    compact={key:state[key] for key in ['step','pipeline','runtime'] if key in state}
    compact['inference_only']=True;compact['full_parent_checkpoint_sha256']=sha(a.checkpoint)
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp')
    torch.save(compact,temp)
    loaded=torch.load(temp,map_location='cpu',weights_only=False)
    original=state['pipeline'];copy=loaded['pipeline']
    if set(original)!=set(copy):raise ValueError('Inference derivative lost state keys')
    for name,value in original.items():
        if not isinstance(value,torch.Tensor) or not torch.equal(value,copy[name]):
            raise ValueError('Inference tensor differs: '+name)
    if getattr(original,'_metadata',{})!=getattr(copy,'_metadata',{}):raise ValueError('State metadata differs')
    temp.replace(a.output)
    write(a.output.with_suffix('.json'),dict(step=state['step'],inference_only=True,
        full_parent_checkpoint=str(a.checkpoint),full_parent_checkpoint_sha256=compact['full_parent_checkpoint_sha256'],
        inference_checkpoint_sha256=sha(a.output),all_pipeline_tensors_roundtrip_equal=True,
        bytes=a.output.stat().st_size,full_parent_retained=True,optimizer_state_present=False))


if __name__=='__main__':main()
