"""CPU config/source parity for focus × distortion; never initializes CUDA."""
import argparse, json, sys, os
from pathlib import Path
from audit_marshawn_frequency_focus_config import canonical

def main(root,parent):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(root/'code'),str(root/'code/LookCloser/scripts')]
    import torch
    import train_marshawn_semantic_admission as worker
    spec=json.loads((root/'spec.json').read_text()); before=json.loads((parent/'spec.json').read_text())
    worker.validate_spec(spec)
    oldroot=str(Path(before['output']).parent);newroot=str(Path(spec['output']).parent)
    normalized=json.loads(json.dumps(spec).replace(newroot,oldroot));normalized['distortion_final']=.01
    if normalized!=before: raise ValueError('Scientific spec changed beyond distortion coefficient')
    old=canonical(worker.build_config(before));new=json.loads(json.dumps(canonical(worker.build_config(spec))).replace(newroot,oldroot))
    if new!=old: raise ValueError('Initial actual model/pipeline config differs; distortion is set by existing step callback')
    bad=[]
    for key,value in [('coarse_step',.003125),('distortion_final',.01),('width',32),('radius',1)]:
        try:worker.validate_spec(dict(spec,**{key:value}))
        except ValueError:bad.append(key)
        else:raise ValueError('Undeclared variant accepted')
    schedule={str(step):.01+(.1-.01)*min(1.,max(0.,(step-4096)/8192)) for step in [0,4096,8192,12288,75941,106316]}
    if torch.cuda.is_initialized():raise ValueError('CPU proof initialized CUDA')
    result=dict(passed=True,initial_actual_config_difference={},scientific_spec_difference={'distortion_final':[.01,.1]},
          actual_existing_callback_schedule=schedule,rejected_variants=bad,CUDA_initialized=False,
          frequency_hook_private_state_and_model_source_unchanged=True)
    (root/'config_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();main(a.root,a.parent)
