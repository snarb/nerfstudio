#!/usr/bin/env python3
"""Clone the selected LookCloser density branch, preserving native continuation state."""
from copy import deepcopy
from pathlib import Path
import argparse,json,shutil
import torch,yaml
from run_dec5_hair_campaign import atomic_json,sha


def expand(checkpoint,optimizer=False):
    prefix='_model.field.'
    names=['encoding.params','mlp_geo.params','direction_encoding.params','mlp_color.params']
    actual=[k for k in checkpoint['pipeline'] if k.startswith(prefix) and k.endswith('.params')]
    if actual!=[prefix+n for n in names]:raise ValueError('Unsupported field parameter layout')
    if optimizer:
        opt=checkpoint['optimizers']['fields'];groups=opt['param_groups']
        if len(groups)!=1 or len(groups[0]['params'])!=4:raise ValueError('Unsupported Adam layout')
        ids=groups[0]['params'][:];new=[max(ids)+1,max(ids)+2]
        for name,index in zip(names,ids):
            if checkpoint['pipeline'][prefix+name].numel():
                if opt['state'][index]['exp_avg'].shape!=checkpoint['pipeline'][prefix+name].shape:
                    raise ValueError('Adam parameter layout mismatch')
        for old,new_id in zip(ids[:2],new):opt['state'][new_id]=deepcopy(opt['state'][old])
        groups[0]['params'].extend(new)
    original={k:v.clone() for k,v in checkpoint['pipeline'].items()}
    for old,new in [('encoding','density_geometry_encoding'),('mlp_geo','density_geometry_mlp')]:
        checkpoint['pipeline'][prefix+new+'.params']=checkpoint['pipeline'][prefix+old+'.params'].clone()
    assert all(torch.equal(v,checkpoint['pipeline'][k]) for k,v in original.items())
    return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    configs=list(a.parent.rglob('config.yml'));natives=list(a.parent.rglob('step-*.ckpt'))
    if len(configs)!=1 or len(natives)!=1:raise ValueError('Use one pinned parent config and native checkpoint')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    if cfg.pipeline.model.separate_density_encoding:raise ValueError('Parent is already split')
    cfg.pipeline.model.separate_density_encoding=True
    (a.output/'config.yml').write_text(yaml.dump(cfg))
    native=torch.load(natives[0],map_location='cpu',weights_only=False)
    step=native['step'];gate=a.parent/('gate_%06d.pt'%step)
    target=a.output/'native'/natives[0].name;target.parent.mkdir()
    torch.save(expand(native,True),target)
    weights=torch.load(gate,map_location='cpu',weights_only=False);torch.save(expand(weights),a.output/gate.name)
    for name in ['request.json','scene_bounds.json','history.json']:shutil.copy2(a.parent/name,a.output/name)
    atomic_json(a.output/'conversion.json',dict(parent=str(a.parent),native_parent_sha256=sha(natives[0]),
        parent_gate_sha256=sha(gate),native_sha256=sha(target),gate_sha256=sha(a.output/gate.name),
        step=step,original_pipeline_parameters_unchanged=True,cloned_density_parameters=True,
        adam_moments_cloned=True,scheduler_scaler_rng_preserved=True,forward_parity='required before training',
        implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
