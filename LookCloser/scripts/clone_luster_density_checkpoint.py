#!/usr/bin/env python3
"""Expand a volume checkpoint with a density clone and matching Adam moments."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import torch

from run_luster_opacity import sha,write


def convert(selection,output):
    if output.exists():
        raise FileExistsError('Preserve checkpoint conversions')
    chosen=json.loads(selection.read_text());chosen=chosen.get('selected',chosen)
    source=Path(chosen['checkpoint'])
    checkpoint=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    prefix='_model.field.'
    names=['encoding.params','mlp_geo.params','direction_encoding.params','mlp_color.params']
    keys=[k for k in checkpoint['pipeline'] if k.startswith(prefix) and k.endswith('.params')]
    if keys!=[prefix+n for n in names]:
        raise ValueError('Unsupported source parameter layout; do not guess optimizer indices')
    optimizer=checkpoint['optimizers']['fields']
    groups=optimizer['param_groups']
    if len(groups)!=1 or len(groups[0]['params'])!=len(names):
        raise ValueError('Unsupported source optimizer groups')
    old_ids=groups[0]['params'][:]
    for name,index in zip(names,old_ids):
        parameter=checkpoint['pipeline'][prefix+name]
        if optimizer['state'][index]['exp_avg'].shape!=parameter.shape:
            raise ValueError('Optimizer state does not match expected parameter')
    new_ids=[max(old_ids)+1,max(old_ids)+2]
    for old,new,old_id,new_id in zip(['encoding','mlp_geo'],
            ['density_geometry_encoding','density_geometry_mlp'],old_ids[:2],new_ids):
        checkpoint['pipeline'][prefix+new+'.params']=checkpoint['pipeline'][prefix+old+'.params'].clone()
        optimizer['state'][new_id]=deepcopy(optimizer['state'][old_id])
    groups[0]['params'].extend(new_ids)
    request=json.loads((source.parents[4]/'request.json').read_text())
    request.update(output=str(output),resume=str(source),end_step=chosen['step'],eval_steps=[])
    request['model']['separate_density_encoding']=True
    target=output/'field/lookcloser-opacity/run/nerfstudio_models'/source.name
    target.parent.mkdir(parents=True)
    torch.save(checkpoint,target)
    write(output/'request.json',request)
    write(output/'selection.json',dict(step=chosen['step'],checkpoint=str(target)))
    write(output/'conversion.json',dict(source=str(source),source_sha256=sha(source),
        checkpoint=str(target),checkpoint_sha256=sha(target),parameter_names=names,
        cloned_geometry_parameters=['density_geometry_encoding.params','density_geometry_mlp.params'],
        optimizer_mapping=dict(zip(map(str,new_ids),old_ids[:2])),
        original_parameters_unchanged=True,scheduler_scaler_rng_preserved=True,
        source_sha256_script=sha(Path(__file__)),forward_parity='pending'))
    print(json.dumps(dict(output=str(output),step=chosen['step'],new_parameter_ids=new_ids)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();convert(a.selection,a.output)
