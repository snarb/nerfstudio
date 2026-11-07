"""CPU-only local path materialization and real frozen-parser/ray proof."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from prepare_marshawn_lc_semantic_capture import CAMPAIGN, sha, write

SC = '/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
ROOT = CAMPAIGN/'frozen_LC_semantic_capture_plan_v2'


def materialize_yaml(source, target):
    """Edit ONLY pathname nodes; keep every other tagged scalar/type identical."""
    original = yaml.compose(source.read_text())
    result = copy.deepcopy(original)
    differences = []
    def walk(node, location):
        if isinstance(node, SequenceNode) and node.tag == 'tag:yaml.org,2002:python/object/apply:pathlib.PosixPath':
            before = str(Path(*(x.value for x in node.value)))
            after = before.replace(SC,str(CAMPAIGN)) if before.startswith(SC) else before
            if location == 'output_dir': after = str(ROOT/'unused_config_output')
            if before != after:
                differences.append(dict(key=location,before=before,after=after,type='PosixPath'))
                node.value = [ScalarNode('tag:yaml.org,2002:str',s) for s in Path(after).parts]
        elif isinstance(node,ScalarNode):
            if node.tag == 'tag:yaml.org,2002:str' and node.value.startswith(SC):
                before=node.value;node.value=before.replace(SC,str(CAMPAIGN),1)
                differences.append(dict(key=location,before=before,after=node.value,type='str'))
        elif isinstance(node,MappingNode):
            for k,v in node.value: walk(v,location+'.'+k.value if location else k.value)
        elif isinstance(node,SequenceNode):
            for i,v in enumerate(node.value):walk(v,location+f'[{i}]')
    walk(result,'')
    # Independent structural proof normalizes path-valued nodes only.
    def skeleton(node):
        if isinstance(node,SequenceNode) and node.tag.endswith('python/object/apply:pathlib.PosixPath'):
            return (node.tag,'PATH')
        if isinstance(node,ScalarNode):
            return (node.tag,'PATH' if node.tag.endswith(':str') and node.value.startswith('/') else node.value)
        if isinstance(node,MappingNode):return (node.tag,[(skeleton(k),skeleton(v))for k,v in node.value])
        return (node.tag,[skeleton(v)for v in node.value])
    if skeleton(original) != skeleton(result):raise ValueError('Nonpath YAML scalar/type changed')
    if target.exists():raise FileExistsError('Never replace a materialized config')
    target.write_text(yaml.serialize(result))
    if SC in target.read_text():raise ValueError('Unresolved silver root remains in local config')
    return differences, result


def parser_proof(config_node):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set empty CUDA_VISIBLE_DEVICES for CPU parser proof')
    branch=CAMPAIGN/'semantic_quarter_distortion_candidate_20261107'
    sys.path[:0]=[str(branch/'code'),str(branch/'code/LookCloser/scripts')]
    import torch
    import numpy as np
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from train_marshawn_semantic_admission import TuningParser
    def member(node,key):return next(v for k,v in node.value if k.value == key)
    parser_node=member(member(member(config_node,'pipeline'),'datamanager'),'dataparser')
    kwargs={k.value:yaml.safe_load(v.value) for k,v in parser_node.value if k.value not in ('data','_target')}
    kwargs['data']=CAMPAIGN/'data'
    parser=TuningParser(NerfstudioDataParserConfig(**kwargs));parser.downscale_factor=1
    train=parser.get_dataparser_outputs(split='train');heldout=parser.get_dataparser_outputs(split='test')
    names=[p.stem.split('_')[-1]for p in train.image_filenames]
    if len(names)!=63 or {p.stem.split('_')[-1]for p in heldout.image_filenames} != {'DC','HC','KA'}:
        raise ValueError('Real frozen parser split differs')
    if torch.cuda.is_initialized():raise ValueError('CPU parser unexpectedly initialized CUDA')
    raw={f['physical_camera']:f for f in json.loads((CAMPAIGN/'data/transforms.json').read_text())['frames']}
    v1=json.loads((CAMPAIGN/'frozen_LC_semantic_capture_plan_v1/manifest.json').read_text())
    maxpose=maxdirection=maxorigin=0.;total=0
    for row in v1['records']:
        cam=row['camera'];camera=train.cameras[names.index(cam):names.index(cam)+1];f=raw[cam]
        error=float((camera.camera_to_worlds[0]-torch.tensor(f['transform_matrix'],dtype=torch.float32)[:3]).abs().max());maxpose=max(maxpose,error)
        for k,v in [('fx','fl_x'),('fy','fl_y'),('cx','cx'),('cy','cy'),('width','w'),('height','h')]:
            if float(getattr(camera,k).reshape(-1)[0]) != float(torch.tensor(f[v],dtype=getattr(camera,k).dtype)):
                raise ValueError('Real parser intrinsics/raster differ')
        with np.load(row['archive'],allow_pickle=False)as q:pixels=q['pixel_xy'].copy()
        for first in range(0,len(pixels),512):
            uv=torch.tensor(pixels[first:first+512],dtype=torch.float32)
            rays=camera.generate_rays(0,coords=uv[:,[1,0]]+.5,keep_shape=False)
            local=torch.stack(((uv[:,0]+.5-camera.cx[0,0])/camera.fx[0,0],-(uv[:,1]+.5-camera.cy[0,0])/camera.fy[0,0],-torch.ones_like(uv[:,0])),-1)
            expected=torch.nn.functional.normalize(local@camera.camera_to_worlds[0,:3,:3].T,dim=-1)
            maxdirection=max(maxdirection,float((expected-rays.directions).abs().max()))
            maxorigin=max(maxorigin,float((rays.origins-camera.camera_to_worlds[0,:3,3]).abs().max()))
        total+=len(pixels)
    if maxpose != 0 or maxdirection > 5e-7 or maxorigin > 2e-7 or total != 69888:
        raise ValueError('Real parser/floor pixel-center ray chain differs')
    return dict(train=63,eval=3,query_rays=total,pose_max_error=maxpose,
        pinhole_direction_max_error=maxdirection,origin_max_error=maxorigin,
        CUDA_initialized=torch.cuda.is_initialized(),model_constructed=False,
        source_parser_module=str(Path(sys.modules[TuningParser.__module__].__file__).resolve()),
        head_world_to_box=np.eye(4).tolist(),EVAL_target_RGB_or_masks_read=False)


def prepare():
    archive=CAMPAIGN/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin'
    root=ROOT;local=root/'local_source';local.mkdir(exist_ok=True)
    config=local/'config_local_paths_only.yml'
    diffs,node=materialize_yaml(archive/'config.yml',config)
    proof=parser_proof(node);write(root/'CPU_parser_ray_proof.json',proof)
    entries=[]
    for d in diffs:
        after=Path(d['after'])
        if d['key']=='output_dir':after.mkdir(exist_ok=True);continue
        if not after.exists():raise FileNotFoundError(str(after))
        if after.is_file():entries.append(dict(**d,sha256=sha(after),same_source_input_bytes=True))
        else:
            for p in [after/'transforms.json',after/'ready.json',after/'box.json']:
                if p.exists():entries.append(dict(key=d['key'],before=d['before']+'/'+p.name,after=str(p),sha256=sha(p),same_source_input_bytes=True))
    write(root/'path_materialization.json',dict(source_config=str(archive/'config.yml'),source_config_sha256=sha(archive/'config.yml'),
        local_config=str(config),local_config_sha256=sha(config),path_differences=diffs,
        nonpath_tagged_scalars_identical=True,input_hashes=entries,
        source_spec_and_parent_manifests_rewritten=False,
        note='Source spec/parent lineage files are hash-bound provenance only; all executable config references resolve locally. Their serialized silver path strings remain untouched.'))
    print(json.dumps(dict(config=str(config),path_changes=len(diffs),proof=proof,GPU_started=False)))


if __name__=='__main__':prepare()
