"""Actual YAML object graph and full pre-Torch source-authorizer CPU gate."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import yaml


def main():
    p=Path(sys.argv[1]);m=json.loads((p/'source_execution_manifest.json').read_text())
    sys.path[:0]=[str(p/'code'),m['helpers'],m['branch']+'/code',m['branch']+'/code/LookCloser/scripts']
    import run_marshawn_lc_semantic_capture_v3 as worker
    release=json.loads((p/'source_loader_release_TEMPLATE.json').read_text())
    release.update(capture_authorized=True,CPU_authorizer_fixture_only=True)
    # This file is never the spec's GPU release path. It tests the pre-Torch
    # complete hash closure with the production authorizer, without monkeypatch.
    fixture=p/'CPU_authorizer_fixture_NOT_GPU_RELEASE.json'
    fixture.write_text(json.dumps(release,indent=2)+'\n')
    worker.authorize(p/'source_execution_manifest.json',fixture,Path(m['campaign']))
    before_torch='torch' in sys.modules
    original=Path(m['campaign'])/'frozen_LC_semantic_capture_plan_v2/local_source/config_local_paths_only.yml'
    a=yaml.load(original.read_text(),Loader=yaml.Loader)
    b=yaml.load((p/'config_execution_alias.yml').read_text(),Loader=yaml.Loader)
    C='/fsx/tmp/marshawn_001660_hd/tuning_20261106';SC=m['campaign']
    def graph(v,relocate,memo):
        typ=type(v).__module__+'.'+type(v).__qualname__
        if v is None or isinstance(v,(bool,int,float,str)):
            return [typ,v.replace(C,SC)if relocate and isinstance(v,str)else v]
        if isinstance(v,type)or callable(v):return ['callable',v.__module__+'.'+v.__qualname__]
        if id(v)in memo:return ['alias',memo[id(v)]]
        memo[id(v)]=len(memo)
        if isinstance(v,Path):return [typ,str(v).replace(C,SC)if relocate else str(v)]
        if isinstance(v,dict):return [typ,[[graph(k,relocate,memo),graph(x,relocate,memo)]for k,x in v.items()]]
        if isinstance(v,(tuple,list)):return [typ,[graph(x,relocate,memo)for x in v]]
        if hasattr(v,'__dict__'):return [typ,graph(vars(v),relocate,memo)]
        raise TypeError('Unhandled config type '+typ)
    if graph(a,True,{})!=graph(b,False,{}):raise ValueError('Actual object graph/types/aliases changed beyond paths')
    b.eval_checkpoint=Path(m['local_checkpoint'])
    expected={}
    for dotted,value in m['expected_config_paths'].items():
        actual=b
        for k in dotted.split('.'):actual=getattr(actual,k)
        if str(actual)!=value:raise ValueError('Wrong typed path '+dotted)
        expected[dotted]=str(actual)
    leaves=[];seen=set()
    def walk(v,name):
        if isinstance(v,type)or callable(v)or id(v)in seen:return
        seen.add(id(v))
        if isinstance(v,Path):leaves.append([name,str(v)]);return
        if isinstance(v,dict):
            for k,x in v.items():walk(x,name+'.'+str(k))
        elif isinstance(v,(tuple,list)):
            for i,x in enumerate(v):walk(x,name+'['+str(i)+']')
        elif hasattr(v,'__dict__'):
            for k,x in vars(v).items():walk(x,name+'.'+k)
    walk(b,'cfg')
    if any(s.startswith(C)for _,s in leaves):raise ValueError('Unrelocated Path leaf')
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU audit created CUDA context')
    result=dict(passed=True,exact_source_runner_authorize=True,all63_query_hashes=True,
        authorizer_imported_Torch=before_torch,actual_yaml_Loader=True,
        actual_object_graph_types_tuples_aliases_equal_except_paths=True,
        expected_config_paths=expected,all_Path_leaves=leaves,CUDA_initialized=False,
        model_constructed=False,fixture_is_not_GPU_release=True)
    (p/'actual_typed_config_CPU_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
