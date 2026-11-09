"""Reload exported tensors without training-image caches or optimizer allocation."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from common import read,atomic,sha
from runtime import bootstrap,config


def load_pipeline(root,spec_path,checkpoint,output,data_override=None):
    root=Path(root);settings=read(root/'settings.json');spec=read(spec_path)
    # Relocated artifacts use the receiving host's qualified runtime, while model
    # parameters and the selected source checkpoint remain explicitly recorded.
    for key in ('source_code','frozen_helpers','nerfacc_extension'):
        spec[key]=settings.get(key)
    local_data=root/'variants'/Path(spec['data']).name
    if local_data.exists():spec['data']=str(local_data)
    if data_override:spec['data']=str(Path(data_override).resolve())
    data=Path(spec['data'])
    spec['semantic_admission']=dict(manifest=str(data/'admission/manifest.json'),manifest_sha256=sha(data/'admission/manifest.json'))
    spec=copy.deepcopy(spec);spec['output']=str(output)
    bootstrap(spec)
    torch.set_num_threads(4)
    cfg=config(spec);pipe=cfg.pipeline.setup(device='cuda:0',test_mode='inference')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    pipe.load_pipeline(state['pipeline'],state['step']);step=state['step'];del state
    pipe.eval()
    return pipe,spec,step


def run(root,spec_path,checkpoint,output,caps,data_override=None):
    pipe,spec,step=load_pipeline(root,spec_path,checkpoint,output,data_override)
    from worker import score
    results=[]
    for cap in caps:
        result=score(SimpleNamespace(pipeline=pipe),spec,step,cap)
        results.append(result)
    atomic(Path(output)/'inference_evaluation.json',dict(checkpoint_sha256=sha(checkpoint),caps=caps,results=results,training_image_cache=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--spec',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--data');p.add_argument('--caps',default='9,10,11,12');a=p.parse_args()
    run(a.root,a.spec,a.checkpoint,a.output,[None if c=='none' else float(c) for c in a.caps.split(',')],a.data)
