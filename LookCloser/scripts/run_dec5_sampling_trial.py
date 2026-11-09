#!/usr/bin/env python3
"""Explicit paired sampling-policy ablation from a frozen-background parent.

All arms restore the same complete native training state. Only the allocation
within the fixed hair quota changes; background attachment and empty-space
supervision remain identical. This is an experimental wrapper, never a default.
"""
import argparse
import json
from pathlib import Path
import sys
from run_dec5_hair_campaign import atomic_json,sha


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--background',type=Path,required=True)
    parser.add_argument('--background-semantics',type=Path,required=True)
    parser.add_argument('--boundary-share',type=float,default=.3)
    parser.add_argument('--frequency-weighted-hair',action='store_true')
    args,rest=parser.parse_known_args()
    out=Path(rest[rest.index('--output')+1]);parent=Path(rest[rest.index('--resume-from')+1])
    parent_request=json.loads((parent/'request.json').read_text())
    if not parent_request.get('background_separation_experiment'):raise ValueError('A matched separated-background parent is required')
    import train_dec5_hair_baseline as baseline
    import dec5_subject_sampling as sampling
    from dec5_background_separation import install_training
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    original_sampler=sampling.SubjectSampler
    class PolicySampler(original_sampler):
        def __init__(self,*a,**kw):
            kw.update(warmup=0,ramp=1,boundary_share=args.boundary_share,frequency_weighted_hair=args.frequency_weighted_hair)
            super().__init__(*a,**kw)
    sampling.SubjectSampler=PolicySampler
    original_load=LookCloserPipeline.load_state_dict;installed=[]
    def load(pipe,*a,**kw):
        result=original_load(pipe,*a,**kw)
        if installed:raise ValueError('Background attached more than once')
        entry=parent_request['ray_background']
        provider,counts=install_training(pipe,args.background,args.background_semantics,out,
            weight=entry['empty_space_weight'],known_background_rgb=entry.get('known_background_rgb',False))
        request=json.loads((out/'request.json').read_text())
        request.update(ray_background=dict(entry,path=str(args.background),sha256=sha(args.background)),
            background_separation_experiment=True,sampling_ablation=dict(boundary_share=args.boundary_share,
                frequency_weighted_hair=args.frequency_weighted_hair,parent_request_sha256=sha(parent/'request.json')))
        atomic_json(out/'request.json',request);installed.append(provider);return result
    LookCloserPipeline.load_state_dict=load
    sys.argv=[sys.argv[0]]+rest;baseline.main()
    if len(installed)!=1:raise ValueError('Background attachment missing')
    installed[0].verify()


if __name__=='__main__':main()
