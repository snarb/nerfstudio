#!/usr/bin/env python3
"""Source-depth interpolation ablation; RGB/semantic interpolation is unchanged."""
from pathlib import Path
import argparse,json,sys
import torch
import study_dec5_hair_geometry as geometry
from run_dec5_hair_campaign import atomic_json


def normalized_depth_sample(image,uv,minimum_support=.5,sampling=geometry.sample):
    numerator=sampling(image,uv,mode='bilinear')
    support=sampling((image>0).to(image),uv,mode='bilinear')
    return torch.where(support>=minimum_support,numerator/support.clamp_min(1e-12),torch.zeros_like(numerator))


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--source-depth-interpolation',choices=['normalized','nearest'],required=True)
    a,rest=p.parse_known_args();out=Path(rest[rest.index('--output')+1])
    original=geometry.sample
    def sample(image,uv,mode='bilinear'):
        if image.shape[0]!=1:return original(image,uv,mode)
        if a.source_depth_interpolation=='nearest':return original(image,uv,'nearest')
        return normalized_depth_sample(image,uv,sampling=original)
    geometry.sample=sample
    import render_dec5_native_head_benchmark as benchmark
    sys.argv=[sys.argv[0]]+rest;benchmark.main()
    receipt=json.loads((out/'complete.json').read_text());receipt.update(source_depth_interpolation=a.source_depth_interpolation,
        minimum_valid_footprint=.5 if a.source_depth_interpolation=='normalized' else None,
        rgb_and_semantic_sampling_unchanged=True,depth_field_unchanged=True)
    atomic_json(out/'complete.json',receipt)


if __name__=='__main__':main()
