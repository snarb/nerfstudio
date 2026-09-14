"""Matched RGB/skin controls for depth-hypothesis discrimination."""
from pathlib import Path
import argparse
import review_confidence_boundary_completion as scorer
from joint_temporal_texture import read,sha,atomic_json


def run(output,frames,include_pilot):
    scorer.ROOTS.update(previous=Path('/mnt/data/dec5_forearm_rgb_qualified_curve'),
        comparison_pilot=Path('/mnt/data/dec5_forearm_contrastive_guard'),
        supported_comparison=Path('/mnt/data/dec5_forearm_contrastive_guard_supported'))
    labels=['previous']+(['comparison_pilot'] if include_pilot else [])+['supported_comparison']
    scorer.run(output,frames,variants=labels)
    result=read(output/'metrics.json');result['wrapper_sha256']=sha(__file__)
    atomic_json(output/'metrics.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001037'])
    p.add_argument('--include-pilot',action='store_true')
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_contrastive_review'))
    a=p.parse_args();run(a.output,a.frames,a.include_pilot)
