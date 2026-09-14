"""Fixed train-ROI and matched moving-view review for measured shape residuals."""
from pathlib import Path
import argparse
import review_confidence_boundary_completion as scorer
from joint_temporal_texture import read,sha,atomic_json


def run(output,frames):
    scorer.ROOTS['previous']=Path('/mnt/data/dec5_forearm_rgb_qualified_curve')
    scorer.ROOTS['anchor_residual']=Path('/mnt/data/dec5_forearm_anchor_residual')
    scorer.run(output,frames,variants=['previous','anchor_residual'])
    result=read(output/'metrics.json');result['wrapper_sha256']=sha(__file__)
    atomic_json(output/'metrics.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_anchor_residual_review'))
    a=p.parse_args();run(a.output,a.frames)
