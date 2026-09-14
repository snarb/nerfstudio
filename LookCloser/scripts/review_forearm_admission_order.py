"""Compare shape-first admission to the exact-replayed plane-first control."""
from pathlib import Path
import argparse
import review_confidence_boundary_completion as scorer
from joint_temporal_texture import read,sha,atomic_json


def run(output,frames):
    scorer.ROOTS['plane_first']=Path('/mnt/data/dec5_forearm_contrastive_guard_supported')
    scorer.ROOTS['shape_first']=Path('/mnt/data/dec5_forearm_admission_quadric_bounded')
    for frame in frames:
        plane=Path('/mnt/data/dec5_forearm_admission_plane_bounded')/frame
        for name in ['transferred.ply','guarded.ply']:
            if sha(plane/name)!=sha(scorer.ROOTS['plane_first']/frame/name):raise ValueError('Plane control failed exact mesh replay')
    scorer.run(output,frames,variants=['plane_first','shape_first'])
    result=read(output/'metrics.json');result.update(wrapper_sha256=sha(__file__),baseline_mesh_replay_exact=True)
    atomic_json(output/'metrics.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_admission_order_review'))
    a=p.parse_args();run(a.output,a.frames)
