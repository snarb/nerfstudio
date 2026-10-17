"""Evaluation-only association of source boundaries and local LPIPS changes.

Never creates labels, confidence or RGB for construction. Spatial LPIPS has
large receptive fields, so a group's errors cannot be attributed causally to
the pixels in that group.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import read,sha,write


def summarize(selection,delta,baseline,hybrid):
    n=int(selection.sum())
    return dict(pixels=n,mean_delta=float(delta[selection].mean()) if n else None,
        baseline_mean=float(baseline[selection].mean()) if n else None,
        hybrid_mean=float(hybrid[selection].mean()) if n else None,
        worse_fraction=float((delta[selection]>0).mean()) if n else None)


def run(root,name):
    candidate=root/'candidates'/name;spatial=root/'evaluation'/f'{name}_spatial';rows=[]
    for target in [7,12,95,150]:
        prefix=f'cam{target:03d}';lp=candidate/f'{prefix}_sources.npy';wp=candidate/f'{prefix}_confidence.npy'
        mp=spatial/f'{prefix}_lpips_maps.npz'
        labels=np.load(lp,mmap_mode='r');weight=np.load(wp,mmap_mode='r')
        distance=np.zeros(labels.shape,np.float32)
        for source in np.unique(labels[labels>=0]):
            selected=labels==source
            d=cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,5)
            distance[selected]=d[selected]
        external=cv2.distanceTransform((labels>=0).astype(np.uint8),cv2.DIST_L2,5)
        with np.load(mp) as maps:
            x0,y0,x1,y1=map(int,maps['crop']);sl=np.s_[y0:y1,x0:x1]
            mask=maps['mask'];baseline=maps['baseline'];hybrid=maps['hybrid'];delta=hybrid-baseline
            active=mask&(weight[sl]>0);d=distance[sl];w=weight[sl]
            edges=[0,3,8,16,32,np.inf];groups=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                selection=active&(d>=lo)&(d<hi)
                groups.append(dict(min_source_boundary_distance=lo,max_source_boundary_distance=None if not np.isfinite(hi) else hi,
                    **summarize(selection,delta,baseline,hybrid)))
            assert sum(g['pixels'] for g in groups)==int(active.sum())
            interior=d<external[sl]-.5
            by_kind={k:summarize(s,delta,baseline,hybrid) for k,s in {
                'inactive':mask&~active,'full_weight':active&(w>=.999),
                'faded_weight':active&(w<.999),
                'within_8px_internal_boundary':active&(d<8)&interior,
                'within_8px_external_or_tied_boundary':active&(d<8)&~interior,
                'more_than_32px_from_any_source_boundary':active&(d>=32)}.items()}
            assert by_kind['inactive']['pixels']+by_kind['full_weight']['pixels']+by_kind['faded_weight']['pixels']==int(mask.sum())
        rows.append(dict(camera=target,boundary_distance=groups,regions=by_kind,
            inputs={str(p):sha(p) for p in [lp,wp,mp,candidate/f'{prefix}_receipt.json']}))
        print('transfer_region_diagnostic_complete',target,flush=True)
    write(root/'evaluation'/f'{name}_transfer_regions.json',dict(candidate=name,
        usage='evaluation_only_never_construction',code_sha256=sha(Path(__file__)),
        limitation='LPIPS receptive fields span regions; associations are not causal pixel attribution.',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();run(a.root,a.name)
