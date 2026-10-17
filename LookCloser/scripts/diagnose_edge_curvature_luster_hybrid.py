"""Evaluation-only conditioning of curvature on frozen-render coherent edges.

No confidence gates, source labels or prediction RGB are written. Spatial LPIPS
associations do not establish causal pixel-level error.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write,sha,log
from edge_coherence_luster_hybrid import feature


def run(root,name):
    out=root/'evaluation'/f'{name}_edge_curvature';out.mkdir(exist_ok=True);rows=[]
    for cid in [7,12,95,150]:
        cdir=root/'candidates'/name
        labels=np.load(cdir/f'cam{cid:03d}_sources.npy',mmap_mode='r')
        coherence,strength=feature(geometry(root,cid)['rgb'])
        curvature_path=root/'evaluation'/f'{name}_warp_curvature'/f'cam{cid:03d}_feature.npz'
        with np.load(curvature_path) as d:value=d['value'];known=d['known']
        maps_path=root/'evaluation'/f'{name}_spatial'/f'cam{cid:03d}_lpips_maps.npz'
        with np.load(maps_path) as d:
            x0,y0,x1,y1=map(int,d['crop']);sl=np.s_[y0:y1,x0:x1]
            mask=d['mask']&(labels[sl]>=0)&known[sl];delta=d['hybrid']-d['baseline']
        edge=(coherence[sl]>=.8)&(strength[sl]>=.002)
        groups=[]
        for group,selection in [('coherent_edge',edge),('other',~edge)]:
            for lo,hi in zip([0,.01,.05,.2],[.01,.05,.2,np.inf]):
                selected=mask&selection&(value[sl]>=lo)&(value[sl]<hi);n=int(selected.sum())
                groups.append(dict(group=group,min_curvature=lo,max_curvature=hi if np.isfinite(hi) else None,pixels=n,
                    mean_lpips_delta=float(delta[selected].mean()) if n else None,
                    positive_delta_fraction=float((delta[selected]>0).mean()) if n else None))
        assert sum(g['pixels'] for g in groups)==int(mask.sum())
        rows.append(dict(camera=cid,groups=groups,inputs={str(p):sha(p) for p in [curvature_path,maps_path,cdir/f'cam{cid:03d}_receipt.json',root/'geometry'/f'cam{cid:03d}'/'receipt.json']}))
        write(out/'summary.json',dict(candidate=name,usage='evaluation_only_never_construction',
              coherence_threshold=.8,gradient_rms_threshold=.002,code_sha256=sha(Path(__file__)),
              feature_code_sha256=sha(Path(__file__).with_name('edge_coherence_luster_hybrid.py')),rows=rows))
        log(root,'edge_curvature_diagnostic_complete',camera=cid,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();run(a.root,a.name)
