"""Evaluation-only association of photo-gradient-weighted projection curvature and spatial LPIPS error.

This does not produce texture labels, gates or candidate images. Associations
of broad-receptive-field LPIPS maps are diagnostic, never causal pixel errors.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read,sha,write,log
from photometric_curvature_luster_hybrid import feature
from edge_coherence_luster_hybrid import feature as edge_feature
from luster_hybrid_common import geometry


def run(root,name):
    candidate=root/'candidates'/name;config=read(candidate/'config.json')
    out=root/'evaluation'/f'{name}_photometric_curvature';out.mkdir(exist_ok=True)
    rows=[];edges=[0,.0005,.001,.002,.005,.01,.02,np.inf]
    for target in [7,12,95,150]:
        labels=np.load(candidate/f'cam{target:03d}_sources.npy',mmap_mode='r')
        full=np.zeros(labels.shape,np.float32);known=np.zeros(labels.shape,bool)
        inputs={str(p):sha(p) for p in [candidate/f'cam{target:03d}_sources.npy',candidate/f'cam{target:03d}_receipt.json']}
        for source in np.unique(labels[labels>=0]):
            value,valid,box=feature(root,target,int(source),config)
            x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];selected=labels[sl]==source
            full[sl][selected]=value[selected];known[sl][selected]=valid[selected]
            p=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json';inputs[str(p)]=sha(p)
        coherence,strength=edge_feature(geometry(root,target)['rgb'])
        path=root/'evaluation'/f'{name}_spatial'/f'cam{target:03d}_lpips_maps.npz';inputs[str(path)]=sha(path)
        with np.load(path) as data:
            x0,y0,x1,y1=map(int,data['crop']);sl=np.s_[y0:y1,x0:x1];mask=data['mask']&(labels[sl]>=0)
            delta=data['hybrid']-data['baseline'];value=full[sl];valid=known[sl]
            bins=[]
            edge=(coherence[sl]>=.8)&(strength[sl]>=.002)
            for group,selection in [('coherent_edge',edge),('other',~edge)]:
                for lo,hi in zip(edges[:-1],edges[1:]):
                    selected=mask&valid&selection&(value>=lo)&(value<hi);n=int(selected.sum())
                    bins.append(dict(group=group,min=lo,max=None if not np.isfinite(hi) else hi,pixels=n,
                        mean_lpips_delta=float(delta[selected].mean()) if n else None,
                        positive_delta_fraction=float(np.mean(delta[selected]>0)) if n else None))
            unknown=mask&~valid;n=int(unknown.sum())
            assert sum(b['pixels'] for b in bins)+n==int(mask.sum())
            row=dict(camera=target,bins=bins,unknown_pixels=n,
                unknown_mean_delta=float(delta[unknown].mean()) if n else None,inputs=inputs)
        np.savez_compressed(out/f'cam{target:03d}_feature.npz',value=full,known=known)
        rows.append(row);write(out/'summary.json',dict(candidate=name,usage='evaluation_only_never_construction',
            code_sha256=sha(Path(__file__)),feature_code_sha256=sha(Path(__file__).with_name('photometric_curvature_luster_hybrid.py')),rows=rows))
        log(root,'photometric_curvature_diagnostic_complete',camera=target,candidate=name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();run(a.root,a.name)
