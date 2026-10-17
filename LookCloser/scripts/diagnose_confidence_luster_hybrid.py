"""Evaluation-only association between existing confidence and spatial error.

Reads target GT-derived LPIPS maps only after candidate construction. Never
outputs source labels, transfer masks or any prediction input. Spatial LPIPS
has a broad receptive field, so these associations are not causal pixel errors.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read,write,sha


def run(root,name):
    candidate=root/'candidates'/name;spatial=root/'evaluation'/f'{name}_spatial'
    edges=np.array([0,.05,.1,.2,.35,.5,.75,1,2,4,np.inf])
    result=[]
    for target in [7,12,95,150]:
        receipt=read(candidate/f'cam{target:03d}_receipt.json')
        maps=np.load(spatial/f'cam{target:03d}_lpips_maps.npz')
        mx0,my0,mx1,my1=map(int,maps['crop'])
        mask=maps['mask'];delta=maps['hybrid']-maps['baseline']
        labels=np.load(candidate/f'cam{target:03d}_sources.npy',mmap_mode='r')
        selected=labels[my0:my1,mx0:mx1]
        fields={k:np.full(mask.shape,np.nan,np.float32) for k in
                ['source_phase','view_phase','roundtrip','uncertainty','resolution','source_surface']}
        for source in receipt['sources']:
            d=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'
            x0,y0,x1,y1=read(d/'receipt.json')['box']
            ax0,ay0=max(x0,mx0),max(y0,my0);ax1,ay1=min(x1,mx1),min(y1,my1)
            if ax0>=ax1 or ay0>=ay1:continue
            local=np.s_[ay0-y0:ay1-y0,ax0-x0:ax1-x0]
            dest=np.s_[ay0-my0:ay1-my0,ax0-mx0:ax1-mx0]
            chosen=selected[dest]==source
            phase=root/'source_phase_quality'/f'cam{target:03d}'/f'src{source:03d}'
            paths={'source_phase':phase/'source_phase_cost.npy','view_phase':phase/'view_phase_cost.npy',
                   'roundtrip':d/'roundtrip.npy','uncertainty':d/'uncertainty.npy',
                   'resolution':d/'resolution_ratio.npy',
                   'source_surface':root/'source_surface_projected'/f'cam{target:03d}'/f'src{source:03d}'/'cost.npy'}
            for key,path in paths.items():
                values=np.load(path,mmap_mode='r')[local]
                fields[key][dest][chosen]=values[chosen]
        rows={}
        for key,values in fields.items():
            support=mask&(selected>=0)&np.isfinite(values)
            assert int(support.sum())==int((mask&(selected>=0)).sum()),key
            bins=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                keep=support&(values>=lo)&(values<hi)
                if not keep.any():continue
                bins.append(dict(lower=float(lo),upper=float(hi) if np.isfinite(hi) else None,
                    pixels=int(keep.sum()),mean_delta=float(delta[keep].mean()),
                    worse_fraction=float((delta[keep]>0).mean()),
                    baseline_lpips=float(maps['baseline'][keep].mean()),
                    hybrid_lpips=float(maps['hybrid'][keep].mean())))
            assert sum(b['pixels'] for b in bins)==int(support.sum()),key
            rows[key]=bins
        result.append(dict(camera=target,fields=rows))
        print('confidence_association_complete',target,flush=True)
    write(root/'evaluation'/f'{name}_confidence_association.json',dict(candidate=name,
        usage='evaluation_only_never_construction',spatial_device=read(spatial/'summary.json')['device'],
        limitation='LPIPS receptive fields span neighboring pixels; associations are not causal.',
        code_sha256=sha(Path(__file__)),rows=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    a=p.parse_args();run(a.root,a.name)
