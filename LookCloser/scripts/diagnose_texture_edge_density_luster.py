"""Evaluation-only association of sparse photo edges and transfer error.

The feature measures gradient occupancy, not gradient direction or warp
curvature. No candidate, source labels, gate or confidence cache is emitted.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import read, write, sha
from supported_luster_hybrid import geometric_support
from risk_single_luster_hybrid import masked_low
from transfer_luster_hybrid import gaussian
from diagnose_transfer_regions_luster_hybrid import summarize


def feature(rgb, support):
    low,_=masked_low(np.asarray(rgb),support,1.)
    gray=low@np.array([.2126,.7152,.0722],np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    safe=cv2.erode(support.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    amplitude=np.sqrt(gx*gx+gy*gy)
    mass=gaussian(safe.astype(np.float32),8.)
    mean=gaussian(amplitude*safe,8.)/np.maximum(mass,1e-8)
    energy=gaussian(amplitude*amplitude*safe,8.)/np.maximum(mass,1e-8)
    occupancy=np.clip(mean*mean/np.maximum(energy,1e-12),0,1)
    known=safe&(mass>=.8)&(energy>=.002**2)
    assert np.isfinite(occupancy).all()
    return occupancy,known


def run(root,name):
    candidate=root/'candidates'/name
    config=read(candidate/'config.json'); results=[]
    for target in [7,12,95,150]:
        prefix=f'cam{target:03d}'
        lp=candidate/f'{prefix}_sources.npy';wp=candidate/f'{prefix}_confidence.npy'
        labels=np.load(lp,mmap_mode='r');weight=np.load(wp,mmap_mode='r')
        value=np.zeros(labels.shape,np.float32);known=np.zeros(labels.shape,bool)
        inputs={str(p):sha(p) for p in [lp,wp,candidate/f'{prefix}_receipt.json']}
        for source in np.unique(labels[labels>=0]):
            d=root/'variants/cubic75/warps'/prefix/f'src{source:03d}'
            r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            v,k=feature(a['rgb'],geometric_support(a,config))
            selected=labels[sl]==source
            value[sl][selected]=v[selected];known[sl][selected]=k[selected]
            inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        # Evaluation input is read only after all prediction-only features exist.
        mp=root/'evaluation'/f'{name}_spatial'/f'{prefix}_lpips_maps.npz'
        inputs[str(mp)]=sha(mp)
        with np.load(mp) as maps:
            x0,y0,x1,y1=map(int,maps['crop']);sl=np.s_[y0:y1,x0:x1]
            active=maps['mask']&(weight[sl]>0);reliable=active&known[sl]
            baseline=maps['baseline'];hybrid=maps['hybrid'];delta=hybrid-baseline
            bins=[0,.1,.2,.4,.6,.8,1.0001];groups=[]
            for lo,hi in zip(bins[:-1],bins[1:]):
                selected=reliable&(value[sl]>=lo)&(value[sl]<hi)
                groups.append(dict(min_occupancy=lo,max_occupancy=hi,**summarize(selected,delta,baseline,hybrid)))
            assert sum(g['pixels'] for g in groups)==int(reliable.sum())
            unknown=summarize(active&~known[sl],delta,baseline,hybrid)
        row=dict(camera=target,groups=groups,unknown=unknown,inputs=inputs)
        results.append(row)
        print({k:v for k,v in row.items() if k!='inputs'},flush=True)
        write(root/'evaluation'/f'{name}_texture_edge_density.json',dict(
            candidate=name,usage='evaluation_only_never_construction',complete=len(results)==4,
            feature='E(|gradient G1(photo)|)^2 / E(|gradient G1(photo)|^2), support-normalized G8 window',
            code_sha256=sha(Path(__file__)),rows=results,
            limitation='Spatial LPIPS receptive fields cross groups; associations are not causal per-pixel error labels.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',default='single_photometric_depth25_seam');a=p.parse_args();run(a.root,a.name)
