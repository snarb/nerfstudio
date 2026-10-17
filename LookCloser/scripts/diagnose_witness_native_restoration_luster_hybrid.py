"""Evaluation-only association of native TRAIN witness restoration risk and local LPIPS change.

No construction labels, confidence, RGB or thresholds are written. Signed risk retains positive utility; zero combines neutral/missing evidence.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read,sha,write
from diagnose_transfer_regions_luster_hybrid import summarize


def run(root,name,cameras=(7,12,95,150)):
    candidate=root/'candidates'/name;rows=[]
    for target in cameras:
        prefix=f'cam{target:03d}';lp=candidate/f'{prefix}_sources.npy';wp=candidate/f'{prefix}_confidence.npy'
        mp=root/'evaluation'/f'{name}_spatial'/f'{prefix}_lpips_maps.npz'
        labels=np.load(lp,mmap_mode='r');weight=np.load(wp,mmap_mode='r')
        risk=np.full(labels.shape,np.nan,np.float32)
        inputs={str(p):sha(p) for p in [lp,wp,mp,candidate/f'{prefix}_receipt.json']}
        for source in np.unique(labels[labels>=0]):
            d=root/'witness_native_restoration_projected'/prefix/f'src{source:03d}'
            rp=d/'receipt.json';r=read(rp);cp=d/'cost.npy'
            assert sha(cp)==r['hashes']['cost.npy']
            x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1];chosen=labels[sl]==source
            values=np.load(cp,mmap_mode='r');risk[sl][chosen]=2*(values[chosen]-.5)
            inputs.update({str(p):sha(p) for p in [rp,cp]})
        with np.load(mp) as maps:
            x0,y0,x1,y1=map(int,maps['crop']);sl=np.s_[y0:y1,x0:x1]
            active=maps['mask']&(weight[sl]>0);rr=risk[sl]
            assert np.isfinite(rr[active]).all()
            baseline=maps['baseline'];hybrid=maps['hybrid'];delta=hybrid-baseline
            edges=[-1.0001,-.6,-.4,-.2,-.1,-1e-7,1e-7,.1,.2,.4,.6,1.0001];groups=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                selection=active&(rr>=lo)&(rr<hi)
                groups.append(dict(min_risk=lo,max_risk=hi,**summarize(selection,delta,baseline,hybrid)))
            assert sum(g['pixels'] for g in groups)==int(active.sum())
        rows.append(dict(camera=target,risk_groups=groups,inputs=inputs))
        print('witness_native_restoration_diagnostic_complete',target,flush=True)
    suffix='' if tuple(cameras)==(7,12,95,150) else '_'+'_'.join(f'cam{c:03d}' for c in cameras)
    write(root/'evaluation'/f'{name}_witness_native_restoration_association{suffix}.json',dict(candidate=name,
        usage='evaluation_only_never_construction',code_sha256=sha(Path(__file__)),
        limitation='LPIPS receptive fields span regions; associations are not causal. Negative risk means positive TRAIN restoration utility. Zero risk includes missing evidence.',rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--cameras',nargs='+',type=int,default=[7,12,95,150]);a=p.parse_args();assert set(a.cameras)<={7,12,95,150};run(a.root,a.name,a.cameras)
