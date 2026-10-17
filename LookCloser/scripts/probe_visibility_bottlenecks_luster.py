"""Prediction-only availability diagnostic; never relaxes a production gate."""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write,sha,assert_source,native_rows
from supported_luster_hybrid import geometric_support


def run(root):
    config=read(root/'candidates/single_photometric_depth25/config.json')
    cases={'reference':{},'alpha095':{'alpha':.95},'depth010':{'depth_error':.01},
           'roundtrip2':{'roundtrip':2.},'uncertainty4':{'uncertainty':4.},
           'angle60':{'max_score':2.},'resolution03':{'min_resolution':.3}}
    rows=[];native=native_rows()
    for target in [7,12,95,150]:
        parent=root/'candidates/single_photometric_depth25_seam'
        receipt=parent/f'cam{target:03d}_receipt.json';r=read(receipt)
        g=geometry(root,target);labels=np.load(parent/f'cam{target:03d}_sources.npy',mmap_mode='r')
        inputs={str(receipt):sha(receipt)};unions=None
        for source in r['sources']:
            assert_source(native,target,source)
            d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
            wr=read(d/'receipt.json');x0,y0,x1,y1=wr['box']
            if unions is None:
                unions={k:np.zeros((y1-y0,x1-x0),bool) for k in cases};box=wr['box']
            assert wr['box']==box
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in wr['arrays'] if k!='rgb'}
            for key,change in cases.items():unions[key]|=geometric_support(a,{**config,**change})
            inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        domain=np.isfinite(g['depth_z_q50'][y0:y1,x0:x1])&(g['alpha'][y0:y1,x0:x1]>=.95)
        fallback=domain&(labels[y0:y1,x0:x1]<0)
        result={}
        for key,valid in unions.items():
            newly=domain&valid&~unions['reference']
            result[key]=dict(geometrically_available_pixels=int((domain&valid).sum()),
                existing_fallback_with_some_source=int((fallback&valid).sum()),
                new_geometric_pixels=int(newly.sum()),new_fraction=float(newly.sum()/max(1,domain.sum())),
                new_fallback_pixels=int((newly&fallback).sum()))
        row=dict(camera=target,domain_pixels=int(domain.sum()),fallback_pixels=int(fallback.sum()),cases=result,inputs=inputs)
        rows.append(row);print(row,flush=True)
    write(root/'preflight/visibility_bottlenecks.json',dict(usage='prediction_only_diagnostic_not_construction',
        code_sha256=sha(Path(__file__)),domain='finite frozen q50 and target alpha >= .95',
        cases=cases,rows=rows,reads_rgb=False,uses_masks=False,
        limitation='Availability is before patch checks and does not establish that relaxed visibility is correct or improves appearance.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
