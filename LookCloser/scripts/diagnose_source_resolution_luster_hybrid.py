"""Evaluation-only native source sampling coverage inside predefined ROIs."""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read,write,sha


def run(root,region='nails'):
    config=read(root/'candidates/single_cubic_eligible_phase_rt2/config.json')
    protocol=read(root/'evaluation/protocol.json');result=[]
    for target in [7,12,95,150]:
        box=protocol['cameras'][str(target)]['boxes'].get(region)
        if box is None:continue
        x0,y0,x1,y1=box;records=[]
        for d in sorted((root/'warps'/f'cam{target:03d}').glob('src*')):
            if not (d/'receipt.json').exists():continue
            r=read(d/'receipt.json');a,b,c,e=r['box']
            ix0,iy0,ix1,iy1=max(x0,a),max(y0,b),min(x1,c),min(y1,e)
            if ix0>=ix1 or iy0>=iy1:continue
            sl=np.s_[iy0-b:iy1-b,ix0-a:ix1-a]
            valid=np.load(d/'valid.npy',mmap_mode='r')[sl]>0
            for key,limit,lower in [('alpha',config['alpha'],True),('depth_error',config['depth_error'],False),
                ('roundtrip',config['roundtrip'],False),('uncertainty',config['uncertainty'],False),('score',config['max_score'],False)]:
                values=np.load(d/f'{key}.npy',mmap_mode='r')[sl];valid&=(values>=limit if lower else values<=limit)
            ratio=np.load(d/'resolution_ratio.npy',mmap_mode='r')[sl];valid&=ratio>=config['min_resolution']
            if valid.sum()<100:continue
            records.append(dict(source=r['source'],valid_pixels=int(valid.sum()),
                valid_roi_fraction=float(valid.sum()/((x1-x0)*(y1-y0))),
                median_native_scale=float(np.median(ratio[valid])),
                warp_receipt=str(d/'receipt.json'),warp_receipt_sha256=sha(d/'receipt.json')))
        records.sort(key=lambda x:-x['median_native_scale']);result.append(dict(target=target,box=box,sources=records))
        print(dict(target=target,best_scale_sources=[{k:x[k] for k in ['source','median_native_scale','valid_roi_fraction']} for x in records[:5]]),flush=True)
    out=root/'evaluation'/f'{region}_source_sampling.json'
    write(out,dict(usage='evaluation_only_not_source_selection',scope='existing original bilinear warp pool; not all possible cameras',
        region=region,config=config,protocol_sha256=sha(root/'evaluation/protocol.json'),
        code_sha256=sha(Path(__file__)),rows=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--region',default='nails');a=p.parse_args();run(a.root,a.region)
