"""Read-only coverage probe for uncertain target depth with strict TRAIN evidence.

No target photographs, evaluation masks, candidate RGB or production gates are
changed. The source-native evidence retains its original strict depth gates.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import assert_source, geometry, native_rows, project, read, sha, unproject, write
from supported_luster_hybrid import geometric_support
from witness_native_restoration_luster_hybrid import sample_supported_scalar


def run(root):
    cfg_path=root/'candidates/single_photometric_depth25/config.json'
    config=read(cfg_path); rows=native_rows(); results=[]
    thresholds=[.4,.45,.5]
    for target in [7,12,95,150]:
        parent=root/'candidates/single_photometric_depth25_seam'
        rp=parent/f'cam{target:03d}_receipt.json'; r=read(rp)
        lp=parent/f'cam{target:03d}_sources.npy'
        labels=np.load(lp,mmap_mode='r'); g=geometry(root,target)
        inputs={str(p):sha(p) for p in [cfg_path,rp,lp,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
        strict_union=None; source_records=[]
        for source in r['sources']:
            assert_source(rows,target,source)
            wd=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
            wr=read(wd/'receipt.json'); x0,y0,x1,y1=wr['box']
            if strict_union is None:
                box=wr['box']; shape=(y1-y0,x1-x0)
                strict_union=np.zeros(shape,bool); relaxed_union=np.zeros(shape,bool)
                evidence_union={str(t):np.zeros(shape,bool) for t in thresholds}
            assert wr['box']==box
            arrays={k:np.load(wd/f'{k}.npy',mmap_mode='r') for k in wr['arrays'] if k!='rgb'}
            strict=geometric_support(arrays,config)
            relaxed=geometric_support(arrays,{**config,'uncertainty':4.})
            strict_union|=strict; relaxed_union|=relaxed
            sd=root/'witness_native_restoration'/f'cam{source:03d}'
            sr=read(sd/'receipt.json')
            assert sr['source']==source and sr['gates']=={k:config[k] for k in sr['gates']}
            for witness in sr['witnesses']: assert_source(rows,target,witness)
            costs=np.load(sd/'cost.npy',mmap_mode='r')
            counts=np.load(sd/'count.npy',mmap_mode='r')
            assert all(sha(sd/k)==v for k,v in sr['hashes'].items())
            new=relaxed&~strict; counters={str(t):0 for t in thresholds}
            for y in range(y0,y1,128):
                stop=min(y+128,y1); sl=np.s_[y-y0:stop-y0]
                yy,xx=np.mgrid[y:stop,x0:x1]; xy=np.stack([xx,yy],-1).astype(np.float32)
                uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[source])
                sampled,known=sample_supported_scalar(costs,counts>=2,uv,sr['box'])
                for threshold in thresholds:
                    admitted=relaxed[sl]&known&(sampled<threshold)
                    evidence_union[str(threshold)][sl]|=admitted
                    counters[str(threshold)]+=int((admitted&new[sl]).sum())
            source_records.append(dict(source=source,new_source_pixels=int(new.sum()),supported_new_source_pixels=counters))
            for p in [wd/'receipt.json',sd/'receipt.json',sd/'cost.npy',sd/'count.npy']:
                inputs[str(p)]=sha(p)
        domain=np.isfinite(g['depth_z_q50'][y0:y1,x0:x1])&(g['alpha'][y0:y1,x0:x1]>=.95)
        fallback=domain&(labels[y0:y1,x0:x1]<0)
        newly=domain&relaxed_union&~strict_union
        supported={}
        for threshold,available in evidence_union.items():
            supported[threshold]=dict(new_union_pixels=int((newly&available).sum()),
                new_union_fraction=float((newly&available).sum()/max(1,domain.sum())),
                fallback_pixels=int((fallback&available).sum()),
                newly_available_fallback_pixels=int((fallback&newly&available).sum()))
        record=dict(camera=target,domain_pixels=int(domain.sum()),
            newly_available_pixels=int(newly.sum()),supported=supported,sources=source_records,inputs=inputs)
        results.append(record)
        print({k:v for k,v in record.items() if k not in ['inputs','sources']},flush=True)
    write(root/'preflight/witness_supported_uncertainty.json',dict(
        usage='prediction_only_coverage_diagnostic_no_construction',rows=results,
        code_sha256=sha(Path(__file__)),target_uncertainty_probe=4.,
        source_evidence='strict original gates; median positive restoration utility; at least two TRAIN witnesses; every scalar interpolation tap known',
        limitation='Source-native corroboration does not certify target correspondence; these are geometry-domain counts, not person coverage or quality gains.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
