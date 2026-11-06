"""TRAIN surface-depth retention under diagnostic support thresholds; no fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from marshawn_occupancy_audit import ray_intervals


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.counts.parent/'receipt.json').read_text())
    assert sha(a.counts)==receipt['counts_sha256']
    with np.load(receipt['grid_path']) as z:base=z['semantic_gate0'].copy();pad=z['semantic_gate_pad1'].copy();box=z['aabb'].copy()
    n=base.shape[0];keys=['W64_overlap_no_unknown_footprint','nominal_actor_overlap_no_unknown_footprint'];counts={}
    with np.load(a.counts) as z:
        for k in keys:
            counts[k]=np.zeros_like(base,dtype=np.uint8);counts[k].reshape(-1)[z['flat_indices']]=z[k]
    grids={'original_gate0':base,'original_pad1':pad}
    for k in keys:
        for threshold in [2,3]:
            g=base&(counts[k]>=threshold);name=k+'_min'+str(threshold)
            grids[name+'_gate0']=g;grids[name+'_pad1']=binary_dilation(g,structure=np.ones((3,3,3),bool))
    q=json.loads(a.queries.read_text());assert q['split']=='train' and len(q['pixels'])==103
    def index(x):
        i=np.floor((np.asarray(x)-box[0])/(box[1]-box[0])*n).astype(int)
        assert ((i>=0)&(i<n)).all();return tuple(i)
    rows=[]
    for p,r in zip(q['pixels'],q['rays']):
        idx=index(p['witness_xyz']);o=np.asarray(r['origin']);d=np.asarray(r['direction']);assert abs(np.linalg.norm(d)-1)<1e-8
        depth=p['query_witness_depth'];projected=o+depth*d;results={}
        for name,g in grids.items():
            intervals=ray_intervals(g[None],box.reshape(1,6),o,d,near=max(0,depth-.02),far=depth+.02)[0]['intervals']
            length=sum(v['t_end']-v['t_start'] for v in intervals if v['occupied'])
            results[name]=dict(witness_cell_retained=bool(g[idx]),pixel_ray_at_witness_depth_retained=bool(g[index(projected)]),band_halfwidth=.02,retained_band_length=length,retained_band_fraction=length/.04)
        rows.append(dict(camera=p['camera_name'],witness_id=p['witness_id'],group=p['diagnostic_group'],actor=p['actor'],region=p['region'],xyz_floor=p['witness_xyz'],index=list(map(int,idx)),counts={k:int(c[idx]) for k,c in counts.items()},variants=results))
    anchors=[]
    for anchor in receipt['anchors']:
        idx=tuple(anchor['index']);anchors.append(dict(name=anchor['name'],label=anchor['label'],xyz_floor=anchor['xyz_floor'],counts={k:int(c[idx]) for k,c in counts.items()},retained={name:bool(g[idx]) for name,g in grids.items()}))
    summaries={name:dict(cells=int(g.sum()),witness_records_retained=sum(r['variants'][name]['witness_cell_retained'] for r in rows),pixel_depth_records_retained=sum(r['variants'][name]['pixel_ray_at_witness_depth_retained'] for r in rows),minimum_band_fraction=min(r['variants'][name]['retained_band_fraction'] for r in rows)) for name,g in grids.items()}
    result=dict(kind='TRAIN_WITNESS_DEPTH_SUPPORT_DIAGNOSTIC',queries=str(a.queries),queries_sha256=sha(a.queries),counts_sha256=sha(a.counts),records=103,unique_witness_ids=len({p['witness_id'] for p in q['pixels']}),summaries=summaries,rows=rows,anchors=anchors,no_field_GPU_or_eval_input=True,no_gate_exported_or_applied=True,limitations=['103 records include repeated views and two declared duplicates, not103 independent3D points.','Geometric band coverage is not field termination mass.','The +/- .02 diagnostic band is not a calibrated uncertainty interval.','Positive support with an UNKNOWN-free rectangle is a conservative semantic criterion, not physical visibility or empty-space truth.'])
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['counts','queries','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
