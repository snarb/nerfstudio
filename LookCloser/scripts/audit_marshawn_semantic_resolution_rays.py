"""Exact actor-ray/grid intersection diagnostic, not surface retention proof."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from marshawn_occupancy_audit import ray_intervals

def hits(origins,directions,grid,box):
    # Exact voxel-plane partition, not fixed-spacing ray samples.
    n=len(directions);res=grid.shape[0];device=directions.device;dtype=directions.dtype
    o=origins.expand(n,3);d=directions
    parallel=d.abs()<1e-12;safe=torch.where(parallel,torch.ones_like(d),d)
    t0=(box[0]-o)/safe;t1=(box[1]-o)/safe
    entry=torch.where(parallel,torch.full_like(d,-torch.inf),torch.minimum(t0,t1)).amax(-1).clamp_min(0)
    leave=torch.where(parallel,torch.full_like(d,torch.inf),torch.maximum(t0,t1)).amin(-1)
    floorhit=(leave>entry)&~(parallel&((o<box[0])|(o>=box[1]))).any(-1)
    fractions=torch.linspace(0,1,res+1,device=device,dtype=dtype)
    planes=box[0]+fractions[:,None]*(box[1]-box[0])
    ts=(planes[None]-o[:,None])/safe[:,None]
    ts=torch.where(parallel[:,None],torch.full_like(ts,torch.inf),ts).flatten(1)
    ts=torch.cat([ts,entry[:,None],leave[:,None]],-1)
    ts=torch.maximum(ts,entry[:,None]);ts=torch.minimum(ts,leave[:,None]);ts=ts.sort(-1).values
    mid=(ts[:,:-1]+ts[:,1:])*.5;xyz=o[:,None]+mid[...,None]*d[:,None]
    index=torch.floor((xyz-box[0])/(box[1]-box[0])*res).long().clamp(0,res-1)
    admitted=grid[index[...,0],index[...,1],index[...,2]]
    yes=(admitted&(ts[:,1:]>ts[:,:-1])&floorhit[:,None]).any(-1)
    return floorhit,yes

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(932);testgrid=rng.random((8,8,8))>.7;testbox=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
    origins=rng.uniform(-3,3,(40,3));directions=rng.normal(size=(40,3));directions/=np.linalg.norm(directions,axis=-1)[:,None]
    ft,gt=hits(torch.tensor(origins,device='cuda',dtype=torch.float64),torch.tensor(directions,device='cuda',dtype=torch.float64),torch.tensor(testgrid,device='cuda'),torch.tensor(testbox,device='cuda'))
    for i,(o,d) in enumerate(zip(origins,directions)):
        check=ray_intervals(testgrid[None],testbox.reshape(1,6),o,d)[0]
        assert bool(ft[i])==check['hit'] and bool(gt[i])==any(x['occupied'] for x in check['intervals'])
    ready=json.loads(a.ready.read_text());tf=json.loads(a.transforms.read_text());records={r['camera']:dict(r) for r in ready['records']}
    for r in json.loads(a.corrections.read_text())['records']:records[r['camera']]['semantic']=r['corrected_archive']
    frames=[f for f in tf['frames'] if f['file_path'] in tf['train_filenames']];assert len(frames)==63 and not({f['physical_camera'] for f in frames}&{'DC','HC','KA'})
    with np.load(a.grid) as z:
        keys=['semantic_gate0','semantic_gate_pad1','semantic_gate_pad2'];arrays={k:z[k].copy() for k in keys};box_np=z['aabb'].copy()
    assert np.all(arrays[keys[0]]<=arrays[keys[1]]) and np.all(arrays[keys[1]]<=arrays[keys[2]])
    box=torch.tensor(box_np,device='cuda',dtype=torch.float64);grids={k:torch.tensor(v,device='cuda') for k,v in arrays.items()};rows=[];misses={};started=time.monotonic()
    with torch.inference_mode():
        for f in frames:
            camera=f['physical_camera'];r=records[camera]
            with np.load(r['semantic']) as z:mask=z['black']|z['grey'];black=z['black'].copy()
            ys,xs=np.where(mask);pose=torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float64);origin=pose[:3,3];failed={k:[] for k in ['accepted_box',*keys]}
            for offset in range(0,len(xs),a.chunk):
                x=torch.tensor(xs[offset:offset+a.chunk],device='cuda',dtype=torch.float64);y=torch.tensor(ys[offset:offset+a.chunk],device='cuda',dtype=torch.float64)
                local=torch.stack([(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-torch.ones_like(x)],-1);d=local@pose[:3,:3].T;d/=d.norm(dim=-1,keepdim=True)
                floor,yes=hits(origin,d,grids[keys[0]],box);failed['accepted_box'].extend((torch.where(~floor)[0]+offset).cpu().tolist());left=torch.where(floor&~yes)[0];failed[keys[0]].extend((left+offset).cpu().tolist())
                for k in keys[1:]:
                    if len(left):
                        _,yes=hits(origin,d[left],grids[k],box);left=left[~yes]
                        failed[k].extend((left+offset).cpu().tolist())
            counts={k:len(ids) for k,ids in failed.items()};by_actor={k:dict(black=int(black[ys[ids],xs[ids]].sum()),grey=int(len(ids)-black[ys[ids],xs[ids]].sum())) for k,ids in failed.items()}
            for k,ids in failed.items():misses[camera+'_'+k]=np.stack([xs[ids],ys[ids]],-1)
            rows.append(dict(camera=camera,actor_rays=len(xs),miss_counts=counts,miss_by_actor=by_actor,semantic_source=r['semantic'],semantic_sha256=hashlib.sha256(Path(r['semantic']).read_bytes()).hexdigest()))
            progress=dict(camera=camera,completed=len(rows),seconds=time.monotonic()-started,miss_counts=counts);(a.output/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    np.savez_compressed(a.output/'misses.npz',**misses)
    receipt=dict(kind='EXACT_FULL_TRAIN_RAY_GRID_INTERSECTION_NOT_SURFACE_CERTIFICATION',grid=str(a.grid),grid_sha256=hashlib.sha256(a.grid.read_bytes()).hexdigest(),resolution=list(arrays[keys[0]].shape),rows=rows,total_actor_rays=sum(r['actor_rays'] for r in rows),totals={k:sum(r['miss_counts'][k] for r in rows) for k in ['accepted_box',*keys]},no_eval_input=True,no_field_loaded=True,CPU_DDA_40_ray_crosscheck_passed=True,seconds=time.monotonic()-started,limitation='One retained cell along a ray does not certify actor surface preservation, depth, or occupancy correctness.')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['totals']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['grid','ready','transforms','corrections','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--chunk',type=int,default=1024)
    main(p.parse_args())
