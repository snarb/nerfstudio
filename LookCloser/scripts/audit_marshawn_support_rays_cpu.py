"""CPU ray coverage under hypothetical positive-support thresholds; no carving."""
import argparse,ctypes,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from marshawn_occupancy_audit import ray_intervals


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_kernel(output):
    source=Path(__file__).with_name('marshawn_semantic_ray_support.cpp');library=output/'ray_support.so'
    command=['g++','-std=c++11','-O3','-shared','-fPIC','-fopenmp',str(source),'-o',str(library)];subprocess.run(command,check=True)
    function=ctypes.CDLL(str(library)).support_ray_maxima
    function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];function.restype=ctypes.c_int
    def evaluate(origins,directions,box,gates,counts):
        directions=np.ascontiguousarray(directions,np.float64);origins=np.ascontiguousarray(np.broadcast_to(origins,directions.shape),np.float64);box=np.ascontiguousarray(box,np.float64);gates=np.ascontiguousarray(gates,np.uint8);counts=np.ascontiguousarray(counts,np.uint8)
        nr=len(directions);channels=counts.shape[1];result=np.empty((2,nr,channels),np.int16);hit=np.empty(nr,np.uint8)
        code=function(origins.ctypes.data,directions.ctypes.data,nr,box.ctypes.data,gates.shape[1],gates[0].ctypes.data,gates[1].ctypes.data,counts.ctypes.data,channels,result.ctypes.data,hit.ctypes.data);assert code==0,code
        return hit.astype(bool),result
    return evaluate,dict(source_sha256=sha(source),library_sha256=sha(library),compile_command=command)


def crosscheck(evaluate):
    rng=np.random.default_rng(2928);n=8;gates=rng.random((2,n,n,n))>.7;counts=rng.integers(0,10,size=(2,4,n,n,n),dtype=np.uint8);box=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
    origins=rng.uniform(-3,3,(300,3));directions=rng.normal(size=(300,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    origins=np.r_[origins,[[-1.,0,0],[1.,0,0],[0,0,0],[-2,-2,-2],[0,1,0],[1,1,1]]];directions=np.r_[directions,[[1.,0,0],[-1.,0,0],[0,1.,0],[1.,1.,1.],[1.,0,0],[-1.,-1.,-1.]]]
    hit,result=evaluate(origins,directions,box,gates,counts)
    for i,(o,d) in enumerate(zip(origins,directions)):
        for group in range(2):
            ref=ray_intervals(gates[group:group+1],box.reshape(1,6),o,d)[0];assert hit[i]==ref['hit']
            expected=np.full(4,-1)
            for interval in ref['intervals']:
                if interval['occupied']:expected=np.maximum(expected,counts[(group,slice(None),*interval['index_xyz'])])
            assert np.array_equal(expected,result[group,i]),(i,group,expected,result[group,i])
    return len(origins)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);evaluate,kernel=load_kernel(a.output);tested=crosscheck(evaluate)
    b=a.base;data=b.parent/'data';audit=a.counts.parent;count_receipt=json.loads((audit/'receipt.json').read_text());assert sha(a.counts)==count_receipt['counts_sha256']
    with np.load(count_receipt['grid_path']) as z:gate0=z['semantic_gate0'].copy();pad1=z['semantic_gate_pad1'].copy();box=z['aabb'].copy()
    keys=['W64_overlap','W64_overlap_no_unknown_footprint','nominal_actor_overlap','nominal_actor_overlap_no_unknown_footprint'];n=gate0.shape[0];counts=np.zeros((2,4,n,n,n),np.uint8)
    with np.load(a.counts) as z:
        for c,key in enumerate(keys):
            counts[0,c].reshape(-1)[z['flat_indices']]=z[key]
            counts[1,c]=maximum_filter(np.where(gate0,counts[0,c],0),size=3,mode='constant',cval=0)
    gates=np.stack([gate0,pad1]);records={r['camera']:dict(r) for r in json.loads((data/'ready.json').read_text())['records']}
    for r in json.loads((b/'corrected_v2/mask_corrections_final.json').read_text())['records']:records[r['camera']].update(semantic=r['corrected_archive'],semantic_sha256=r['corrected_archive_sha256'])
    tf=json.loads((data/'transforms.json').read_text());frames=[r for r in tf['frames'] if r['file_path'] in tf['train_filenames']];assert len(frames)==63 and not({r['physical_camera'] for r in frames}&{'DC','HC','KA'})
    rows=[];misses={};started=time.monotonic()
    for f in frames:
        name=f['physical_camera'];r=records[name];assert sha(r['semantic'])==r['semantic_sha256']
        with np.load(r['semantic']) as z:mask=z['black']|z['grey'];black=z['black'].copy()
        y,x=np.where(mask);pose=np.asarray(f['transform_matrix']);local=np.c_[(x+.5-f['cx'])/f['fl_x'],-(y+.5-f['cy'])/f['fl_y'],-np.ones(len(x))];directions=local@pose[:3,:3].T;directions/=np.linalg.norm(directions,axis=1)[:,None]
        floor,maximum=evaluate(pose[:3,3],directions,box,gates,counts);baseline=[int((floor&(maximum[group,:,0]<0)).sum()) for group in range(2)];variants={}
        for group in range(2):
            for c,key in enumerate(keys):
                for threshold in [2,3]:
                    variant=('filtered_gate0' if group==0 else 'fresh_pad1_after_filter')+'_'+key+'_min'+str(threshold);failed=floor&(maximum[group,:,c]<threshold);xy=np.stack([x[failed],y[failed]],1).astype(np.int32);misses[name+'_'+variant]=xy;variants[variant]=dict(misses=int(failed.sum()),black=int(black[y[failed],x[failed]].sum()),grey=int((~black[y[failed],x[failed]]).sum()))
        row=dict(camera=name,actor_rays=len(x),accepted_box_misses=int((~floor).sum()),original_gate0_misses=baseline[0],original_pad1_misses=baseline[1],hypothetical_variants=variants);rows.append(row);print(json.dumps(dict(camera=name,completed=len(rows),seconds=time.monotonic()-started)),flush=True)
    totals={k:{actor:sum(r['hypothetical_variants'][k][actor] for r in rows) for actor in ['misses','black','grey']} for k in rows[0]['hypothetical_variants']}
    assert sum(r['original_gate0_misses'] for r in rows)==528 and sum(r['original_pad1_misses'] for r in rows)==0
    np.savez_compressed(a.output/'hypothetical_threshold_misses.npz',**misses)
    receipt=dict(kind='CPU_FULL_TRAIN_RAY_SUPPORT_THRESHOLD_DIAGNOSTIC_NO_GATE_APPLIED',kernel=kernel,independent_plane_partition_crosscheck_rays=tested,grid_sha256=sha(count_receipt['grid_path']),counts_sha256=sha(a.counts),actor_rays=sum(r['actor_rays'] for r in rows),baseline_gate0_misses=528,baseline_pad1_misses=0,rows=rows,totals=totals,seconds=time.monotonic()-started,no_field_GPU_or_eval_input=True,no_gate_exported_or_applied=True,definition='Hypothetical gate0=existinggate0 AND supportcount>=k; hypotheticalpad1 is freshonecell dilationofthatfilteredgate0. Neighborhood maximum counts encode this exactly without applying or rendering a gate.',limitations=['Ray intersection is necessary, not proof of retained actor surfaces.','Counts encode source-mask overlaps and uncertainty, not physical empty/occupied labels.'])
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','counts','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
