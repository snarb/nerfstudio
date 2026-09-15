"""Exact continuation replay, fixed cohorts, and anatomical failure localization."""
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
import fit_mhr_silhouette_conformance as fit
import continue_mhr_silhouette_convergence as continuation
from study_multiview_face_prior import read,save,sha
from triangulate_face_prior import quantiles

ROOT=continuation.ROOT


def main():
    import open3d as o3d
    assert not (ROOT/'audit.json').exists()
    q=read(ROOT/'protocol.json'); result=read(ROOT/'result.json'); state=read(ROOT/'continuation_result.json'); a=np.load(ROOT/'fit.npz')
    assert sha(continuation.__file__)==q['continuation']['wrapper_sha256']
    assert sha(fit.__file__)==q['script_sha256']
    expected=read(continuation.CONTROL/'protocol.json')['recipe']
    assert {k:v for k,v in q['recipe'].items() if k!='outer_iterations'}=={k:v for k,v in expected.items() if k!='outer_iterations'}
    bindings={}
    def check(path,digest):assert sha(path)==digest,str(path);bindings[str(path)]=digest
    for p,h in q['input_hashes'].items():check(p,h)
    for p,h in q['helpers'].items():check(Path(fit.__file__).with_name(p),h)
    for p,h in result['hashes'].items():check(ROOT/p,h)
    check(q['original_mesh'],q['original_mesh_sha256'])
    _,rows,masks,names,evidence,validation=fit.prepare(); assert evidence==q['evidence']
    obs=np.load(fit.SOURCE/'anchors.npz'); train=~validation[obs['camera']]
    sdfs=[]
    for row in rows:
        mask=masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
    continuation.ROOT=ROOT/'audit_replay';continuation.ROOT.mkdir(exist_ok=False)
    continuation.FROZEN_OPTIMIZER=fit.optimize
    worker,replayed_state=continuation.make_optimizer()
    v,history=worker(a['baseline'],a['triangles'],a['neutral'],obs['points'][train],obs['normals'][train],obs['neck'][train],
                      [r for r,x in zip(rows,validation) if not x],[s for s,x in zip(sdfs,validation) if not x])
    np.testing.assert_array_equal(v,a['vertices']);assert history==result['history']
    for key in ['records','stop_reason','consecutive_small','consecutive_increases','first10_exact']:assert replayed_state[key]==state[key],key
    for iteration in range(1,len(history)+1):
        np.testing.assert_array_equal(np.load(ROOT/'iterates'/f'{iteration:03d}.npz')['vertices'],np.load(continuation.ROOT/'iterates'/f'{iteration:03d}.npz')['vertices'])
    active=a['active'];np.testing.assert_array_equal(v[~active],a['baseline'][~active])
    grids=[]
    for vv in [a['baseline'][active],v[active]]:
        values=np.full((62,len(vv)),np.nan)
        for ci,(ids,s,_) in enumerate(fit.silhouette_samples(vv,rows,sdfs)):values[ci,ids]=s
        grids.append(values)
    b,f=grids; common=np.isfinite(b)&np.isfinite(f);cohorts={}
    for split,selected in [('train',~validation),('validation',validation)]:
        for region,subset in [('all',np.ones(active.sum(),bool)),('front',a['neutral'][active,2]>=0),('back',a['neutral'][active,2]<0)]:
            chosen=common&selected[:,None]&subset[None,:];before=np.maximum(b[chosen]-2,0);after=np.maximum(f[chosen]-2,0)
            cohorts[split+'_'+region]=dict(samples=len(before),before_mean=float(before.mean()),after_mean=float(after.mean()),
                before_p90=float(np.percentile(before,90)),after_p90=float(np.percentile(after,90)),
                before_outside=int((before>0).sum()),after_outside=int((after>0).sum()),
                after_nonzero_excess=quantiles(after[after>0]))
    np.savez_compressed(ROOT/'audit_evidence.npz',before_sdf=b,after_sdf=f,common_available=common,validation=validation)
    topo=np.load(ROOT/'review_v2/silhouette_topology.npz');base_topo=np.load(ROOT/'review_v2/baseline_topology.npz')
    base_pairs=set(map(tuple,base_topo['strict_pairs']));new_pairs=np.array([p for p in topo['strict_pairs'] if tuple(p) not in base_pairs],int).reshape(-1,2)
    tri=a['triangles'];neutral=a['neutral'];centers=neutral[tri].mean(1);rim=np.load(ROOT/'locality/silhouette.npz')['rim_points']
    local=np.load(ROOT/'locality/silhouette.npz')['points'];localids=fit.Scene2(v,tri).compute_closest_points(o3d.core.Tensor(local.astype(np.float32)))['primitive_ids'].numpy()
    localization={}
    for name,ids in [('normal_reversed',topo['reversed_triangles']),('new_strict_crossing',np.unique(new_pairs))]:
        c=centers[ids];scene=fit.Scene2(v,tri[ids]);closest=scene.compute_closest_points(o3d.core.Tensor(rim.astype(np.float32)))['points'].numpy()
        localization[name]=dict(triangles=len(ids),neutral_y=quantiles(c[:,1]),neutral_z=quantiles(c[:,2]),
            neutral_vertex_y_min=float(neutral[tri[ids],1].min()),neutral_vertex_y_max=float(neutral[tri[ids],1].max()),
            front_centroids=int((c[:,2]>=0).sum()),back_centroids=int((c[:,2]<0).sum()),
            minimum_surface_distance_to_requested_rim=float(np.linalg.norm(closest-rim,axis=1).min()),
            requested_first_hit_facets_in_set=int(np.isin(localids,ids).sum()),triangle_ids=ids.tolist())
    for directory,producer in [('review_v2','review_mhr_silhouette_conformance.py'),('locality','probe_mhr_silhouette_locality.py')]:
        receipt=read(ROOT/directory/'result.json');check(Path(__file__).with_name(producer),receipt['script_sha256'])
        for p,h in receipt['input_hashes'].items():check(p,h)
        for row in receipt.get('files',[]):check(row['path'],row['sha256'])
    save(ROOT/'audit.json',dict(status='passed',exact_all_iterates_replayed=True,first10_matches_previous_control=True,
        outside_active_exact=True,fit_cameras=54,reserved_cameras=8,checked_bindings=bindings,
        fixed_availability_cohorts=cohorts,topology_localization=localization,
        evidence_sha256=sha(ROOT/'audit_evidence.npz'),script_sha256=sha(__file__),production_accepted=False))
    print('audit passed',len(history),'exact iterates',flush=True)


if __name__=='__main__':main()
