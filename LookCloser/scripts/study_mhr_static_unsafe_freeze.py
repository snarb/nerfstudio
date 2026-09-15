"""One static unsafe-vertex restriction of the identical zero-margin objective."""
from pathlib import Path
import argparse
import inspect
import numpy as np
import study_mhr_zero_margin as zero
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_static_unsafe_freeze')
TEN=ROOT/'control10'
FINAL=ROOT/'fit100'
FAILED=zero.FINAL
fit=zero.fit
continuation=zero.continuation


def derive_frozen(triangles,active,baseline_pairs,final_pairs,reversed_triangles):
    inherited=set(map(tuple,baseline_pairs))
    new=np.asarray([p for p in final_pairs if tuple(p) not in inherited],int).reshape(-1,2)
    facets=np.union1d(np.unique(new),reversed_triangles)
    vertices=np.unique(triangles[facets])
    return vertices[active[vertices]],facets,new


def selection():
    a=np.load(FAILED/'fit.npz')
    before=np.load(FAILED/'review_v2/baseline_topology.npz')
    after=np.load(FAILED/'review_v2/silhouette_topology.npz')
    ids,facets,pairs=derive_frozen(a['triangles'],a['active'],before['strict_pairs'],after['strict_pairs'],after['reversed_triangles'])
    assert len(ids)==120 and a['active'].sum()==1566
    return ids,facets,pairs


def optimizer_replacements():
    return [
        ('excess = np.maximum(values-2, 0)','excess = np.maximum(values-0, 0)',1),
        ('    ids = np.flatnonzero(active)\n    n, count = len(base), len(ids)',
         '    full_ids = np.flatnonzero(active)\n    ids = np.setdiff1d(full_ids, FROZEN_IDS, assume_unique=True)\n    n, count, free_count = len(base), len(full_ids), len(ids)\n    slot = np.full(n, -1, int); slot[full_ids] = np.arange(count)\n    free_columns = (3*slot[ids, None]+np.arange(3)).ravel()\n    assert count == 1566 and free_count == 1446',1),
        ('uniform_laplacian(triangles, n)[ids][:, ids]','uniform_laplacian(triangles, n)[full_ids][:, ids]',1),
        ('sparse.eye(count*3, format=\'csr\')','sparse.eye(free_count*3, format=\'csr\')',1),
        ('silhouette_samples(current[ids], rows, sdfs)','silhouette_samples(current[full_ids], rows, sdfs)',1),
        ("sil = sparse.coo_matrix((vv, (ri, ci)), shape=(cursor, count*3)).tocsr()",
         "sil = sparse.coo_matrix((vv, (ri, ci)), shape=(cursor, count*3)).tocsr()[:, free_columns]",1),
        ('np.testing.assert_array_equal(current[~active], base[~active])',
         'np.testing.assert_array_equal(current[~active], base[~active])\n        np.testing.assert_array_equal(current[FROZEN_IDS], base[FROZEN_IDS])',1),
        ('silhouette_stats(current[ids], rows, sdfs)','silhouette_stats(current[full_ids], rows, sdfs)',1)]


def configure():
    zero.configure()
    frozen,facets,pairs=selection()
    fit.__dict__['FROZEN_IDS']=frozen
    fit.optimize,optimizer_proof=zero.adapter(zero.FROZEN_OPTIMIZE,optimizer_replacements(),fit.__dict__)
    continuation.CONTROL=TEN;continuation.ROOT=FINAL
    namespace=dict(continuation.__dict__,RESTRICTED_SOURCE=optimizer_proof['generated_source'])
    continuation.make_optimizer,observer_proof=zero.adapter(zero.FROZEN_MAKE,[
        ('excess = np.maximum(values-2,0)','excess = np.maximum(values-0,0)',1),
        ("current[context['ids']], context['rows']", "current[context['full_ids']], context['rows']",1),
        ('source = inspect.getsource(FROZEN_OPTIMIZER)','source = RESTRICTED_SOURCE',1)],namespace)
    return dict(wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__),
        zero_helper_path=str(Path(zero.__file__).resolve()),zero_helper_sha256=sha(zero.__file__),
        frozen_fit_path=str(Path(fit.__file__).resolve()),frozen_fit_sha256=sha(fit.__file__),
        frozen_continuation_path=str(Path(continuation.__file__).resolve()),frozen_continuation_sha256=sha(continuation.__file__),
        selection_source=str(FAILED),selection_seal_sha256=sha(zero.ROOT/'final_seal.json'),
        frozen_vertex_ids=frozen.tolist(),unsafe_parent_ids=facets.tolist(),new_crossing_pairs=pairs.tolist(),
        original_active_count=1566,free_count=1446,frozen_count=120,
        frozen_coordinate_reference='original smooth100 base, not failed zero-offset vertices',
        normalization_and_sample_rows=1566,laplacian_rows='all original 1566 active vertices',
        silhouette_rows='all available original active vertices, including constant fixed rows',
        optimizer=optimizer_proof,observer=observer_proof,stop=continuation.STOP,
        target_used=False,production_modified=False,adaptive_exclusion_iterations=0)


def fit_stage(steps):
    proof=configure()
    if steps==10:
        seal=read(zero.ROOT/'final_seal.json');assert seal['status']=='passed'
        for p,h in seal['inventory'].items():assert sha(zero.ROOT/p)==h,p
        for p,h in seal['checked_bindings'].items():assert sha(p)==h,p
    original_save=fit.save
    def annotated(path,value):
        if Path(path).name=='protocol.json':value=dict(value,static_freeze_adapter=proof)
        original_save(path,value)
    fit.save=annotated
    try:
        if steps==10:
            fit.ROOT=TEN;fit.RECIPE=dict(fit.RECIPE,outer_iterations=10);fit.main()
        else:continuation.main()
    finally:fit.save=original_save


def review():
    configure();fit.ROOT=FINAL
    import review_mhr_silhouette_conformance as native
    import probe_mhr_silhouette_locality as locality
    native.main();locality.main()
    save(ROOT/'review_wrapper.json',dict(wrapper_sha256=sha(__file__),root=str(FINAL),target_used_posthoc_only=True,
        helpers={str(Path(m.__file__).resolve()):sha(m.__file__) for m in [native,locality]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['fit10','fit100','review']);args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    fit_stage(10) if args.stage=='fit10' else fit_stage(100) if args.stage=='fit100' else review()
