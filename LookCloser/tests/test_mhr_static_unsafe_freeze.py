import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import study_mhr_static_unsafe_freeze as study


def test_static_selection_is_topological_not_spatial():
    tri=np.array([[0,1,2],[2,3,4],[0,4,5]])
    ids,facets,pairs=study.derive_frozen(tri,np.array([1,1,1,1,1,0],bool),np.array([[0,1]]),np.array([[0,1],[1,2]]),np.array([0]))
    np.testing.assert_array_equal(ids,[0,1,2,3,4]);np.testing.assert_array_equal(facets,[0,1,2]);np.testing.assert_array_equal(pairs,[[1,2]])


def test_fixed_columns_preserve_full_operator_energy():
    rng=np.random.default_rng(13);lap=sparse.csr_matrix(rng.normal(size=(7,7)));free=np.array([0,2,3,6]);d=rng.normal(size=(4,3))
    expanded=np.zeros((7,3));expanded[free]=d
    full=sparse.kron(lap,sparse.eye(3),format='csr');cols=(3*free[:,None]+np.arange(3)).ravel()
    np.testing.assert_allclose(full@expanded.ravel(),full[:,cols]@d.ravel(),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(expanded)/np.sqrt(7),np.linalg.norm(d)/np.sqrt(7))


def test_constant_silhouette_rows_are_not_removed():
    jac=np.array([[1.,0.,2.],[0.,5.,0.],[3.,0.,4.]])
    reduced=jac[:,[0,2]];rhs=np.array([1.,9.,2.]);delta=np.array([.1,.2]);full=np.array([.1,0,.2])
    np.testing.assert_array_equal(jac@full-rhs,reduced@delta-rhs)
    assert reduced.shape[0]==3 and (reduced[1]==0).all()


def test_real_count_and_original_laplacian_are_preserved():
    frozen,_,_=study.selection();assert len(frozen)==120
    source=study.zero.replace_exact(study.inspect.getsource(study.zero.FROZEN_OPTIMIZE),study.optimizer_replacements())
    assert 'len(full_ids)' in source and '[full_ids][:, ids]' in source
    assert "shape=(cursor, count*3)).tocsr()[:, free_columns]" in source
    assert 'weights = np.sqrt(4.*robust/(len(rows)*count))/2.' in source
