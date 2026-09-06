from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_multitime_epipolar_models import normalize_pixels,fit_epipolar_model,balanced_indices
from audit_fixed_camera_feature_geometry import epipolar_distance


def test_normalization_and_block_cap():
    k=np.array([[100.,0,20],[0,200.,30],[0,0,1]])
    np.testing.assert_allclose(normalize_pixels(np.array([[30,70],[20,30]]),k),[[.1,.2],[0,0]])
    points=np.c_[np.arange(40),np.arange(40)]
    ids=balanced_indices(points,[(i//20,0) for i in range(40)])
    assert len(ids)==16 and len(set(ids))==16 and sum(ids<20)==8


@pytest.mark.parametrize('kind',['essential','fundamental'])
def test_true_camera_epipolar_geometry_predicts_unseen_points(kind):
    rng=np.random.default_rng(22);world=rng.uniform([-1,-1,3],[1,1,6],(200,3))
    k=np.array([[800.,0,320],[0,800.,240],[0,0,1.]])
    a=world@k.T;b=(world+[-.2,.03,0])@k.T
    a=a[:,:2]/a[:,2:];b=b[:,:2]/b[:,2:]
    matrix,stats=fit_epipolar_model(a[:100],b[:100],k,k,kind)
    assert stats['fit_points']==100 and np.max(abs(epipolar_distance(matrix,a[100:],b[100:])))<1e-3


def test_invalid_correspondences_fail_closed():
    with pytest.raises(ValueError):fit_epipolar_model(np.zeros((10,2)),np.zeros((10,2)),np.eye(3),np.eye(3),'essential')


def test_duplicate_sift_locations_cannot_repeat_fit_observations():
    points=np.array([[1.,1.],[1.,1.],[2.,2.],[2.,2.],[7.,7.],[7.,7.]])
    ids=balanced_indices(points,[(0,0)]*len(points))
    assert len(ids)==len(set(ids))==3
    assert len(np.unique(points[ids],axis=0))==3
    assert set(ids)=={0,2,4}
