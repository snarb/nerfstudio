from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_spatial_temporal_residuals import fit_residual_model,predict_residual,held_block_summary


def fixture():
    rng=np.random.default_rng(12);xy=rng.normal(size=(600,2))*150+[900,500]
    speed=rng.normal(size=600)*3;groups=[(i//200,i//10) for i in range(600)]
    return xy,speed,groups


def test_joint_model_separates_known_spatial_bias_and_timing():
    xy,speed,groups=fixture();residual=.2+(xy[:,0]-900)*.002-(xy[:,1]-500)*.001+speed*.3
    model=fit_residual_model(xy[:400],speed[:400],residual[:400],groups[:400],spatial=True,timing=True,ridge=1e-6)
    assert abs(model['motion_coefficient_available_frames']-.3)<1e-7
    np.testing.assert_allclose(predict_residual(model,xy[400:],speed[400:]),residual[400:],atol=1e-6)


def test_spatial_bias_without_timing_does_not_create_lag():
    xy,speed,groups=fixture();residual=.6+(xy[:,1]-500)*.002
    model=fit_residual_model(xy,speed,residual,groups,spatial=True,timing=True,ridge=1e-6)
    assert abs(model['motion_coefficient_available_frames'])<1e-7


def test_same_block_duplicates_do_not_reweight_the_fit():
    xy,speed,groups=fixture();residual=.3*speed+np.sin(xy[:,0]/100)
    a=fit_residual_model(xy,speed,residual,groups,spatial=True,timing=True)
    index=np.r_[np.arange(600),np.tile(np.arange(10),30)]
    b=fit_residual_model(xy[index],speed[index],residual[index],[groups[i] for i in index],spatial=True,timing=True)
    np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-10)


def test_held_values_are_not_fit_normalization_inputs():
    xy,speed,groups=fixture();residual=speed*.2
    model=fit_residual_model(xy[:400],speed[:400],residual[:400],groups[:400],spatial=True,timing=True)
    before=dict(model);predict_residual(model,xy[400:]+1e6,speed[400:]*1000)
    assert model==before


def test_block_balanced_summary_and_nonfinite_guard():
    a=held_block_summary(np.array([1.,3.,3.]),[(0,0),(0,1),(0,1)])
    assert a['block_median_absolute_error']==2
    xy,speed,groups=fixture();xy[0,0]=np.nan
    with pytest.raises(ValueError):fit_residual_model(xy,speed,speed,groups,spatial=True)


def test_secondary_response_coordinates_cannot_be_spatial_predictors():
    xy,speed,groups=fixture()
    with pytest.raises(ValueError):fit_residual_model(np.c_[xy,xy],speed,speed,groups,spatial=True)
