from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_depth_conditioned_bandwidth import source_projection_geometry,fit_depth_bandwidth,predict_relative_variance


def test_identity_camera_has_unit_variance_scale_and_native_depth():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=100.,cx=20.,cy=15.)
    depth=np.full((30,40),2.)
    z,scale,condition=source_projection_geometry(np.array([[10,10],[22,18]]),depth,camera,[camera])
    np.testing.assert_allclose(z,2.);np.testing.assert_allclose(scale,1.,atol=1e-12)
    np.testing.assert_allclose(condition,1.,atol=1e-12)
    source=dict(camera,fl_x=200.,fl_y=200.)
    _,scale,_=source_projection_geometry(np.array([[10,10]]),depth,camera,[source])
    np.testing.assert_allclose(scale,.25,atol=1e-12)


def test_source_geometry_is_invariant_to_global_rig_rotation_and_translation():
    camera=dict(transform_matrix=np.eye(4).tolist(),fl_x=100.,fl_y=110.,cx=20.,cy=15.)
    pose=np.eye(4);pose[:3,3]=[.1,.03,.2]
    source=dict(camera,transform_matrix=pose.tolist())
    depth=np.full((30,40),2.);xy=np.array([[10,10],[22,18]])
    expected=source_projection_geometry(xy,depth,camera,[source])
    transform=np.array([[0.,0.,1.,2.],[0.,1.,0.,3.],[-1.,0.,0.,4.],[0.,0.,0.,1.]])
    moved_target=dict(camera,transform_matrix=transform.tolist())
    moved_source=dict(source,transform_matrix=(transform@pose).tolist())
    actual=source_projection_geometry(xy,depth,moved_target,[moved_source])
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12)


def synthetic_rows():
    rows=[];depth=[];scales=[]
    rng=np.random.default_rng(40);coeff=np.array([[0.,.1,-.1],[.4,-.2,.05],[-.3,.3,.15],[.1,-.1,.3]])
    for block in range(45):
        q=rng.uniform(.8,1.5,4);scale=rng.uniform(.7,1.3,4)
        value=(coeff[:,0]+coeff[:,1]*q+coeff[:,2]*q*q)*scale
        for i in range(4):
            for j in range(i+1,4):
                for _ in range(3):
                    rows.append(dict(primary_rank=i,source_rank=j,relative_blur_variance=value[j]-value[i],block=[block,0],held=block%5==0))
                    depth.append(1/q);scales.append(scale)
    return rows,np.array(depth),np.array(scales)


def test_native_depth_model_predicts_independent_blocks_better_than_constant():
    rows,z,scales=synthetic_rows()
    constant=fit_depth_bandwidth(rows,z,scales,degree=0,ridge=1e-7)
    quadratic=fit_depth_bandwidth(rows,z,scales,degree=2,ridge=1e-7)
    assert quadratic['held_absolute_error_p90']<1e-6
    assert constant['held_absolute_error_median']>.02
    assert quadratic['fit_spatial_blocks']==36 and quadratic['held_spatial_blocks']==9


def test_fitted_relative_bandwidth_does_not_depend_on_scene_length_units():
    rows,z,scale=synthetic_rows()
    a=fit_depth_bandwidth(rows,z,scale,degree=1)
    b=fit_depth_bandwidth(rows,z*10,scale,degree=1)
    pa,_=predict_relative_variance(a,z,scale);pb,_=predict_relative_variance(b,z*10,scale)
    np.testing.assert_allclose(pa,pb,atol=1e-11)


def test_held_measurements_do_not_affect_coefficients_or_normalization():
    rows,z,scales=synthetic_rows()
    before=fit_depth_bandwidth(rows,z,scales,degree=2)
    for k,r in enumerate(rows):
        if r['held']:r['relative_blur_variance']+=20;z[k]*=2
    after=fit_depth_bandwidth(rows,z,scales,degree=2)
    for key in ['coefficients','inverse_depth_center_fit_only','inverse_depth_scale_fit_only','inverse_depth_bounds_fit_only']:
        np.testing.assert_array_equal(before[key],after[key])


def test_overlap_of_fit_held_blocks_fails():
    rows,z,scales=synthetic_rows();rows[0]['held']=False
    with pytest.raises(ValueError,match='overlap'):fit_depth_bandwidth(rows,z,scales)


def test_model_prediction_is_clamped_to_fit_depth_support():
    rows,z,scales=synthetic_rows();model=fit_depth_bandwidth(rows,z,scales,degree=1)
    bounds=np.asarray(model['inverse_depth_bounds_fit_only'])
    inside,_=predict_relative_variance(model,(1/bounds[:,1])[None],np.ones((1,4)))
    outside,clipped=predict_relative_variance(model,(1/bounds[:,1])[None]/10,np.ones((1,4)))
    np.testing.assert_allclose(inside,outside);assert clipped.all()
