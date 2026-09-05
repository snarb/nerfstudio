import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from patchmatch_color_calibration import decode_exposed_linear,encode_exposed_linear,apply_camera_gain,solve_relative_gains,grid_basis


def test_inverse_ingest_curve_round_trip():
    values=np.linspace(0,.999,1000)
    np.testing.assert_allclose(encode_exposed_linear(decode_exposed_linear(values)),values,atol=1e-14)


def test_identity_gain_preserves_rgb_and_finite_white():
    rgb=torch.linspace(0,1,99).reshape(3,3,11)
    out=apply_camera_gain(rgb,[1,1,1])
    torch.testing.assert_close(out,rgb,atol=5e-7,rtol=1e-6)
    assert torch.isfinite(out).all()


def test_gain_matches_exposure_change_without_blur():
    rng=np.random.default_rng(2);linear=rng.random((3,20,30))
    rgb=torch.tensor(encode_exposed_linear(linear),dtype=torch.float32)
    gain=np.array([.8,1.2,1.4])
    expected=torch.tensor(encode_exposed_linear(linear*gain[:,None,None]),dtype=torch.float32)
    torch.testing.assert_close(apply_camera_gain(rgb,gain),expected,atol=3e-7,rtol=1e-6)


def test_camera_graph_recovers_response_up_to_gauge():
    response=np.array([[.5,.6,.7],[1,1.1,1.2],[2,2.2,2.4]])
    pairs=[(0,1),(1,2),(0,2)]
    delta=[np.log(response[j]/response[i]) for i,j in pairs]
    gains=solve_relative_gains(3,pairs,delta,[1,1,1])
    corrected=response*gains
    np.testing.assert_allclose(corrected,np.broadcast_to(corrected[0],corrected.shape),atol=1e-10)
    np.testing.assert_allclose(np.log(gains).mean(0),0,atol=1e-12)


def test_disconnected_camera_graph_is_not_calibrated():
    with pytest.raises(ValueError,match='disconnected'):
        solve_relative_gains(3,[(0,1)],[[.1,.1,.1]],[1])


def test_invalid_gain_rejected():
    with pytest.raises(ValueError,match='positive finite'):
        apply_camera_gain(torch.ones((3,2,2)),[1,-1,1])


def test_spatial_basis_matches_torch_interpolation():
    grid=np.arange(12,dtype=float).reshape(3,4)/100
    y,x=np.indices((7,9));uv=np.column_stack((x.ravel(),y.ravel()))
    ids,weights=grid_basis(uv,9,7,4,3)
    sampled=(grid.ravel()[ids]*weights).sum(-1).reshape(7,9)
    expected=torch.nn.functional.interpolate(torch.tensor(grid)[None,None],size=(7,9),mode='bilinear',align_corners=True)[0,0].numpy()
    np.testing.assert_allclose(sampled,expected,atol=1e-15)
    np.testing.assert_allclose(weights.sum(-1),1)


def test_zero_spatial_field_does_not_change_global_correction():
    rgb=torch.rand((3,8,10))
    torch.testing.assert_close(apply_camera_gain(rgb,[.9,1,1.1],np.zeros((3,4))),apply_camera_gain(rgb,[.9,1,1.1]))


def test_spatial_fit_never_uses_held_surface_observations():
    from patchmatch_color_calibration import fit_spatial_exposure
    rng=np.random.default_rng(7);n=2000
    frames=[dict(w=200,h=100),dict(w=200,h=100)]
    uv=rng.random((2,n,2))*[199,99]
    lum=np.stack((.2*uv[0,:,0]/199,-.2*uv[1,:,0]/199))
    held=np.arange(n)%5==0;valid=np.ones((2,n),bool)
    args=(frames,uv,valid,held)
    grid,values,_=fit_spatial_exposure(*args,lum,np.ones((2,3)),[(0,1)],4,3,1.,2.)
    contaminated=lum.copy();contaminated[:,held]+=rng.normal(0,100,(2,held.sum()))
    other,_,_=fit_spatial_exposure(*args,contaminated,np.ones((2,3)),[(0,1)],4,3,1.,2.)
    np.testing.assert_array_equal(other,grid)
    corrected=lum+values
    assert np.median(abs(corrected[0,held]-corrected[1,held]))<.02
    assert np.max(abs(grid))<=np.log(2)


def test_projected_overlap_correction_is_gain_only_and_validity_is_unchanged():
    from patchmatch_color_calibration import correct_projected_exposure
    rgb=torch.full((3,128,128),.45);rgb[:,1::2]=.5
    source=apply_camera_gain(rgb,[1.3]*3)
    validity=[torch.ones((128,128),dtype=torch.bool) for _ in range(2)]
    outputs,stats=correct_projected_exposure([rgb,source],validity,4,3)
    assert stats['validation_pair_l1_after']<stats['validation_pair_l1_before']*.2
    for original,output,grid in zip([rgb,source],outputs,stats['gain_grids']):
        torch.testing.assert_close(output,apply_camera_gain(original,[1]*3,grid))
    assert all(mask.all() for mask in validity)
    assert stats['uses_eval_rgb'] is False and stats['source_averaging'] is False


def test_projected_correction_skips_disjoint_sources():
    from patchmatch_color_calibration import correct_projected_exposure
    images=[torch.full((3,64,64),.5) for _ in range(2)]
    valid=[torch.zeros((64,64),dtype=torch.bool) for _ in range(2)]
    valid[0][:32]=True;valid[1][32:]=True
    out,stats=correct_projected_exposure(images,valid,4,3)
    assert not stats['enabled']
    for actual,expected in zip(out,images):assert actual is expected
