from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_marshawn_common_domain import fixed_exposures,display_codes,display_ap1_relative,quantize_png,native_rgb,roundtrip_audit,difference


@pytest.fixture
def vendor():
    path=Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py')
    if not path.exists():pytest.skip('Pinned local vendor source unavailable')
    spec=importlib.util.spec_from_file_location('pq_common_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_exact_signed_gain_inverse_without_black_subtraction(vendor):
    gain=52.13360318135631;raw=np.array([[[.18,.18,.18]]]);code=vendor.pq_encode(raw*gain)
    actual=display_codes(code,vendor,np.eye(3),gain,1.)
    reinhard=(8*.18)/(1+8*.18);expected=1.055*reinhard**(1/2.4)-.055
    assert np.max(np.abs(actual-expected))<1e-12
    floor=display_codes(np.full((1,1,3),vendor.pq_encode(.005)),vendor,np.eye(3),gain,0.)
    assert (floor>0).all() # Incorrect black subtraction would erase this.
    assert (quantize_png(floor)==1).all()


def test_old_fixed_camera_ev_is_not_refitted_or_silently_missing():
    values={'DC':-.2,'HC':-.3,'KA':.1}
    contract=dict(eval_rgb_used_to_fit_gains=False,global_base_gain=4,
        eval_gain_predictions={key:dict(ev=value) for key,value in values.items()},
        rows=[dict(camera=key,exposure_correction_ev=value) for key,value in values.items()])
    assert fixed_exposures(contract)==values
    contract['rows'][0]['exposure_correction_ev']=.5
    with pytest.raises(ValueError,match='mismatch'):fixed_exposures(contract)
    contract['eval_rgb_used_to_fit_gains']=True
    with pytest.raises(ValueError,match='train-derived'):fixed_exposures(contract)


def test_matrix_orientation_positive_gamut_clamp_and_quantization():
    raw=np.array([[[1.,2.,3.]]]);matrix=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    result=display_ap1_relative(raw,matrix,0)
    assert result[0,0,1]>result[0,0,0]>result[0,0,2]==0
    assert quantize_png(np.array([0.,.5,1.])).tolist()==[0,128,255]
    with pytest.raises(ValueError,match='Nonfinite'):quantize_png(np.array([np.nan]))


def test_roundtrip_separates_floor_policy_from_u16_quantization(vendor):
    gain=52.13360318135631;raw=np.array([[[0.,.18,100.]]],np.float32)
    code=np.rint(vendor.encode_tier(raw.astype(float)*gain)*65535.)/65535.
    report=roundtrip_audit(code,raw,vendor,np.eye(3),gain,0)
    assert report['policy_floor_channel_fraction']==pytest.approx(1/3)
    assert report['policy_peak_channel_fraction']==pytest.approx(1/3)
    assert report['code_error_in_u16_units']['max_abs']==0
    assert report['display_quantization_precision_only_8bit_code']['max_abs']<=1


def test_no_preview_upscale_and_psnr_range_is_explicit():
    with pytest.raises(ValueError,match='native1920'):native_rgb(np.zeros((337,640,3)))
    report=difference(np.array([0,255]),np.array([0,254]),255.)
    assert report['psnr_db']==pytest.approx(-10*np.log10(.5/255**2))
