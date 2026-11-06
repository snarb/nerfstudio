import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_marshawn_clothing_traces import inspect_trace

def test_partial_native_band_integrates_tau_and_keeps_foreground_transmittance(tmp_path):
    start=np.array([0.,1.]);end=start+1;sigma=np.ones(2);trans=np.exp(-start);weights=trans*(-np.expm1(-sigma))
    np.savez(tmp_path/'trace.npz',t_start=start,t_end=end,sigma=sigma,transmittance=trans,weights=weights)
    row=dict(origin=[0,0,0],direction=[1,0,0],pixel={'witness_xyz':[1,0,0]},profiles={'native':{'trace':'trace.npz'}})
    result=inspect_trace(tmp_path,row,'native')
    assert result['positive_band_tau']==pytest.approx(.02)
    assert result['positive_band_local_opacity']==pytest.approx(-np.expm1(-.02))
    assert result['positive_band_unconditional_mass']==pytest.approx(np.exp(-.99)*(-np.expm1(-.02)))
    assert result['mass'][3]==pytest.approx(np.exp(-2))
    assert sum(result['mass'])==pytest.approx(1.)

def test_disjoint_samples_do_not_fabricate_surface_band_density(tmp_path):
    start=np.array([.1,1.8]);end=np.array([.2,1.9]);sigma=np.ones(2);trans=np.exp(-np.array([0.,.1]));weights=trans*(-np.expm1(-.1))
    np.savez(tmp_path/'trace.npz',t_start=start,t_end=end,sigma=sigma,transmittance=trans,weights=weights)
    row=dict(origin=[0,0,0],direction=[1,0,0],pixel={'witness_xyz':[1,0,0]},profiles={'native':{'trace':'trace.npz'}})
    result=inspect_trace(tmp_path,row,'native');assert result['positive_band_tau']==0
    assert result['positive_band_unconditional_mass']==0
    assert result['mass'][3]==pytest.approx(np.exp(-.2))
