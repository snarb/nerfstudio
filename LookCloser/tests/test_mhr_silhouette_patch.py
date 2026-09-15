import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import admit_mhr_silhouette_patch as wrapper


def test_final_prior_configuration_does_not_replace_guards(monkeypatch):
    module=wrapper.admission
    functions={name:getattr(module,name) for name in ['certificates','initial_admission','native_guard','footprint_veto','measured_pixel_veto','interpolation_admission']}
    for name in ['OUT','CANDIDATES','PRIOR','ARMS']:monkeypatch.setattr(module,name,getattr(module,name))
    wrapper.configure()
    assert module.PRIOR==wrapper.CANDIDATES/'prior'
    assert module.ARMS==['silhouette100']
    assert all(getattr(module,name) is fn for name,fn in functions.items())


def test_new_domain_keeps_gap_gate_and_excludes_unsafe_parents():
    import inspect
    from build_mhr_silhouette_patch_candidates import build
    source=inspect.getsource(build)
    assert 'local&(gap>=.00002)' in source
    assert 'band&~bad' in source
    assert "bad[np.unique(topology['intersections'])]=True" in source


def test_occlusion_sign_excludes_missing_rays():
    import numpy as np
    from localize_mhr_silhouette_patch_occlusion import depth_statistics
    delta,stats=depth_statistics(np.array([[1.,1.,0.]]),np.array([[.985,1.004,1.]]))
    np.testing.assert_allclose(delta,[[-.015,.004,0.]])
    assert stats['nearer_over_003']==stats['nearer_over_01']==stats['farther_over_003']==1
    assert stats['common_changed']==2
