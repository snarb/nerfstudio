from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from source_bandwidth_prior import bandwidth_observations,bandwidth_source_costs


def texture():
    rng=np.random.default_rng(22)
    return cv2.GaussianBlur(rng.random((480,640)).astype(np.float32),(0,0),.8)


def test_blurred_source_gets_penalty_but_rgb_is_unchanged():
    a=texture();b=cv2.GaussianBlur(a,(0,0),1.)
    rgb=np.repeat(np.stack([a,b])[...,None],3,-1);original=rgb.copy()
    costs,stats=bandwidth_source_costs(rgb,np.ones(rgb.shape[:-1],bool))
    assert not costs[0].any() and np.all(costs[1]>0)
    assert stats['sources'][0]['qualified'] and not stats['modifies_rgb']
    np.testing.assert_array_equal(rgb,original)


def test_equal_bandwidth_translation_does_not_create_blur_prior():
    a=texture()
    # Opposite half shifts impose identical interpolation kernels on both inputs.
    a1=cv2.warpAffine(a,np.array([[1,0,-.35],[0,1,.2]],np.float32),(640,480))
    a2=cv2.warpAffine(a,np.array([[1,0,.35],[0,1,-.2]],np.float32),(640,480))
    valid=np.ones(a.shape,bool)
    rows=bandwidth_observations(a1,a2,valid,valid)
    assert len(rows)>40
    assert abs(np.median([r['relative_blur_variance'] for r in rows]))<.2


def test_insufficient_overlap_keeps_zero_prior():
    a=texture();rgb=np.repeat(np.stack([a,a])[...,None],3,-1)
    valid=np.ones(rgb.shape[:-1],bool);valid[0]=False
    costs,stats=bandwidth_source_costs(rgb,valid)
    assert not costs.any() and not stats['sources'][0]['qualified']


def test_primary_can_be_penalized_only_in_opt_in_signed_mode():
    sharp=texture();soft=cv2.GaussianBlur(sharp,(0,0),1.)
    rgb=np.repeat(np.stack([soft,sharp])[...,None],3,-1);original=rgb.copy()
    valid=np.ones(rgb.shape[:-1],bool)
    legacy,_=bandwidth_source_costs(rgb,valid)
    assert not legacy.any()
    costs,stats=bandwidth_source_costs(rgb,valid,allow_primary_penalty=True)
    assert np.all(costs[0]>0) and not costs[1].any()
    assert not stats['primary_exempt'] and stats['sources'][0]['qualified']
    np.testing.assert_array_equal(rgb,original)


def test_unqualified_source_is_not_rewarded_as_sharper_than_primary(monkeypatch):
    import source_bandwidth_prior as module
    observations=[{'relative_blur_variance':-1.,'held':i%5==0} for i in range(50)]
    rows=iter([observations,[]])
    monkeypatch.setattr(module,'bandwidth_observations',lambda *args,**kwargs:next(rows))
    costs,_=module.bandwidth_source_costs(np.ones((3,4,5,3)),np.ones((3,4,5),bool),allow_primary_penalty=True)
    np.testing.assert_array_equal(costs[0],costs[2])
    assert costs[0].min()>0 and not costs[1].any()
