from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from primary_fallback_source_blending import fallback_weights,blend_fallback_sources


def test_occluded_patch_uses_all_visible_fallbacks_not_just_old_label():
    valid=torch.ones((3,24,64),dtype=torch.bool);valid[0,:,24:48]=False
    labels=torch.zeros((24,64),dtype=torch.long);labels[:,24:48]=1
    weights,band,stats=fallback_weights(valid,labels,torch.ones((24,64)),[1,1,1],radius=8)
    assert not weights[0,:,24:48].any()
    assert bool((weights[1:,8:16,30:42]>.49).all())
    assert torch.equal(weights[0,:,:16],torch.ones((24,16)))
    assert stats['missing_primary_pixels']==24*24


def test_primary_missing_on_another_depth_layer_does_not_blur_foreground():
    valid=torch.ones((2,24,64),dtype=torch.bool);valid[0,:,32:]=False
    labels=torch.zeros((24,64),dtype=torch.long);labels[:,32:]=1
    depth=torch.ones((24,64));depth[:,32:]=2
    weights,band,_=fallback_weights(valid,labels,depth,[1,1],radius=8)
    assert not band[:,:32].any() and torch.equal(weights[0,:,:32],torch.ones((24,32)))


def test_primary_only_and_missing_depth_support_do_not_create_color():
    valid=torch.ones((1,24,32),dtype=torch.bool);valid[:,:,:4]=False
    labels=torch.zeros((24,32),dtype=torch.long);labels[:,:4]=-1
    depth=torch.ones((24,32));depth[:,:4]=0
    weights,band,_=fallback_weights(valid,labels,depth,[1],radius=8)
    assert not band.any() and not weights[:,:,:4].any()
    assert torch.equal(weights[0,:,4:],torch.ones((24,28)))


def test_primary_interior_rgb_is_exactly_unchanged_and_patch_is_mixed():
    torch.set_num_threads(4)
    valid=torch.ones((3,24,64),dtype=torch.bool);valid[0,:,24:48]=False
    labels=torch.zeros((24,64),dtype=torch.long);labels[:,24:48]=1
    sources=[torch.full((3,24,64),v) for v in [.2,.3,.5]]
    out,weights,band,_=blend_fallback_sources(sources,valid,labels,torch.ones((24,64)),[1,1,1],radius=8)
    for value in out.values():
        assert torch.equal(value[:,:,:16],sources[0][:,:,:16])
        assert bool((value[:,8:16,30:42]>.4).all()) and bool((value[:,8:16,30:42]<.42).all())
