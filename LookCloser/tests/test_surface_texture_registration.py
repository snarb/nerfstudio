from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from surface_texture_registration import registration_observations,register_surface_textures


def test_known_subpixel_translation_and_separate_holdout():
    rng=np.random.default_rng(11);a=cv2.GaussianBlur(rng.random((300,400)).astype(np.float32),(0,0),.8)
    b=cv2.warpAffine(a,np.array([[1,0,1.25],[0,1,-.75]],np.float32),(400,300))
    valid=np.ones(a.shape,bool);valid[:8]=False;valid[-8:]=False
    rows=registration_observations(a,b,valid,valid)
    assert any(r['held'] for r in rows) and any(not r['held'] for r in rows)
    np.testing.assert_allclose(np.median([[r['dx'],r['dy']] for r in rows],axis=0),[1.25,-.75],atol=.12)


def test_no_texture_or_no_overlap_produces_no_correction_observations():
    a=np.zeros((100,100),np.float32);valid=np.ones(a.shape,bool)
    assert not registration_observations(a,a,valid,valid)
    a=np.random.default_rng(2).random((100,100)).astype(np.float32)
    assert not registration_observations(a,a,valid,~valid)


def test_surface_registration_preserves_primary_and_visibility():
    import torch
    from render_mesh_image_blend import grid_sample
    torch.set_num_threads(4)
    a=cv2.GaussianBlur(np.random.default_rng(5).random((192,256)).astype(np.float32),(0,0),.8)
    b=cv2.warpAffine(a,np.array([[1,0,1],[0,1,0]],np.float32),(256,192))
    images=[torch.tensor(np.repeat(c[None],3,axis=0)) for c in (a,b)]
    valid=torch.ones((192,256),dtype=torch.bool)
    y,x=torch.meshgrid(torch.arange(192),torch.arange(256),indexing='ij');uv=(x.float(),y.float())
    result,masks,offsets,stats=register_surface_textures(images,[valid,valid],torch.ones((192,256)),images,[uv,uv],grid_sample)
    torch.testing.assert_close(result[0],images[0],rtol=0,atol=0)
    assert torch.equal(masks[1],valid) and stats['source_averaging'] is False
    assert stats['sources'][0]['corrected'] and offsets[0].shape==(2,192,256)
    before=(images[0]-images[1]).abs()[:,20:-20,20:-20].mean()
    after=(images[0]-result[1]).abs()[:,20:-20,20:-20].mean()
    assert after<before*.5
