"""Proposal, confidence and source contracts without constructing a model."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_actor_confidence import weighted_rgb_mse
from marshawn_actor_training_pipeline import (ActorConfidenceStore,actor_crop_rectangle,
    draw_uniform_proposal,file_sha,random_background_opaque_variance,weighted_scalar_mean)


def test_confidence_once_fixed_mean_and_exact_zero_gradient():
    pred=torch.tensor([[.2,.3,.4],[.5,.6,.7],[.8,.9,1.]],requires_grad=True)
    gt=torch.ones_like(pred)*.1
    confidence=torch.tensor([1.,.1,0.])
    before=gt.clone();loss=weighted_rgb_mse(pred,gt,confidence,proposal_mean_weight=.4)
    loss.backward()
    expected=2*(pred.detach()-gt)*confidence[:,None]/(3*3*.4)
    torch.testing.assert_close(pred.grad,expected,rtol=1e-6,atol=1e-8)
    assert torch.equal(gt,before)and torch.equal(pred.grad[-1],torch.zeros(3))
    # A zero-weight packet is valid. Batch self-normalization would be undefined.
    x=torch.tensor([3.,7.],requires_grad=True)
    zero=weighted_scalar_mean(x,torch.zeros(2),.4);zero.backward()
    assert zero==0 and torch.equal(x.grad,torch.zeros(2))


def test_global_pixel_proposal_does_not_equalize_camera_counts():
    # Camera0 has one eligible pixel, camera1 has three. Each pixel is uniform.
    flat=torch.tensor([0,4,5,7]);g=torch.Generator().manual_seed(55)
    indices=draw_uniform_proposal(flat,40000,2,2,generator=g)
    encoded=indices[:,0]*4+indices[:,1]*2+indices[:,2]
    assert set(encoded.tolist())==set(flat.tolist())
    assert abs(float((indices[:,0]==0).float().mean())-.25)<.01
    counts=torch.stack([(encoded==i).sum()for i in flat]).float()
    assert (counts/40000-.25).abs().max()<.01


def fixture_manifest(tmp_path):
    rgb=tmp_path/'untouched.rgb';rgb.write_bytes(b'original RGB source')
    proposal=np.ones((2,2),bool);veto=np.zeros((2,2),bool);proposal[1,1]=False;veto[1,1]=True
    hard=np.array([[1,0],[0,0]],np.float32);W32=hard.copy();W32[0,1]=.1
    W64=W32.copy();W64[1,0]=.2
    archive=tmp_path/'weights.npz';np.savez(archive,hard=hard,W32=W32,W64=W64,proposal=proposal,source_ineligible=veto)
    payload={'rows':[{'camera':'AA','split':'train','rgb':str(rgb),'rgb_sha256':file_sha(rgb),
        'archive':str(archive),'archive_sha256':file_sha(archive)}],
        'normalizer_policy_train':{'hard':1/3,'W32':1.1/3,'W64':1.3/3}}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(payload))
    return manifest,payload


def test_fixed_normalizer_includes_common_zero_weight_pixels(tmp_path):
    manifest,p=fixture_manifest(tmp_path);store=ActorConfidenceStore(manifest,'hard')
    assert store.normalizers['train']['hard']==pytest.approx(1/3)
    assert store.for_dataset([p['rows'][0]['rgb']],'train')[0]['camera']=='AA'
    p['normalizer_policy_train']['hard']=1 # Wrong positive-only normalizer.
    manifest.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='common global proposal'):ActorConfidenceStore(manifest,'hard')


def test_source_corruption_fails_before_training(tmp_path):
    manifest,p=fixture_manifest(tmp_path);Path(p['rows'][0]['rgb']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):ActorConfidenceStore(manifest,'W64')


def test_native_crop_contains_mask_preserves_lattice_and_margin():
    mask=np.zeros((4320,8192),bool);mask[1000:2000,2100:3300]=True
    x0,y0,x1,y1=actor_crop_rectangle(mask)
    assert x0%32==y0%32==0 and x0<=2100-256 and y0<=1000-256
    assert x1>=3300+256 and y1>=2000+256
    assert mask[y0:y1,x0:x1].sum()==mask.sum()


def test_random_background_startup_opacity_bias_is_not_removed_by_confidence():
    alpha=torch.tensor([.2,.8],requires_grad=True)
    penalty=random_background_opaque_variance(alpha)
    (.1*penalty.sum()).backward()
    assert (alpha.grad<0).all() # Descent increases opacity, even confidence=.1.
    torch.testing.assert_close(alpha.grad,-.1*(1-alpha.detach())/6)
