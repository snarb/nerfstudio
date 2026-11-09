"""Actual native sampler regression for CacheDataloader's shuffled camera order."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def make_sampler_and_batch(order):
    from marshawn_actor_training_pipeline import ActorProposalPixelSampler
    from nerfstudio.data.pixel_samplers import PixelSamplerConfig
    n,h,w=3,2,3
    proposal=torch.zeros(n,h,w,dtype=torch.bool)
    proposal[0,0,0]=True
    proposal[1,0,:2]=True
    proposal[2,1,:]=True
    identity=torch.arange(n*h*w).reshape(n,h,w)
    rgb=torch.stack([identity.float(),identity.float()+20,identity.float()+40],-1)
    batch=dict(image_idx=torch.arange(n),image=rgb,mask=proposal[...,None],
               depth_image=(identity.float()+100)[...,None])
    order=torch.tensor(order)
    batch={k:v[order].clone() for k,v in batch.items()}
    sampler=ActorProposalPixelSampler(PixelSamplerConfig(num_rays_per_batch=12000))
    sampler.actor_expected_proposal=proposal
    return sampler,batch,proposal


def test_actual_shuffled_batch_preserves_global_pixels_rgb_auxiliary_and_uniformity():
    reference,canonical,_=make_sampler_and_batch([0,1,2])
    sampler,shuffled,proposal=make_sampler_and_batch([2,0,1])
    before={k:v.clone() for k,v in shuffled.items()}
    torch.manual_seed(721);expected=reference.sample(canonical)
    torch.manual_seed(721);actual=sampler.sample(shuffled)
    for key in ['indices','image','mask','depth_image']:
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    for key,value in before.items():
        assert torch.equal(shuffled[key],value), 'Cached source batch was mutated'
    camera,y,x=actual['indices'].unbind(-1)
    encoded=camera*6+y*3+x
    assert torch.equal(actual['image'][:,0],encoded.float())
    assert torch.equal(actual['depth_image'][:,0],encoded.float()+100)
    assert proposal[camera,y,x].all()
    # Uniform eligible pixels => camera masses1:2:3, not uniform cameras.
    mass=torch.bincount(camera,minlength=3).float()/len(camera)
    torch.testing.assert_close(mass,torch.tensor([1/6,2/6,3/6]),rtol=0,atol=.015)
    cached=sampler._actor_canonical_batch
    sampler.sample(shuffled)
    assert sampler._actor_canonical_batch is cached, 'Do not recopy all63 RGB images each step'


@pytest.mark.parametrize('order',[[0,0,2],[0,1],[0,1,3]])
def test_sampler_rejects_missing_duplicate_or_out_of_split_camera_ids(order):
    sampler,batch,_=make_sampler_and_batch([0,1,2])
    if len(order)!=3:
        batch={k:v[:len(order)] for k,v in batch.items()}
    batch['image_idx']=torch.tensor(order)
    with pytest.raises(ValueError,match='ALL unique split cameras'):
        sampler.sample(batch)
