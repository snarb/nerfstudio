import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_tuning_sampling import install_sampling, body80_quotas, weighted_charbonnier
from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSampler, LookCloserPixelSamplerConfig


def setup(tmp_path, *, objective='actor',sampling='default'):
    for name in ('semantics','lookcloser_frequencies','images'):
        (tmp_path/name).mkdir(exist_ok=True)
    n,h,w=2,32,40
    filenames=[tmp_path/'images'/f'train_{c}.png' for c in ('AA','AB')]
    for i,c in enumerate(('AA','AB')):
        core=np.zeros((h,w),bool);core[8:24,8:32]=True
        band=np.zeros_like(core);band[4:28,4:36]=True;band&=~core
        valid=np.ones_like(core);valid[:,0:2]=False
        weight=np.where(core,1.,np.where(band,.25,0.)).astype(np.float32)
        weight[~valid]=0
        black=core.copy();black[:,20:]=False
        grey=core&~black
        np.savez(tmp_path/'semantics'/f'{c}.npz',core=core,band=band,black=black,grey=grey,valid=valid,W32=weight*.8,W64=weight)
        path=tmp_path/'lookcloser_frequencies'/f'train_{c}.pt'
        freq=torch.ones((h//8,w//8))*16
        freq[:,2:]=8192
        torch.save(freq,path)
        path.with_suffix('.json').write_text(json.dumps(dict(value_type='scalar_resolution',patch_size=8,stride=8,min_res=16,max_res=8192,n_levels=16,image_shape=[h,w])))
    dataset=SimpleNamespace(image_filenames=filenames)
    sampler=LookCloserPixelSampler(LookCloserPixelSamplerConfig(num_rays_per_batch=4096),dataset=dataset)
    def original_loss(outputs,batch,metrics_dict=None):
        return dict(rgb_loss=torch.sqrt((outputs['rgb']-batch['image']).square()+1e-4).mean(),distortion_loss=torch.tensor(.123))
    model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=original_loss)
    pipeline=SimpleNamespace(datamanager=SimpleNamespace(train_pixel_sampler=sampler,train_dataset=dataset),model=model)
    grid=torch.arange(n*h*w).reshape(n,h,w,1)
    images=grid.expand(n,h,w,3).float()/(n*h*w)
    valid=torch.stack([torch.from_numpy(np.load(tmp_path/'semantics'/f'{c}.npz')['valid']) for c in ('AA','AB')])[...,None]
    batch=dict(image=images,image_idx=torch.arange(n),depth=grid.float(),mask=valid)
    adapter=install_sampling(pipeline,tmp_path,objective,sampling,64)
    return pipeline,adapter,batch


def test_default_actual_sampler_and_rng_unchanged(tmp_path):
    pipeline,adapter,batch=setup(tmp_path,objective='full')
    sampler=pipeline.datamanager.train_pixel_sampler
    sampler._initialize_buckets(pipeline.datamanager.train_dataset)
    torch.manual_seed(123)
    reference=adapter._original_sample(batch)
    reference_rng=torch.get_rng_state()
    sampler.sample_count=0
    torch.manual_seed(123)
    actual=sampler.sample(batch)
    assert torch.equal(torch.get_rng_state(),reference_rng)
    assert actual.keys()==reference.keys()
    for k in reference:assert torch.equal(actual[k],reference[k])


def test_baseline_normalizer_matches_exhaustive_pixel_proposal(tmp_path):
    pipeline,adapter,batch=setup(tmp_path)
    sampler=pipeline.datamanager.train_pixel_sampler
    normalizer=adapter._baseline_normalizer(batch)
    # Two equal-size endpoint classes with exact 1:3 quotas; class means are
    # evaluated independently over all patch pixels, including partial weights.
    w=np.stack([row['weight'] for row in adapter.rows])
    low=float(w[:,:,:16].mean());high=float(w[:,:,16:].mean())
    assert normalizer==pytest.approx(.25*low+.75*high,abs=1e-8)
    means=[]
    torch.manual_seed(4)
    for _ in range(20):means.append(float(sampler.sample(batch)['actor_confidence'].mean()))
    assert np.mean(means)==pytest.approx(normalizer,abs=.005)


def test_body80_quotas_aux_fields_and_analytic_normalizer(tmp_path):
    pipeline,adapter,batch=setup(tmp_path,sampling='body80')
    assert body80_quotas(4096,0).tolist()==[2731,0,546,819]
    assert sum(body80_quotas(4096,8192))==4096
    adapter.set_step(8192)
    np.random.seed(9)
    means=[];cams=[]
    for _ in range(30):
        result=pipeline.datamanager.train_pixel_sampler.sample(batch)
        c,y,x=result['indices'].T
        assert torch.equal(result['depth'],batch['depth'][c,y,x])
        assert torch.equal(result['mask'],batch['mask'][c,y,x])
        assert result['mask'].all()
        means.append(float(result['actor_confidence'].mean()));cams.extend(c.tolist())
    assert np.mean(means)==pytest.approx(adapter.last_normalizer,abs=.005)
    assert np.mean(cams)==pytest.approx(.5,abs=.01)
    assert adapter.audit['rays']==30*4096
    assert adapter.source_hits.tolist()==(30*body80_quotas(4096,8192)).tolist()


def test_actor_loss_weights_once_and_does_not_change_auxiliary(tmp_path):
    pipeline,adapter,batch=setup(tmp_path)
    samples=pipeline.datamanager.train_pixel_sampler.sample(batch)
    pred=samples['image']*.7
    result=pipeline.model.get_loss_dict(dict(rgb=pred),samples)
    expected=weighted_charbonnier(pred,samples['image'],samples['actor_confidence'],samples['actor_proposal_mean_weight'])
    assert result['rgb_loss']==expected
    assert float(result['distortion_loss'])==pytest.approx(.123)
    # Constant errors must have expectation sqrt(.01+.0001), not a batch-normalized value.
    p=torch.zeros((2,3));g=torch.ones_like(p)*.1;w=torch.tensor([1.,0.])
    assert float(weighted_charbonnier(p,g,w,.25))==pytest.approx(2*np.sqrt(.0101))


def test_state_restore_and_reject_changed_cache(tmp_path):
    pipeline,adapter,batch=setup(tmp_path,sampling='body80')
    adapter.set_step(8192);pipeline.datamanager.train_pixel_sampler.sample(batch)
    state=adapter.state_dict()
    adapter.set_step(0);adapter.load_state_dict(state)
    assert adapter.state_dict()==state
    with pytest.raises(ValueError,match='complete unique'):
        adapter.sample(dict(batch,image_idx=torch.tensor([1,1])))


def test_default_preserves_shuffled_cache_rng_and_exact_confidence(tmp_path):
    pipeline,adapter,batch=setup(tmp_path)
    sampler=pipeline.datamanager.train_pixel_sampler
    sampler._initialize_buckets(pipeline.datamanager.train_dataset)
    # Give camera1 a different confidence to expose incorrect expectation/indexing.
    adapter.rows[1]['weight'] = adapter.rows[1]['weight']*.2
    batch={key:value.flip(0) for key,value in batch.items()}
    torch.manual_seed(321)
    ref=adapter._original_sample(batch);rng=torch.get_rng_state()
    sampler.sample_count=0;torch.manual_seed(321)
    result=sampler.sample(batch)
    assert torch.equal(torch.get_rng_state(),rng)
    for key in ref: assert torch.equal(result[key],ref[key])
    c,y,x=result['indices'].T.numpy()
    expected=np.asarray([adapter.rows[ci]['weight'][yi,xi] for ci,yi,xi in zip(c,y,x)])
    np.testing.assert_array_equal(result['actor_confidence'].numpy(),expected)
    assert float(result['actor_confidence'].mean())==pytest.approx(result['actor_proposal_mean_weight'],abs=.015)
