"""CPU integration references for independent LC semantic fitting/export."""
import copy
import io
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_LC_semantic_physical_path import capture_lc_path,packed_density_weights
from marshawn_frozen_semantic_head import FrozenActorGrid,rendered_classes,conditional_class_objective
from marshawn_frozen_semantic_selectors import selector_memberships
from render_marshawn_LC_semantic_physical import capture_path
from test_marshawn_lc_semantic_capture import Model,Rays


@pytest.mark.parametrize('cap',[0,1024])
def test_declared_native_and_uncapped_observers_preserve_real_weights(cap):
    model=Model();model.config.max_steps_per_ray=cap;before=model.field.forward
    row=capture_lc_path(model,Rays(),expected_cap=cap,density_key='density',rgb_key='rgb')
    assert max(row['proof'].values())<1e-6
    assert model.field.forward==before and 'forward'not in model.field.__dict__
    assert not row['cap_hit'].any()
    assert row['starts'].tolist()==[2.,3.,8.]
    assert row['native_alpha'][1]==0


def test_declared_cap_must_match_actual_source_before_forward():
    with pytest.raises(ValueError):
        capture_lc_path(Model(),Rays(),expected_cap=0,density_key='density',rgb_key='rgb')


def test_TCNN_half_RGB_and_float_visibility_preserve_native_product_precision():
    from test_marshawn_lc_semantic_capture import Field
    class HalfRGB(Field):
        def forward(self,samples):
            result=super().forward(samples);result['rgb']=result['rgb'].half();return result
    model=Model();model.field=HalfRGB().eval()
    def actual_renderer(rgb,weights,ray_indices,num_rays):
        product=weights*rgb
        return product.new_zeros((num_rays,3)).index_add(0,ray_indices.long(),product)
    model.renderer_rgb=actual_renderer
    row=capture_lc_path(model,Rays(),density_key='density',rgb_key='rgb')
    assert row['rgb'].dtype==torch.float16 and row['weight'].dtype==torch.float32
    assert row['native_rgb'].dtype==torch.float32 and row['proof']['RGB']<1e-6


def test_uncapped_temporary_config_restored_after_native_exception(monkeypatch):
    import marshawn_LC_semantic_physical_path as path
    model=Model()
    def fail(model,rays,*,expected_cap):
        assert expected_cap==0 and model.config.max_steps_per_ray==0
        raise RuntimeError('injected native failure')
    monkeypatch.setattr(path,'capture_lc_path',fail)
    with pytest.raises(RuntimeError,match='injected'):capture_path(model,Rays(),'uncapped0')
    assert model.config.max_steps_per_ray==1024


def test_hidden_removal_prior_changes_physical_visibility_not_original_partition():
    starts=torch.tensor([0.,.1,.2]);ends=starts+.1;sigma=torch.tensor([1.,1000.,500.]);ids=torch.zeros(3,dtype=torch.long)
    native=packed_density_weights(starts,ends,sigma,ids,1)
    assert native[2]<1e-40
    high=selector_memberships(torch.tensor([1.,0.,.99]))['hard_theta05']
    low=selector_memberships(torch.tensor([1.,0.,.01]))['hard_theta05']
    selected_high=packed_density_weights(starts,ends,sigma*high,ids,1)
    selected_low=packed_density_weights(starts,ends,sigma*low,ids,1)
    assert selected_high[2]>.9 and selected_low[2]==0
    assert (native*high)[2]<1e-40


def test_floor_identity_and_XYZ_node_axes_are_explicit():
    head=FrozenActorGrid([[0,0,0],[1,1,1]],np.eye(4),resolution=2,initial_probability=.01)
    with torch.no_grad():
        head.logits.copy_(torch.arange(8,dtype=torch.float32).reshape(1,1,2,2,2))
    xyz=torch.tensor([[1.,0.,0.],[0.,0.,1.],[.5,.5,.5],[-.01,.5,.5]])
    actual=head(xyz)
    assert torch.allclose(actual[:3],torch.tensor([4.,1.,3.5]).sigmoid())
    assert actual[3]==0 and torch.equal(head.world_to_box,torch.eye(4))


def test_CPU_reference_resume_32_plus32_matches64_head_Adam_sampler_exactly():
    torch.set_num_threads(2)
    xyz=torch.tensor([[.2,.3,.4],[.7,.6,.5],[.4,.4,.4]])
    weights=torch.tensor([.8,.9,.2]);ids=torch.tensor([0,1,0]);targets=torch.tensor([1.,0.]);confidence=torch.ones(2)
    def setup():
        head=FrozenActorGrid([[0,0,0],[1,1,1]],np.eye(4),resolution=4,initial_probability=.99)
        return head,torch.optim.Adam([head.logits],lr=.03),np.random.default_rng(42)
    def advance(head,optimizer,sampler,count):
        history=[]
        for _ in range(count):
            packet=int(sampler.integers(60));history.append(packet)
            q=rendered_classes(head(xyz),weights,ids,2)
            loss=conditional_class_objective(q,targets,confidence)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        return history
    head,optimizer,sampler=setup();history=advance(head,optimizer,sampler,32)
    payload=dict(head=head.state_dict(),Adam=optimizer.state_dict(),sampler=sampler.bit_generator.state,torch_RNG=torch.get_rng_state())
    buffer=io.BytesIO();torch.save(payload,buffer);buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    resumed,Adam,new_sampler=setup();resumed.load_state_dict(saved['head'],strict=True);Adam.load_state_dict(saved['Adam']);new_sampler.bit_generator.state=saved['sampler'];torch.set_rng_state(saved['torch_RNG'])
    assert torch.equal(resumed.logits,head.logits)
    for key,value in Adam.state_dict()['state'][0].items():assert torch.equal(value,optimizer.state_dict()['state'][0][key])
    resumed_history=history+advance(resumed,Adam,new_sampler,32)
    independent,reference_Adam,reference_sampler=setup();reference_history=advance(independent,reference_Adam,reference_sampler,64)
    assert resumed_history==reference_history and torch.equal(resumed.logits,independent.logits)
    for key,value in Adam.state_dict()['state'][0].items():assert torch.equal(value,reference_Adam.state_dict()['state'][0][key])
    assert new_sampler.bit_generator.state==reference_sampler.bit_generator.state
    assert not torch.cuda.is_initialized()
