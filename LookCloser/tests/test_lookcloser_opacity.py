"""Behavioral tests for opacity profiles and hard first-hit rendering."""
import torch
from nerfstudio.model_components.lookcloser_opacity import first_crossing, sdf_weights, opacity_profile_loss, depth_interval_loss


def test_first_hit_occludes_second_surface_and_preserves_empty_ray():
    sdf=torch.tensor([[1.,-1.,1.,-1.],[1.,2.,3.,4.]])
    indices,hit=first_crossing(sdf)
    assert indices.tolist()==[0,0]
    assert hit.tolist()==[True,False]


def test_sign_orientation_and_exact_zero():
    indices,hit=first_crossing(torch.tensor([[-1.,1.,-1.],[1.,0.,-1.],[0.,0.,0.]]))
    assert hit.tolist()==[True,True,False]
    assert indices.tolist()==[0,0,0]


def test_profile_invariant_to_resampling_constant_density():
    def compute(count):
        edges=torch.linspace(0,1,count+1,dtype=torch.float64)
        a=1-torch.exp(torch.tensor(-2./count,dtype=torch.float64))
        weights=a*torch.exp(-2*edges[:-1])
        return opacity_profile_loss(weights,edges[:-1],edges[1:],torch.zeros(count,dtype=torch.long),1)
    assert torch.allclose(compute(1),compute(100),atol=1e-10)


def test_partial_layer_empty_gap_is_penalized_more_than_opaque_layer():
    starts=torch.tensor([.2,.8]); ends=starts+.001
    indices=torch.zeros(2,dtype=torch.long)
    partial=opacity_profile_loss(torch.tensor([.5,.5]),starts,ends,indices,1)
    opaque=opacity_profile_loss(torch.tensor([1.,0.]),starts,ends,indices,1)
    assert partial>opaque+.1


def test_segmented_profile_has_no_cross_ray_contamination():
    weights=torch.tensor([.2,.3,.9,.1])
    lo=torch.tensor([.1,.5,.2,.4]); hi=lo+.01
    both=opacity_profile_loss(weights,lo,hi,torch.tensor([0,0,1,1]),2)
    single=sum(opacity_profile_loss(weights[i:i+2],lo[i:i+2],hi[i:i+2],torch.zeros(2,dtype=torch.long),1)
               for i in [0,2])/2
    assert torch.allclose(both,single,atol=1e-7)


def test_profile_backward_is_finite_near_empty_and_saturated():
    for value in [0.,1e-7,.5,1.-1e-7]:
        w=torch.tensor([value],requires_grad=True)
        loss=opacity_profile_loss(w,torch.tensor([.2]),torch.tensor([.21]),torch.zeros(1,dtype=torch.long),1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(w.grad).all()


def test_depth_distribution_rejects_correct_mean_from_wrong_layers():
    mids=torch.tensor([1.,2.,3.]); ids=torch.zeros(3,dtype=torch.long)
    args=(mids,ids,1,torch.tensor([2.]),torch.tensor([.1]),torch.tensor([True]))
    bad=depth_interval_loss(torch.tensor([.5,0.,.5]),*args)
    good=depth_interval_loss(torch.tensor([0.,1.,0.]),*args)
    assert bad>good+1


def test_invalid_depth_has_zero_loss_and_zero_gradient():
    w=torch.tensor([.2,.3],requires_grad=True)
    loss=depth_interval_loss(w,torch.tensor([1.,3.]),torch.zeros(2,dtype=torch.long),1,
                             torch.tensor([0.]),torch.tensor([0.]),torch.tensor([False]))
    loss.backward()
    assert loss==0 and torch.equal(w.grad,torch.zeros_like(w))


def test_far_wrong_depth_retains_a_correction_gradient():
    midpoint=torch.tensor([20.],requires_grad=True)
    loss=depth_interval_loss(torch.tensor([1.]),midpoint,torch.zeros(1,dtype=torch.long),1,
                             torch.tensor([2.]),torch.tensor([.1]),torch.tensor([True]))
    loss.backward()
    assert midpoint.grad is not None and midpoint.grad>0


def test_soft_sdf_weights_concentrate_at_first_surface():
    w=sdf_weights(torch.tensor([[1.,.5,-.5,-1.,.5,-.5]]),100.)
    assert w[0,1]>.999
    assert w[0,3:].sum()<1e-5


def _toy_surface(representation='opaque_surface',permission=False):
    from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModel,LookCloserOpacityModelConfig
    from nerfstudio.data.scene_box import SceneBox
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.geometry=torch.nn.Parameter(torch.tensor(0.))
            self.alpha=torch.nn.Parameter(torch.tensor(0.))
        def query_surface(self,p,d=None):
            sdf=((p[:,2:3]-1)*(p[:,2:3]-2)+p[:,:1].abs()+self.geometry)
            if d is None:
                return sdf
            first=p[:,2:3]<1.5
            rgb=torch.cat([first.float(),torch.zeros_like(sdf),(~first).float()],-1)
            return sdf,rgb,torch.sigmoid(self.alpha).expand(len(p),1)
        def finite_gradients(self,p):
            return torch.stack([p[:,0].sign(),torch.zeros_like(p[:,1]),2*p[:,2]-3],-1)
    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel)
    torch.nn.Module.__init__(model)
    model.config=LookCloserOpacityModelConfig(representation=representation,surface_num_samples=64,
                                              surface_refine_steps=12,eikonal_loss_mult=0)
    model.field=Field()
    model.current_train_step=5000
    model.scene_box=SceneBox(aabb=torch.tensor([[-20.,-20.,-20.],[20.,20.,20.]]))
    model.register_buffer('semantic_permissions',torch.full((4,4,4),permission,dtype=torch.bool))
    return model


def _toy_rays():
    from nerfstudio.cameras.rays import RayBundle
    return RayBundle(origins=torch.tensor([[0.,0.,0.],[10.,0.,0.]]),
                     directions=torch.tensor([[0.,0.,1.],[0.,0.,1.]]),
                     pixel_area=torch.ones(2,1),nears=torch.zeros(2,1),fars=torch.full((2,1),3.))


def test_hard_renderer_rgb_and_depth_match_first_hit():
    model=_toy_surface().eval()
    result=model._surface_outputs(_toy_rays())
    assert torch.allclose(result['rgb'],torch.tensor([[1.,0.,0.],[0.,0.,0.]]))
    assert result['accumulation'].flatten().tolist()==[1.,0.]
    assert abs(float(result['depth'][0])-1)<1e-4
    assert result['depth'][1]==0


def test_hard_forward_has_geometry_surrogate_gradient():
    model=_toy_surface().train()
    result=model._surface_outputs(_toy_rays())
    assert torch.allclose(result['rgb'],torch.tensor([[1.,0.,0.],[0.,0.,0.]]))
    result['depth'][0].backward()
    assert model.field.geometry.grad is not None
    assert torch.isfinite(model.field.geometry.grad)
    assert model.field.geometry.grad.abs()>0


def test_guided_samples_partition_the_ray_and_leave_unknown_ray_unchanged():
    model=_toy_surface().train()
    model.config.depth_prior_strength=16.
    rays=_toy_rays()
    rays.metadata=dict(depth_prior=torch.tensor([[1.],[2.]]),depth_uncertainty=torch.tensor([[.001],[.001]]),
                       depth_valid=torch.tensor([[True],[False]]))
    t=torch.linspace(0,3,65).expand(2,-1)
    guided=model._guided_edges(rays,t)
    assert torch.all(guided[:,1:]>=guided[:,:-1])
    assert torch.allclose((guided[:,1:]-guided[:,:-1]).sum(-1),torch.tensor([3.,3.]))
    assert ((guided[0]-1).abs()<.0021).sum()>30
    assert torch.equal(torch.unique(guided[1]),torch.unique(t[1]))
    model.config.depth_prior_strength=0
    assert torch.equal(model._guided_edges(rays,t),t)


def test_semantic_alpha_is_fixed_outside_permissions_and_gets_rgb_gradient(monkeypatch):
    import nerfacc
    def pure_weights(alpha,packed_info):
        trans=torch.cumprod(torch.cat([torch.ones_like(alpha[:1]),1-alpha]),0)[:-1]
        return alpha*trans,trans
    monkeypatch.setattr(nerfacc,'render_weight_from_alpha',pure_weights)
    def pure_pack(indices,n):
        counts=torch.bincount(indices,minlength=n)
        return torch.stack([counts.cumsum(0)-counts,counts],-1)
    monkeypatch.setattr(nerfacc,'pack_info',pure_pack)
    forbidden=_toy_surface('semantic_surface',False).eval()._surface_outputs(_toy_rays())
    assert torch.equal(forbidden['rgb'][0],torch.tensor([1.,0.,0.]))
    allowed=_toy_surface('semantic_surface',True).train()
    outputs=allowed._surface_outputs(_toy_rays())
    assert torch.allclose(outputs['rgb'][0],torch.tensor([.5,0.,.25]),atol=1e-5)
    (outputs['rgb'][0]-torch.tensor([1.,0.,0.])).square().sum().backward()
    assert allowed.field.alpha.grad is not None and allowed.field.alpha.grad.abs()>0
    with torch.no_grad():
        allowed.field.alpha.copy_(torch.logit(torch.tensor(.2)))
    outputs=allowed.eval()._surface_outputs(_toy_rays())
    assert abs(float(outputs['depth'][0])-1)<1e-4
    assert float(outputs['expected_depth'][0])>1.4
