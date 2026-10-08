"""A sampled cord slab should reject nearby visible layers."""

import torch

from nerfstudio.model_components.lookcloser_opacity import cord_termination_mass_loss


def test_termination_mass_prefers_cord_slab_to_nearby_front_density():
    eps=torch.tensor([.003])
    free=torch.zeros(1,6)
    band=torch.zeros(1,60)
    band[:,24:36]=2000.
    good=cord_termination_mass_loss(band,eps,free)
    nearby=band.clone()
    nearby[:,23]=2000.  # Just outside the slab; old sparse free probes miss it.
    assert float(good)<1e-3
    assert float(cord_termination_mass_loss(nearby,eps,free))>float(good)+.5
    assert float(cord_termination_mass_loss(band*.1,eps,free))>float(good)+.08
    assert float(cord_termination_mass_loss(band,eps,free+3000))>float(good)+.5


def test_termination_mass_gradients_move_density_from_front_into_slab():
    sigma=torch.zeros(1,60)
    sigma[:,23:36]=300.
    sigma.requires_grad_()
    free=torch.zeros(1,6,requires_grad=True)
    cord_termination_mass_loss(sigma,torch.tensor([.003]),free).backward()
    assert sigma.grad[0,23]>0
    assert (sigma.grad[0,24:36]<0).all()
    assert torch.isfinite(sigma.grad).all()


def test_termination_mass_empty_batch_has_zero_gradient():
    sigma=torch.empty(0,60,requires_grad=True)
    free=torch.empty(0,6,requires_grad=True)
    value=cord_termination_mass_loss(sigma,torch.empty(0),free)
    assert value==0
    value.backward()
    assert sigma.grad.numel()==free.grad.numel()==0
