from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_dense_hair_training import surface_guidance_terms


def test_free_space_integral_respects_partial_native_interval_and_surface_gradient():
    sigma=torch.tensor([2.,3.],requires_grad=True);band=torch.full((1,16),2.,requires_grad=True)
    empty,surface=surface_guidance_terms(sigma,torch.tensor([0.,1.]),torch.tensor([1.,2.]),torch.tensor([0,0]),torch.tensor([1.5]),torch.tensor([.1]),band)
    assert torch.allclose(empty,torch.tensor([3.2]))
    assert torch.allclose(surface,-torch.log1p(-torch.exp(torch.tensor([-.4]))))
    (empty+surface).sum().backward()
    assert bool((sigma.grad>0).all()) and bool((band.grad<0).all())


def test_missing_surface_keeps_a_finite_nonzero_growth_gradient():
    sigma=torch.tensor([],requires_grad=True);band=torch.full((1,16),1e-4,requires_grad=True)
    empty,surface=surface_guidance_terms(sigma,torch.tensor([]),torch.tensor([]),torch.tensor([],dtype=torch.long),torch.tensor([1.]),torch.tensor([1e-4]),band)
    surface.sum().backward()
    assert bool(torch.isfinite(surface).all()) and bool(torch.isfinite(band.grad).all())
    assert bool((band.grad<0).all()) and float(empty[0])==0


def test_opaque_median_constraint_has_known_physical_zero_and_ignores_tail():
    from dec5_dense_hair_training import median_optical_thickness_term
    half=torch.tensor([.2],dtype=torch.float64)
    band=torch.full((1,16),float(torch.log(torch.tensor(2.,dtype=torch.float64))/.2),dtype=torch.float64)
    band[:,8:]*=100  # Density behind the anchor cannot change its first-hit CDF.
    term=median_optical_thickness_term(torch.zeros(1,dtype=torch.float64),half,band)
    assert float(term[0])<1e-25


def test_opaque_median_constraint_moves_mass_toward_the_anchor():
    from dec5_dense_hair_training import median_optical_thickness_term
    band=torch.tensor([[.1]*16,[10.]*16],requires_grad=True)
    before=torch.tensor([0.,.2],requires_grad=True)
    term=median_optical_thickness_term(before,torch.tensor([.2,.2]),band)
    term.sum().backward()
    assert bool(torch.isfinite(term).all())
    assert bool((band.grad[0,:8]<0).all()) and bool((band.grad[1,:8]>0).all())
    assert bool((band.grad[:,8:]==0).all())
    assert before.grad[0]<0 and before.grad[1]>0
