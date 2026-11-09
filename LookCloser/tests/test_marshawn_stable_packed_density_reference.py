"""Adversarial old-versus-exclusive reference arithmetic, CPU only."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_stable_packed_density_reference import stable_packed_density_weights
from marshawn_LC_semantic_physical_path import packed_density_weights as legacy


def serial(starts,ends,sigma,ids,rays):
    before=torch.zeros(rays,dtype=torch.float64);result=[]
    for s,e,d,i in zip(starts,ends,sigma,ids):
        tau=d.double()*(e.double()-s.double())
        result.append(torch.exp(-before[i])*-torch.expm1(-tau))
        before[i]+=tau
    return torch.stack(result)if result else torch.empty(0,dtype=torch.float64)


@pytest.mark.parametrize('densities,ids,rays',[
    ([2.**54,.25,.5],[0,1,1],2),  # global cross-ray cancellation
    ([.25,2.**54],[0,0],1),      # cumsum(tau)-CURRENT huge tau cancels prior
    ([.25,2.**54,.25,.5],[0,0,3,3],5),
])
def test_old_reference_fails_but_padded_exclusive_matches_serial(densities,ids,rays):
    sigma=torch.tensor(densities,dtype=torch.float32);starts=torch.zeros_like(sigma);ends=torch.ones_like(sigma);ids=torch.tensor(ids)
    expected=serial(starts,ends,sigma,ids,rays)
    old=legacy(starts,ends,sigma,ids,rays)
    new=stable_packed_density_weights(starts,ends,sigma,ids,rays)
    assert (old-expected).abs().max()>.01
    assert torch.allclose(new,expected,rtol=1e-14,atol=1e-15)


def test_normal_intervals_zero_sigma_irregular_empty_rays_match_serial():
    starts=torch.tensor([1.,1.5,2.,3.,3.1,4.,5.]);ends=starts+.1
    sigma=torch.tensor([0.,1e-7,1.,2.,3.,1e8,1e-8]);ids=torch.tensor([0,0,0,2,2,5,5])
    actual=stable_packed_density_weights(starts,ends,sigma,ids,8)
    assert torch.allclose(actual,serial(starts,ends,sigma,ids,8),atol=1e-15,rtol=1e-14)
    assert actual[0]==0 and actual[-1]==0


def test_empty_intervals_and_invalid_packing():
    empty=torch.empty(0)
    assert stable_packed_density_weights(empty,empty,empty,empty.long(),3).shape==(0,)
    with pytest.raises(ValueError):
        stable_packed_density_weights(torch.zeros(2),torch.ones(2),torch.ones(2),torch.tensor([1,0]),2)


def test_prefix_is_independent_of_unrelated_other_ray_optical_depth():
    starts=torch.zeros(4);ends=torch.ones(4);ids=torch.tensor([0,0,1,1])
    one=torch.tensor([1.,2.,.125,.25]);two=torch.tensor([1e30,2.,.125,.25])
    a=stable_packed_density_weights(starts,ends,one,ids,2)
    b=stable_packed_density_weights(starts,ends,two,ids,2)
    assert torch.equal(a[2:],b[2:])
    assert not torch.cuda.is_initialized()
