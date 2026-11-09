"""A world-space density plane stays fixed across camera/ray partitions."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from dec5_sigma_surface import first_sigma_surface


def test_same_density_surface_from_different_origins_and_nonuniform_intervals():
 origins=torch.tensor([[0.,0.,0.],[.3,0.,1.],[0.,0.,0.]])
 directions=torch.tensor([[0.,0.,1.],[0.,0.,1.],[1.,0.,0.]])
 rays=SimpleNamespace(origins=origins,directions=directions,nears=torch.zeros(3,1),fars=torch.full((3,1),5.))
 density_fn=lambda p:(p[:,2]-2).clamp_min(0)[:,None]
 indices=torch.tensor([0,1,0,1,2]);starts=torch.tensor([.2,.1,.6,.5,.2]);ends=starts+.1;mid=.5*(starts+ends)*5;p=origins[indices]+directions[indices]*mid[:,None];density=density_fn(p)
 result=first_sigma_surface(density_fn,rays,starts,ends,density,indices,.5,refine_steps=16)
 world=origins+directions*result[:,None]
 torch.testing.assert_close(world[:2,2],torch.full((2,),2.5),rtol=0,atol=4e-5)
 assert result[2]==0


def test_empty_packed_rays_do_not_create_a_surface():
 rays=SimpleNamespace(origins=torch.zeros(2,3),directions=torch.ones(2,3),nears=torch.zeros(2,1),fars=torch.ones(2,1))
 empty=torch.empty(0);value=first_sigma_surface(lambda p:torch.ones(len(p),1),rays,empty,empty,empty,empty.long(),1.)
 assert torch.equal(value,torch.zeros(2))
