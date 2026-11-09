from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_boundary_budget_sampling import BoundaryBudgetSampler
from dec5_leader_person_sampling import LeaderPersonSampler


@pytest.mark.parametrize('frequency_weighted',[False,True])
@pytest.mark.parametrize('scale',[1,2])
def test_boundary_budget_preserves_body_fas_and_rgb_alignment(tmp_path,frequency_weighted,scale):
    images=tmp_path/'images';maps=tmp_path/'lookcloser_frequencies';sem=tmp_path/'semantics'
    for p in [images,maps,sem]:p.mkdir()
    filenames=[]
    for i in range(64):
        name=f'C{i}';path=images/f'frame_train_{name}.png';filenames.append(path)
        probability=np.zeros((32,48,3),np.uint16);probability[...,2]=65535
        probability[8:24,12:36,2]=0;probability[8:24,12:36,0]=65535
        np.savez(sem/(name+'.npz'),probability=probability)
        frequency=torch.full((4*scale,6*scale),16.);frequency[:,3*scale:]=8192;torch.save(frequency,maps/(path.stem+'.pt'))
    ds=SimpleNamespace(image_filenames=filenames,cameras=SimpleNamespace(height=[32*scale]*64,width=[48*scale]*64))
    def base():return SimpleNamespace(num_rays_per_batch=12000,sample_count=0)
    old=LeaderPersonSampler(base(),ds,sem);new=BoundaryBudgetSampler(base(),ds,sem,boundary_fraction=.1,boundary_frequency=frequency_weighted)
    old.step=new.step=12000
    rgb=torch.arange(64*32*48*3*scale**2).reshape(64,32*scale,48*scale,3)
    # Cache order differs from physical camera ID order.
    order=torch.arange(63,-1,-1);inputs=dict(image=rgb[order],image_idx=order)
    b=old.sample(inputs);c=new.sample(inputs)
    keep=old.last_source!=2
    assert torch.equal(b['indices'][keep],c['indices'][keep])
    assert torch.equal(b['indices'][:,0],c['indices'][:,0])
    idx=c['indices'];assert torch.equal(c['image'],rgb[idx[:,0],idx[:,1],idx[:,2]])
    changed=(b['indices']!=idx).any(-1).numpy();assert not changed[keep].any()
    assert .08<changed.mean()<.12
    assert sum(new.audit[k] for k in ['hair','person_nonhair','background','uncertain'])==12000
    assert new.boundary_draws>1000
