from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_leader_person_sampling import LeaderPersonSampler


def test_leader_sources_camera_identity_and_native_raster_scaling(tmp_path):
    images=tmp_path/'images';images.mkdir();maps=tmp_path/'lookcloser_frequencies';maps.mkdir();sem=tmp_path/'semantics';sem.mkdir()
    paths=[]
    for camera in range(64):
        name=f'C{camera:03d}';path=images/f'frame_train_{name}.png';path.touch();paths.append(path)
        values=np.zeros((64,128,3),np.uint16);values[...,2]=65535
        values[16:55,40:88,2]=0;values[16:32,40:88,0]=65535
        np.savez_compressed(sem/(name+'.npz'),probability=values)
        frequency=torch.full((16,32),16.);frequency[:,16:]=8192;torch.save(frequency,maps/(path.stem+'.pt'))
    sampler=SimpleNamespace(num_rays_per_batch=24000,sample_count=0)
    dataset=SimpleNamespace(image_filenames=paths,cameras=SimpleNamespace(height=[128]*64,width=[256]*64))
    body=LeaderPersonSampler(sampler,dataset,sem);body.step=10000
    # Encode physical camera and pixel coordinates in the RGB channels, with a
    # shuffled cache order. Sampling must not confuse cache IDs with camera IDs.
    order=torch.randperm(64);y,x=torch.meshgrid(torch.arange(128),torch.arange(256),indexing='ij')
    batch=torch.empty((64,128,256,3))
    batch[...,0]=order[:,None,None];batch[...,1]=y;batch[...,2]=x
    result=body.sample(dict(image=batch,image_idx=order));indices=result['indices']
    torch.testing.assert_close(result['image'],indices.float(),rtol=0,atol=0)
    fractions=body.source_hits/body.source_hits.sum()
    np.testing.assert_allclose(fractions,[1/3,2/15,1/5,1/3],atol=.012,rtol=0)
    assert set(indices[:,0].tolist())==set(range(64))
    assert indices[:,1].min()>=0 and indices[:,1].max()<128
    assert indices[:,2].min()>=0 and indices[:,2].max()<256
    yy,xx,valid=body.frequency_pixels(torch.arange(64))
    assert valid.all() and yy.max()<128 and xx.max()<256
    repeat=body.sample(dict(image=batch,image_idx=order))
    torch.testing.assert_close(indices,repeat['indices'])  # Per-step independent RNG survives a process restart.
    focused=LeaderPersonSampler(SimpleNamespace(num_rays_per_batch=24000,sample_count=0),dataset,sem,hair_core_mix=.5)
    focused.step=10000;result=focused.sample(dict(image=batch,image_idx=order))
    torch.testing.assert_close(result['image'],result['indices'].float(),rtol=0,atol=0)
    assert focused.audit['hair']/focused.audit['rays']>body.last_batch[0]/24000+.10
    np.testing.assert_allclose(focused.source_hits/focused.source_hits.sum(),[1/3,2/15,1/5,1/3],atol=.012,rtol=0)
    delayed=LeaderPersonSampler(SimpleNamespace(num_rays_per_batch=24000,sample_count=0),dataset,sem,hair_core_mix=.5,hair_mix_start=15189)
    delayed.step=10000
    torch.testing.assert_close(delayed.sample(dict(image=batch,image_idx=order))['indices'],indices)
    delayed.step=focused.step=20000
    torch.testing.assert_close(delayed.sample(dict(image=batch,image_idx=order))['indices'],focused.sample(dict(image=batch,image_idx=order))['indices'])
