"""Cord RGB oversampling preserves ordinary rays and uses source-image RGB."""

from types import SimpleNamespace

import torch

from nerfstudio.data.datamanagers.luster_opacity_datamanager import LusterOpacityDataManager


def test_cord_rgb_replacement_is_deterministic_and_uses_real_pixels():
    yy,xx=torch.meshgrid(torch.arange(10),torch.arange(10),indexing='ij')
    images=[torch.stack((xx/10,yy/10,torch.full_like(xx,float(camera))),dim=-1).float()
            for camera in (1,2)]
    source={'image_idx':torch.tensor([1,2]),'image':images}
    manager=SimpleNamespace(config=SimpleNamespace(cord_rgb_rays_per_batch=4,
                                                     cord_rgb_jitter_px=1.,cord_rgb_seed=42),
                            _cord_rgb_lines=torch.tensor([[[2.,2.],[7.,2.]],[[2.,7.],[7.,7.]]]),
                            _cord_rgb_dataset_indices=torch.tensor([1,2]),
                            _cord_rgb_dimensions=torch.tensor([[10,10],[10,10]]))

    def fresh_batch():
        return {'indices':torch.zeros(16,3,dtype=torch.long),
                'image':torch.zeros(16,3)}

    first,second=fresh_batch(),fresh_batch()
    LusterOpacityDataManager._replace_cord_rgb_rays(manager,123,first,source)
    LusterOpacityDataManager._replace_cord_rgb_rays(manager,123,second,source)
    assert torch.equal(first['indices'],second['indices'])
    assert torch.equal(first['image'],second['image'])
    assert torch.equal(first['indices'][:-4],torch.zeros(12,3,dtype=torch.long))
    assert torch.equal(first['image'][:-4],torch.zeros(12,3))
    for camera,y,x in first['indices'][-4:].tolist():
        assert camera in (1,2)
        assert 0<=x<10 and 0<=y<10
    for (camera,y,x),rgb in zip(first['indices'][-4:].tolist(),first['image'][-4:]):
        assert torch.equal(rgb,images[camera-1][y,x])
