"""Camera filtering must preserve RGB/ray identity and exclude every forbidden view."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.luster_opacity_datamanager import (
    LusterOpacityDataManager, filter_camera_image_batch,
)
from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSamplerConfig


@pytest.mark.parametrize('strength',[0.,.5,1.])
@pytest.mark.parametrize('dense',[False,True])
def test_filtered_fas_preserves_noncontiguous_ids_and_target_colors(strength,dense):
    ids=[10,2,7]
    shapes=[(3,4)]*3 if dense else [(3,3),(4,2),(2,4)]
    images=[torch.full((*shape,3),float(cid)) for cid,shape in zip(ids,shapes)]
    source=dict(image=torch.stack(images) if dense else images,image_idx=torch.tensor(ids))
    batch=filter_camera_image_batch(source,(2,7))
    assert batch['image_idx'].tolist()==[2,7]
    assert source['image_idx'].tolist()==ids
    assert isinstance(batch['image'],list)
    sampler=LookCloserPixelSamplerConfig(num_rays_per_batch=256,fas_strength=strength).setup()
    sampler.is_initialized=True;sampler.patch_size=sampler.patch_stride=1
    sampler.image_shapes={cid:shape for cid,shape in zip(ids,shapes)}
    sampler.buckets={i:torch.empty((0,3),dtype=torch.int32) for i in range(16)}
    sampler.buckets[0]=torch.tensor([[10,1,1],[2,1,1],[7,1,1]])
    sampler.probs=np.array([1.]+[0.]*15)
    result=sampler.sample(batch)
    actual=result['indices'][:,0]
    assert set(actual.tolist())=={2,7}
    assert torch.equal(result['image'][:,0],actual.float())
    assert sampler.sample_count==1


def test_empty_allowed_batch_fails_instead_of_sampling_excluded_camera():
    with pytest.raises(ValueError,match='no allowed'):
        filter_camera_image_batch(dict(image=[torch.zeros(2,2,3)],image_idx=torch.tensor([10])),(2,7))


@pytest.mark.parametrize('ids,expected',[
    ((11,97),(0,2)),((95,),None),((11,11),None),((),None),
])
def test_physical_ids_are_resolved_on_training_dataset_only(monkeypatch,ids,expected):
    monkeypatch.setattr(VanillaDataManager,'setup_train',lambda self:None)
    dm=LusterOpacityDataManager.__new__(LusterOpacityDataManager)
    dm.config=SimpleNamespace(train_camera_ids=ids,cpu_fas_prefetch=False)
    dm.train_image_dataloader=SimpleNamespace(cache_all_images=True)
    dm.train_dataset=SimpleNamespace(image_filenames=[Path(f'cam_{i:03d}_000470.png') for i in [11,23,97]])
    if expected is None:
        with pytest.raises(ValueError):dm.setup_train()
    else:
        dm.setup_train()
        assert dm._allowed_train_dataset_indices==expected


def test_disabled_filter_does_not_change_setup(monkeypatch):
    called=[];monkeypatch.setattr(VanillaDataManager,'setup_train',lambda self:called.append(True))
    dm=LusterOpacityDataManager.__new__(LusterOpacityDataManager)
    dm.config=SimpleNamespace(train_camera_ids=None)
    dm.setup_train()
    assert called==[True] and not hasattr(dm,'_allowed_train_dataset_indices')
