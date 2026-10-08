import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nerfstudio.model_components.lookcloser_occupancy import admit_occupancy_points_, load_occupancy_prior
from nerfstudio.models.lookcloser import LookCloserModel


def test_admission_keeps_unknown_and_existing_cells_and_ignores_invalid_points():
    grid = torch.zeros(1,2,2,2,dtype=torch.bool)
    grid[0,0,0,0] = True
    boxes = torch.tensor([[0.,0.,0.,1.,1.,1.]])
    points = torch.tensor([[.99,.01,.51],[-.01,0.,0.],[1.,0.,0.],[float('nan'),0.,0.]])
    admit_occupancy_points_(grid, boxes, points)
    assert grid[0,0,0,0] and grid[0,1,0,1] and grid.sum() == 2


def test_admission_uses_each_cascade_coordinate_system():
    grid = torch.zeros(2,4,4,4,dtype=torch.bool)
    boxes = torch.tensor([[-1.,-1.,-1.,1.,1.,1.],[-2.,-2.,-2.,2.,2.,2.]])
    admit_occupancy_points_(grid, boxes, torch.tensor([[1.5,1.5,1.5]]))
    assert not grid[0].any() and grid[1,3,3,3] and grid.sum() == 1


def test_prior_is_reapplied_after_learned_occupancy_reset_without_veto():
    learned = torch.zeros(1,2,2,2,dtype=torch.bool)
    learned[0,1,1,1] = True
    prior = torch.zeros_like(learned);prior[0,0,0,0] = True
    model = SimpleNamespace(config=SimpleNamespace(occupancy_prior_grid_path='enabled'),
                            occupancy_grid=SimpleNamespace(binaries=learned),occupancy_prior_grid=prior)
    LookCloserModel._admit_depth_occupancy(model)
    assert learned.sum() == 2
    learned.zero_()
    LookCloserModel._admit_depth_occupancy(model)
    assert torch.equal(learned,prior)
    model.config.occupancy_prior_grid_path = None
    learned.zero_()
    LookCloserModel._admit_depth_occupancy(model)
    assert not learned.any()


def receipt(tmp_path):
    boxes = torch.tensor([[0.,0.,0.,1.,1.,1.]])
    path = tmp_path/'grid.npz'
    np.savez_compressed(path,occupied=np.ones((1,2,2,2),dtype=bool),aabbs=boxes.numpy())
    metadata = dict(train_only=True,transforms_sha256='a'*64,grid_path=path.name,
                    grid_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = tmp_path/'manifest.json';manifest.write_text(json.dumps(metadata))
    return manifest,boxes,metadata


def test_load_requires_matching_hash_calibration_bounds_and_shape(tmp_path):
    path,boxes,_ = receipt(tmp_path)
    value = load_occupancy_prior(path,boxes,(1,2,2,2),'a'*64)
    assert value.dtype == torch.bool and value.all()
    for actual_boxes,shape,digest in [(boxes,(1,3,3,3),'a'*64), (boxes+1,(1,2,2,2),'a'*64),
                                      (boxes,(1,2,2,2),'b'*64)]:
        with pytest.raises(ValueError):
            load_occupancy_prior(path,actual_boxes,shape,digest)


@pytest.mark.parametrize('bad', ['heldout', 'tamper'])
def test_unqualified_or_changed_prior_is_rejected(tmp_path,bad):
    path,boxes,metadata = receipt(tmp_path)
    if bad == 'heldout':metadata['train_only'] = False
    else:metadata['grid_sha256'] = '0'*64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_occupancy_prior(path,boxes,(1,2,2,2),'a'*64)
