import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_release_common import (authoritative_grid, authoritative_labels,
    native_bounds, point_cells, ray_box_interval, trace_profiles, validate_review_path, edge_point_sets,
    interpolate_intrinsics)
from marshawn_occupancy_audit import dilate_one
from probe_marshawn_occupancy import file_sha
from verify_marshawn_release import profile_metrics, MATCHED_SELECTION_SHA


def archive(tmp_path, radius):
    core=np.zeros((1,5,5,5),bool);core[0,2,2,2]=True
    active=dilate_one(core) if radius else core.copy()
    # Deliberately incompatible CPU threshold: exported GPU core wins authority.
    ema=np.zeros(core.shape,np.float32)
    box=np.array([[-1,-1,-1,1,1,1]],np.float32)
    path=tmp_path/'grid.npz'
    np.savez(path,core=core,active=active,ema=ema.reshape(-1),cascade_aabbs=box,
        world_aabb=box.reshape(2,3),resolution=np.array([5,5,5]))
    receipt=tmp_path/'grid.json'
    record=dict(grid_archive_sha256=file_sha(path),checkpoint_sha256='unused',radius=radius,
        core_cells=1,active_cells=int(active.sum()),checkpoint_step=31,core_update_step=16,native_period=16)
    receipt.write_text(json.dumps(record))
    return path,receipt,core,active,box


@pytest.mark.parametrize('radius',[0,1])
def test_exported_gpu_core_authority_and_no_redilation(tmp_path,radius):
    path,receipt,expected,active,box=archive(tmp_path,radius)
    core,loaded,ema,boxes,record=authoritative_grid(path,receipt)
    assert np.array_equal(core,expected) and np.array_equal(loaded,active)
    assert record['adapter_cpu_core_difference']==1
    assert record['cpu_reconstruction_used'] is False
    labels=authoritative_labels(core,loaded)
    assert (labels==1).sum()==1
    assert (labels==2).sum()==(26 if radius else 0)
    assert loaded.sum()==(27 if radius else 1)


def test_reject_cumulative_or_stale_policy_and_hash(tmp_path):
    path,receipt,core,active,box=archive(tmp_path,1)
    record=json.loads(receipt.read_text());record['radius']=0;receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='policy'):authoritative_grid(path,receipt)
    record['radius']=1;record['core_update_step']=0;receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Stale'):authoritative_grid(path,receipt)
    record['core_update_step']=16;record['grid_archive_sha256']='bad';receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='hash'):authoritative_grid(path,receipt)


def test_halfopen_cell_outside_not_false_negative(tmp_path):
    _,_,core,active,boxes=archive(tmp_path,0)
    indices,inside,occupied=point_cells([[0,0,0],[1,0,0],[-1,0,0]],active,boxes)
    assert inside.tolist()==[True,False,True]
    assert occupied.tolist()==[True,False,False]
    assert (inside & ~occupied).sum()==1
    assert ray_box_interval([-2,0,0],[1,0,0],boxes[0])==pytest.approx((1,3))
    with pytest.raises(ValueError):ray_box_interval([-2,2,0],[1,0,0],boxes[0])


def test_native_bounds_without_collider_requires_shared_sampler():
    box=torch.tensor([[-1.,-1,-1],[1.,1,1]])
    grid=SimpleNamespace(aabbs=box.reshape(1,6))
    model=SimpleNamespace(scene_aabb=box.flatten(),scene_box=SimpleNamespace(aabb=box),
        field=SimpleNamespace(aabb=box),occupancy_grid=grid,config=SimpleNamespace(enable_collider=False),
        sampler=SimpleNamespace(occupancy_grid=grid),collider=None)
    assert len(native_bounds(model))==4
    model.field.aabb=box+1
    with pytest.raises(ValueError,match='disagree'):native_bounds(model)
    model.field.aabb=box;model.sampler.occupancy_grid=object()
    with pytest.raises(ValueError,match='clipping'):native_bounds(model)


def test_actual_weights_hook_and_allcells_restore_even_on_failure():
    from nerfstudio.field_components.field_heads import FieldHeadNames
    class Field(nn.Module):
        def forward(self,samples):return {FieldHeadNames.DENSITY:torch.ones((2,1))}
    class Accum(nn.Module):
        def forward(self,*,weights):return weights.sum()[None]
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.field=Field();self.renderer_accumulation=Accum()
            self.occupancy_grid=SimpleNamespace(binaries=torch.tensor([False,True]))
            self.fail=False
        def forward(self,ray):
            if self.fail and self.occupancy_grid.binaries.all():raise RuntimeError('probe failure')
            sample=SimpleNamespace(frustums=SimpleNamespace(starts=torch.tensor([[0.],[1.]]),ends=torch.tensor([[1.],[2.]])))
            sigma=self.field(sample)[FieldHeadNames.DENSITY]
            alpha=1-torch.exp(-sigma)
            weights=torch.cat([alpha[:1],alpha[1:]*torch.exp(-sigma[:1])])
            return {'accumulation':self.renderer_accumulation(weights=weights)}
    model=Model();before=model.occupancy_grid.binaries.clone()
    profiles=trace_profiles(model,None)
    assert set(profiles)=={'native','all_cells'}
    assert profiles['native']['weights'].sum()==pytest.approx(1-np.exp(-2))
    assert profiles['native']['weight_max_error']<1e-6
    assert torch.equal(before,model.occupancy_grid.binaries)
    assert len(model.field._forward_hooks)==len(model.renderer_accumulation._forward_pre_hooks)==0
    model.fail=True
    with pytest.raises(RuntimeError,match='probe failure'):trace_profiles(model,None)
    assert torch.equal(before,model.occupancy_grid.binaries)
    assert len(model.field._forward_hooks)==len(model.renderer_accumulation._forward_pre_hooks)==0


def test_prefix_five_percent_and_band_one_percent_analytic():
    sigma=np.ones(100)*2;starts=np.arange(100)/10;ends=starts+.1
    alpha=1-np.exp(-sigma*.1);weights=np.exp(-2*starts)*alpha
    values=profile_metrics(dict(starts=starts,ends=ends,sigma=sigma,weights=weights),0,10,2)
    assert values['free_end']==pytest.approx(1.9)
    assert values['surface_band_halfwidth']==pytest.approx(.02)
    assert values['free_mass']==pytest.approx(1-np.exp(-3.8))
    assert values['surface_band_mass']==pytest.approx(np.exp(-3.96)-np.exp(-4.04))
    assert len(MATCHED_SELECTION_SHA)==64


def test_path_hash_rigid_pose_and_train_anchor_binding(tmp_path):
    a=np.eye(4);b=np.eye(4);b[0,3]=2;pose=np.eye(4);pose[0,3]=1
    path=tmp_path/'path.json';receipt=tmp_path/'receipt.json'
    data=dict(render_width=640,render_height=320,camera_path=[dict(camera_to_world=pose.reshape(-1).tolist(),fov=60,aspect=2)])
    path.write_text(json.dumps(data))
    record=dict(path_sha256=file_sha(path),records=[dict(anchors=['AA','BB'],fraction=.5)],
        heldout_rgb_used=False,direct_train_rgb_read=False,path_type='test observed arc')
    receipt.write_text(json.dumps(record))
    frames=[dict(physical_camera='AA',transform_matrix=a.tolist()),dict(physical_camera='BB',transform_matrix=b.tolist())]
    assert validate_review_path(path,receipt,frames,expected_views=1)['train_anchor_geometry_verified']
    frames[1]['transform_matrix']=a.tolist()
    with pytest.raises(ValueError,match='anchor geometry'):validate_review_path(path,receipt,frames,expected_views=1)
    data['camera_path'][0]['camera_to_world'][0]=2;path.write_text(json.dumps(data));record['path_sha256']=file_sha(path);receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Nonrigid'):validate_review_path(path,receipt,expected_views=1)


def test_guard_coverage_is_not_mislabeled_as_active_false_negative():
    inside=np.array([1,1,1,1,0],bool);active=np.array([0,1,1,0,0],bool);core=np.array([0,0,1,0,0],bool)
    sets=edge_point_sets(inside,active,core,np.array([100,20,10,200,1000]),[0,1],limit=2)
    assert sets['active_miss'].tolist()==[0,3]
    assert sets['guard_admitted_core_miss'].tolist()==[1]


def test_exact_intrinsics_interpolate_axes_and_principal_point_independently():
    first=dict(fl_x=1400,fl_y=1300,cx=951,cy=502,w=1920,h=1012)
    second=dict(fl_x=1500,fl_y=1350,cx=963,cy=510,w=1920,h=1012)
    result=interpolate_intrinsics(first,second,.25)
    assert result==dict(fl_x=1425.,fl_y=1312.5,cx=954.,cy=504.,w=1920,h=1012)
    assert interpolate_intrinsics(first,second,0)==first
    assert interpolate_intrinsics(first,second,1)==second
    with pytest.raises(ValueError):interpolate_intrinsics(first,dict(second,w=960),.5)
    with pytest.raises(ValueError):interpolate_intrinsics(first,second,1.1)
