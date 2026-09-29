"""Small CUDA integration gate, not a scene-quality experiment."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from nerfstudio.pipelines.mesh_distillation_pipeline import DistillationPipelineConfig, DistillationParserConfig
from nerfstudio.model_components.mesh_distillation import tensor_digest


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Sampler uses CUDA buckets')
def test_boundary_sampling_uses_valid_matte_gradients_and_preserves_cache_identity():
    from types import SimpleNamespace
    from nerfstudio.pipelines.mesh_distillation_pipeline import MaskedFrequencySamplerConfig
    dataset=SimpleNamespace(metadata={'distillation_root':'.','distillation_split':'train'},
        image_filenames=[Path('train_0000.png'),Path('train_0001.png')])
    class Dataset:
        metadata=dataset.metadata
        image_filenames=dataset.image_filenames
        def __len__(self):return 2
    cfg=MaskedFrequencySamplerConfig(num_rays_per_batch=64,enable_fas=False,boundary_fraction=1.,boundary_radius=2)
    sampler=cfg.setup(dataset=Dataset())
    alpha=torch.zeros(2,32,32,1);alpha[:,8:24,8:24]=1
    mask=torch.ones_like(alpha);mask[:,:16,:16]=0
    valid=torch.ones_like(alpha);valid[1]=0
    batch=dict(image=torch.zeros(2,32,32,3,device='cuda'),image_idx=torch.tensor([1,0]),
        alpha_target=alpha,alpha_valid=valid,mask=mask.cuda(),confidence=torch.ones_like(alpha))
    sampled=sampler.sample(batch)
    assert sampled['mask'].all() and sampled['alpha_valid'].all()
    assert (sampled['indices'][:,0]==1).all()
    y,x=sampled['indices'][:,1:].unbind(-1)
    assert (((x>=6)&(x<26)&(((y>=6)&(y<10))|((y>=22)&(y<26))))|
            ((y>=6)&(y<26)&(((x>=6)&(x<10))|((x>=22)&(x<26))))).all()
    # A scene without an eligible boundary falls back to the ordinary buckets.
    batch['alpha_target']=torch.ones_like(alpha);sampler.buckets=None
    assert sampler.sample(batch)['indices'].shape==(64,3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='LookCloser requires CUDA')
def test_data_mask_depth_training_and_frozen_phase_transition(tmp_path, monkeypatch):
    (tmp_path/'images').mkdir();(tmp_path/'lookcloser_frequencies').mkdir()
    frames=[]
    for i,name in enumerate(('train_0000','train_0001','val_0000')):
        image=np.zeros((16,16,3),'uint8');image[4:12,4:12]=[120+i*20,80,40]
        Image.fromarray(image).save(tmp_path/'images'/f'{name}.png')
        mask=np.zeros((16,16),'uint8');mask[4:12,4:12]=255
        Image.fromarray(mask).save(tmp_path/f'{name}_mask.png')
        Image.fromarray(255-mask).save(tmp_path/f'{name}_empty.png')
        with gzip.open(tmp_path/f'{name}.npy.gz','wb') as f:np.save(f,np.where(mask>0,3.,0.).astype('float32'))
        pose=np.eye(4);pose[2,3]=3.;pose[0,3]=i*.05
        frames.append(dict(file_path=f'images/{name}.png',mask_path=f'{name}_mask.png',
            confidence_file_path=f'{name}_mask.png',evaluation_mask_path=f'{name}_mask.png',
            depth_file_path=f'{name}.npy.gz',transform_matrix=pose.tolist(),fl_x=32.,fl_y=32.,cx=8.,cy=8.,w=16,h=16))
        frames[-1]['empty_mask_path']=f'{name}_empty.png'
        detail=np.zeros_like(mask);detail[6:10,:]=255
        Image.fromarray(detail).save(tmp_path/f'{name}_detail.png');frames[-1]['detail_mask_path']=f'{name}_detail.png'
        if name.startswith('train'):
            Image.fromarray(np.full((16,16),128,'uint8')).save(tmp_path/f'{name}_alpha.png')
            Image.fromarray(np.full((16,16,3),100,'uint8')).save(tmp_path/f'{name}_foreground.png')
            frames[-1].update(alpha_file_path=f'{name}_alpha.png',foreground_file_path=f'{name}_foreground.png')
            path=tmp_path/'lookcloser_frequencies'/f'{name}.pt';torch.save(torch.full((2,2),128.),path)
            torch.save(torch.ones(2,2,dtype=torch.bool),path.with_name(name+'.valid.pt'))
            path.with_suffix('.json').write_text(json.dumps(dict(value_type='scalar_resolution',min_res=16,max_res=8192,n_levels=16,patch_size=8,stride=8,image_shape=[16,16,3])))
    meta=dict(camera_model='OPENCV',frames=frames,train_filenames=[f['file_path'] for f in frames[:2]],
              val_filenames=[frames[-1]['file_path']],test_filenames=[frames[-1]['file_path']],
              distillation=dict(actor_bounds=[[-1.,-1.,-1.],[1.,1.,1.]]))
    (tmp_path/'transforms.json').write_text(json.dumps(meta))
    cfg=DistillationPipelineConfig(grid_update_interval=1)
    cfg.datamanager.dataparser=DistillationParserConfig(data=tmp_path,orientation_method='none',center_method='none',
        auto_scale_poses=False,scale_factor=1.,depth_unit_scale_factor=1.,downscale_factor=1,load_3D_points=False,eval_mode='filename')
    cfg.datamanager.train_num_rays_per_batch=32;cfg.datamanager.eval_num_rays_per_batch=32
    cfg.model.grid_resolution=8;cfg.model.log2_hashmap_size=10;cfg.model.ray_sampling_mode='fixed'
    cfg.model.fixed_num_samples_per_ray=16;cfg.model.depth_loss_steps=2
    pipe=cfg.setup(device='cuda');pipe.train()
    opt=torch.optim.Adam(pipe.model.field.parameters(),lr=.01)
    for step in range(3):
        _,losses,_=pipe.get_train_loss_dict(step)
        assert ('depth_loss' in losses)==(step<2)
        opt.zero_grad();sum(losses.values()).backward();opt.step()
    assert pipe.model.freq_grid.observed.any() and pipe.model.freq_grid.initialized
    for _ in range(5):
        _,batch=pipe.datamanager.next_train(3)
        assert batch['mask'].all() and (batch['confidence']>0).all()
        expected=(120+batch['indices'][:,0]*20)/255
        torch.testing.assert_close(batch['image'][:,0].cpu(),expected)
    state={k:v.clone() for k,v in pipe.state_dict().items()};grid_before=pipe.frequency_digest()
    resumed_cfg=deepcopy(cfg);resumed_cfg.model.freeze_frequency_grid=True;resumed_cfg.grid_update_interval=0
    resumed=resumed_cfg.setup(device='cuda');resumed.load_pipeline(state,2);resumed.train()
    assert resumed.frequency_digest()==grid_before
    opt2=torch.optim.Adam(resumed.model.field.parameters(),lr=.002);opt2.load_state_dict(opt.state_dict())
    for group in opt2.param_groups:group['lr']=.002
    first=next(resumed.model.field.parameters());weights_before=tensor_digest(first)
    occupancy_before=tensor_digest(resumed.model.occupancy_grid.occs)
    for step in range(3,103):
        _,losses,_=resumed.get_train_loss_dict(step)
        assert 'depth_loss' not in losses
        opt2.zero_grad();sum(losses.values()).backward();opt2.step()
        if step%16==0:
            resumed.model._stable_update_occupancy_grid(step,lambda x:resumed.model.field.density_fn(x)*.01)
    assert resumed.frequency_digest()==grid_before
    assert tensor_digest(first)!=weights_before
    assert tensor_digest(resumed.model.occupancy_grid.occs)!=occupancy_before
    assert opt2.param_groups[0]['lr']==.002
    empty_cfg=deepcopy(cfg);empty_cfg.datamanager.pixel_sampler.empty_fraction=.25
    empty_cfg.model.empty_opacity_weight=.05
    empty_pipe=empty_cfg.setup(device='cuda');empty_pipe.train()
    _,batch=empty_pipe.datamanager.next_train(0)
    assert int(batch['empty_mask'].sum())==8
    assert not batch['mask'][batch['empty_mask'][:,0]>0].any()
    _,terms,_=empty_pipe.get_train_loss_dict(0)
    assert terms['known_empty']>0 and torch.isfinite(terms['known_empty'])
    detail_cfg=deepcopy(empty_cfg);detail_cfg.datamanager.pixel_sampler.detail_fraction=.25
    detail_pipe=detail_cfg.setup(device='cuda');detail_pipe.train()
    _,detail_batch=detail_pipe.datamanager.next_train(0)
    assert int((detail_batch['detail_mask']*detail_batch['mask']).sum())>=8
    assert int(detail_batch['empty_mask'].sum())==8
    assert torch.all(detail_batch['mask'][detail_batch['empty_mask'][:,0]==0]>0)
    # A batch entirely partitioned into detail and empty rays has no generic
    # frequency-sampled remainder; torch.multinomial must not receive zero.
    detail_pipe.datamanager.train_pixel_sampler.config.detail_fraction=.75
    _,partition_batch=detail_pipe.datamanager.next_train(0)
    assert int(partition_batch['empty_mask'].sum())==8
    assert int((partition_batch['detail_mask']*partition_batch['mask']).sum())==24
    matte_cfg=deepcopy(empty_cfg);matte_cfg.model.matte_opacity_weight=.02
    matte_pipe=matte_cfg.setup(device='cuda');matte_pipe.train()
    _,batch=matte_pipe.datamanager.next_train(0)
    torch.testing.assert_close(batch['foreground_target'],torch.full_like(batch['foreground_target'],100/255*128/255))
    assert batch['alpha_valid'].all()
    assert not matte_pipe.datamanager.eval_dataset[0]['alpha_valid'].any()
    _,terms,_=matte_pipe.get_train_loss_dict(0)
    assert terms['matte_opacity']>0 and torch.isfinite(terms['matte_opacity'])
    sum(terms.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in matte_pipe.model.field.parameters() if p.grad is not None)
    camera_cfg=deepcopy(cfg);camera_cfg.model.optimize_training_cameras=True
    camera_pipe=camera_cfg.setup(device='cuda');camera_pipe.train()
    _,terms,_=camera_pipe.get_train_loss_dict(0);sum(terms.values()).backward()
    adjustment=camera_pipe.model.training_camera_optimizer.pose_adjustment
    assert adjustment.grad is not None and torch.isfinite(adjustment.grad).all()
    assert adjustment.grad.abs().sum()>0
    camera_pipe.eval();rays,_=camera_pipe.datamanager.next_train(0)
    with torch.no_grad():
        first=camera_pipe.model(rays)['rgb'].clone();adjustment.fill_(.0002)
        second=camera_pipe.model(rays)['rgb']
    torch.testing.assert_close(first,second)
    # Frequency observations must use the same refined pose as rendering.
    camera_pipe.train()
    with torch.no_grad():
        adjustment.zero_(); adjustment[:, 0] = .1
    fixed_rays, fixed_batch = camera_pipe.datamanager.next_train(0)
    expected_points = fixed_rays.origins+fixed_rays.directions*(
        fixed_batch['depth_image'].cuda()*fixed_rays.metadata['directions_norm'])
    expected_points[:, 0] += .1
    observed = []
    with monkeypatch.context() as patch:
        patch.setattr(camera_pipe.datamanager, 'next_train', lambda step: (fixed_rays, fixed_batch))
        patch.setattr(camera_pipe.model.freq_grid, 'update_max', lambda points, levels: observed.append(points.clone()))
        camera_pipe._update_frequency_grid(0)
    torch.testing.assert_close(observed[0], expected_points)
    adaptive_cfg=deepcopy(empty_cfg);adaptive_cfg.model.ray_sampling_mode='adaptive'
    adaptive_cfg.model.adaptive_warmup_steps=0
    adaptive_pipe=adaptive_cfg.setup(device='cuda');adaptive_pipe.train()
    adaptive_pipe.model.occupancy_grid.binaries.fill_(True)
    rays,batch=adaptive_pipe.datamanager.next_train(0);rendered=adaptive_pipe.model(rays)
    assert torch.isfinite(rendered['optical_thickness']).all()
    torch.testing.assert_close(rendered['accumulation'],-torch.expm1(-rendered['optical_thickness']),atol=2e-5,rtol=2e-5)
    terms=adaptive_pipe.model.get_loss_dict(rendered,batch)
    assert torch.isfinite(terms['known_empty'])
    importance_cfg=deepcopy(empty_cfg);importance_cfg.model.fixed_importance_samples=32
    importance_cfg.model.fixed_stratified_sampling=True
    importance_cfg.model.depth_distribution_weight=.1
    importance_pipe=importance_cfg.setup(device='cuda');importance_pipe.train()
    _,terms,_=importance_pipe.get_train_loss_dict(0)
    sum(terms.values()).backward()
    assert all(torch.isfinite(v).all() for v in terms.values())
    assert all(torch.isfinite(p.grad).all() for p in importance_pipe.model.field.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='LookCloser requires CUDA')
def test_masked_frequency_fit_ignores_unknown_rgb():
    from nerfstudio.scripts.lookcloser_preprocess import train_progressive_and_estimate_frequency_map
    image=torch.full((16,16,3),float('nan'),device='cuda')
    image[:8,:8]=.4
    valid=torch.zeros((16,16),dtype=torch.bool,device='cuda');valid[:8,:8]=True
    model,frequency,levels,_=train_progressive_and_estimate_frequency_map(
        image,steps=None,train_steps_per_level=2,batch_size=64,lr=.01,
        ssim_threshold=.9,patch_size=8,eval_patch_batch_size=4,n_levels=2,
        n_features=2,min_res=16,max_res=32,log2_hashmap_size=8,ssim_window_size=7,
        validity_mask=valid)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert torch.isfinite(frequency).all()
    assert levels[0,1]==1 and levels[1,0]==1 and levels[1,1]==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='LookCloser requires CUDA')
def test_correct_sh_contract_has_unit_sphere_addition_theorem():
    import math
    import tinycudann as tcnn
    from nerfstudio.fields.lookcloser_field import LookCloserField
    # For orthonormal real SH through degree 3, sum(Y_lm^2)=16/(4*pi)
    # at every unit direction. Passing [-1,1] directly violates this identity.
    holder=type('EncodingHolder',(),{})()
    holder.direction_encoding=tcnn.Encoding(3,dict(otype='SphericalHarmonics',degree=4)).cuda()
    holder.correct_sh_directions=True
    directions=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0],[0,-1.,0]],device='cuda')
    correct=LookCloserField.encode_directions(holder,directions).float()
    torch.testing.assert_close(correct.square().sum(-1),torch.full((5,),16/(4*math.pi),device='cuda'),atol=.003,rtol=.003)
    holder.correct_sh_directions=False
    legacy=LookCloserField.encode_directions(holder,directions).float()
    torch.testing.assert_close(legacy,holder.direction_encoding(directions).float())
    assert legacy.square().sum(-1).max()>100
