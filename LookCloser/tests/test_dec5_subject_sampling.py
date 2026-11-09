from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_subject_sampling import SubjectSampler
from nerfstudio.model_components.lookcloser_grid import normalized_frequency_focal
from nerfstudio.utils.lookcloser_rng import stream_seed,fork_seeded_rng


def test_sparse_depth_stream_is_distinct_and_does_not_advance_global_rng():
    seeds=[stream_seed(42,name,100) for name in ['pixel','occupancy','frequency_grid','sparse_depth']]
    assert len(set(seeds))==4
    torch.manual_seed(12);before=torch.get_rng_state().clone()
    with fork_seeded_rng(42,'sparse_depth',100,'cpu'):draw=torch.rand(5)
    torch.testing.assert_close(torch.get_rng_state(),before)
    with fork_seeded_rng(42,'sparse_depth',100,'cpu'):torch.testing.assert_close(torch.rand(5),draw)


def test_projected_frequency_is_invariant_to_image_resize_and_scene_similarity():
    f2d=64.;fx=9600.;fy=9650.;width=1920.;height=1080.;span=4.;z=5.
    expected=f2d*normalized_frequency_focal(fx,fy,width,height,span)/z
    resized=f2d*normalized_frequency_focal(fx*2,fy*2,width*2,height*2,span)/z
    similarity=f2d*normalized_frequency_focal(fx,fy,width,height,span*3)/(z*3)
    np.testing.assert_allclose([resized,similarity],[expected,expected],rtol=1e-12)
    assert 16 < expected < 8192
    assert f2d*(fx+fy)/2/z > 8192  # Mixing pixel focal with UV-grid resolution saturates this example.


def test_subject_samples_recollate_rgb_for_noncontiguous_cached_camera_ids(tmp_path):
    dataset=SimpleNamespace(
        image_filenames=[Path('frame_train_camera%d.png'%i) for i in range(59)],
        cameras=SimpleNamespace(height=torch.full((59,1),16),width=torch.full((59,1),16)),
    )
    for i in range(59):
        p=np.zeros((16,16,3),np.uint16);p[...,2]=65535
        p[2:6,2:6]=[65535,0,0];p[8:12,8:12]=[0,65535,0]
        np.savez_compressed(tmp_path/('camera%d.npz'%i),probability=p)
    images=torch.arange(2*16*16*3).reshape(2,16,16,3).float();ids=torch.tensor([47,3])
    def baseline(batch,**_):
        indices=torch.zeros((1000,3),dtype=torch.long);indices[:,0]=47
        return dict(indices=indices,image=batch['image'][0,0,0].expand(1000,3).clone())
    sampler=SimpleNamespace(sample=baseline,sample_count=10)
    policy=SubjectSampler(sampler,dataset,tmp_path,.6,.3,warmup=0,ramp=1)
    torch.manual_seed(7);result=policy.sample(dict(image=images,image_idx=ids))
    coordinates=result['indices'];local=(coordinates[:,0]==3).long()
    torch.testing.assert_close(result['image'],images[local,coordinates[:,1],coordinates[:,2]])
    assert set(coordinates[:,0].tolist())=={47,3}
    assert policy.audit['person_nonhair']>=300
    assert policy.audit['hair']>=420
    assert policy.audit['rays']==1000
    model=SimpleNamespace(training=True,get_metrics_dict=lambda outputs,batch:{})
    policy.install(model)
    packed=torch.tensor([0,0,1,5,9,9,20,23])
    model.get_metrics_dict(dict(rgb=torch.zeros((1000,3)),packed_ray_indices=packed),result)
    torch.testing.assert_close(policy.query_counts,torch.bincount(result['subject_sampling_class'][packed].long(),minlength=4))
    # A uniform sampler may expose a counter without advancing it. Quotas must
    # still ramp by real batches rather than silently remaining off forever.
    sampler.sample_count=0;sampler.sample=baseline
    uniform=SubjectSampler(sampler,dataset,tmp_path,1.,0.,warmup=0,ramp=1,boundary_width=3)
    first=uniform.sample(dict(image=images,image_idx=ids))
    second=uniform.sample(dict(image=images,image_idx=ids))
    assert int((first['subject_sampling_class']==0).sum())==0
    assert int((second['subject_sampling_class']==0).sum())>=700


def test_live_frequency_update_uses_camera_z_pixel_centers_and_no_autograd():
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    camera=Cameras(camera_to_worlds=torch.eye(4)[:3][None],fx=960.,fy=540.,cx=960.,cy=540.,width=1920,height=1080)
    grid=FrequencyGridManager(SceneBox(aabb=torch.tensor([[-4.,-4.,-4.],[4.,4.,4.]])),resolution=16,max_res=8192)
    class Dataset(SimpleNamespace):
        def __len__(self):return 1
    class Model:
        freq_grid=grid
        def __call__(self,rays):
            assert not torch.is_grad_enabled()
            z=-rays.directions[:,2]
            u=rays.directions[:,0]/z*960+960
            v=-rays.directions[:,1]/z*540+540
            torch.testing.assert_close((u-.5).round(),u-.5,atol=3e-4,rtol=0)
            torch.testing.assert_close((v-.5).round(),v-.5,atol=3e-4,rtol=0)
            return dict(depth=(2/z)[:,None],accumulation=torch.ones((len(z),1)))
    pipeline=SimpleNamespace(config=SimpleNamespace(independent_rng_streams=True,grid_update_batch_size=64,
        frequency_projection_mode='normalized_image',frequency_stride=8,frequency_patch_size=8),
        datamanager=SimpleNamespace(train_dataset=Dataset(cameras=camera)),model=Model(),device=torch.device('cpu'),
        cached_freq_maps={0:torch.full((135,240),16.)},cached_freq_strides={0:8},cached_freq_patch_sizes={0:8})
    LookCloserPipeline._update_frequency_grid(pipeline,1024)
    assert (grid.grid>0).any()
    torch.testing.assert_close(grid.grid[grid.grid>0],torch.full_like(grid.grid[grid.grid>0],2.))
