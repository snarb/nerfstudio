#!/usr/bin/env python3
"""CPU-only real cache/sampling probe; no radiance model, optimizer, or READY."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace, MethodType


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def check_cache_ids(ids, count):
    if len(ids)!=count or set(ids)!=set(range(count)):
        raise ValueError('Cache must contain every unique absolute camera ID')


def check_map_shape(shape, image_shape, patch=8):
    if tuple(shape)!=(image_shape[0]//patch,image_shape[1]//patch):
        raise ValueError('Map/raster dimensions differ')
    if any(x%patch for x in image_shape):
        raise ValueError('This 4K generation must have complete patch coverage')


def check_source_inventory(root, inventory):
    for name,digest in inventory.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe inventory path')
        if sha(Path(root)/name)!=digest:raise ValueError('Source hash mismatch: '+name)


def run(root, campaign):
    started=time.time();events=[]
    def event(stage,**data):
        row=dict(stage=stage,elapsed_seconds=time.time()-started,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,**data)
        events.append(row)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    code=root/'code';sys.path[:0]=[str(code/'LookCloser/scripts'),str(code)]
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(1)
    from marshawn_semantic_4k_derivation import inspect,bind_config,validate_map_audit
    import train_marshawn_semantic_admission as worker
    from marshawn_tuning_sampling import SamplingAdapter,body80_quotas
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
    assert not torch.cuda.is_initialized()
    original_inventory=json.loads((campaign/'resolution_4k_train63_transfer/file_inventory.json').read_text())
    map_inventory=json.loads((root/'map_inventory.json').read_text())
    check_source_inventory(root,map_inventory)
    manifest=campaign/'semantic_actor_candidate_v2_20261106/source/manifest.json'
    proof=inspect(campaign/'data',root/'data',campaign/'resolution_4k_train63_transfer/file_inventory.json',manifest)
    validate_map_audit(json.loads((root/'aggregate.json').read_text()),proof['splits']['train'])
    receipt=root/'derivation_receipt.json';receipt.write_text(json.dumps(proof,indent=2)+'\n')
    spec=json.loads((campaign/'semantic_actor_candidate_v2_20261106/spec.json').read_text())
    spec['data']=str(campaign/'data');spec['output']=str(root/'NO_TRAINING_OUTPUT');spec['semantic_admission']['manifest']=str(manifest)
    cfg=bind_config(worker.build_config(spec),receipt,sha(receipt))
    assert cfg.pipeline.model.semantic_data==str(campaign/'data')
    assert not cfg.pipeline.datamanager.cache_train_rays and not cfg.pipeline.datamanager.cpu_fas_prefetch
    assert cfg.pipeline.frequency_projection_mode=='legacy'
    event('sources_verified',source_files=len(original_inventory),maps=63,HD_geometry_unchanged=True)
    # Only cache-loading parallelism is bounded. Frozen dataset decoding, ordered
    # future consumption, camera permutation and sampler implementations are intact.
    original_executor=concurrent.futures.ThreadPoolExecutor
    class BoundedExecutor(original_executor):
        def __init__(self,max_workers=None,*args,**kwargs):
            super().__init__(max_workers=min(2,max_workers or 2),*args,**kwargs)
    concurrent.futures.ThreadPoolExecutor=BoundedExecutor
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    try:dm=cfg.pipeline.datamanager.setup(device='cpu',test_mode='val',world_size=1,local_rank=0)
    finally:concurrent.futures.ThreadPoolExecutor=original_executor
    train=dm.train_image_dataloader.cached_collated_batch
    evaluation=dm.eval_image_dataloader.cached_collated_batch
    ids=train['image_idx'].tolist();check_cache_ids(ids,63)
    assert tuple(train['image'].shape)==(63,2024,3840,3)
    assert tuple(evaluation['image'].shape)==(3,1012,1920,3)
    assert len(dm.train_dataset)==63 and len(dm.eval_dataset)==3
    assert train['image'].device.type=='cpu' and evaluation['image'].device.type=='cpu'
    event('actual_datamanager_cached',train_shape=list(train['image'].shape),eval_shape=list(evaluation['image'].shape),camera_ids=ids,train_bytes=train['image'].numel()*train['image'].element_size(),loading_workers=2)
    grid=FrequencyGridManager(dm.train_dataset.scene_box,resolution=128,num_levels=16,min_res=16,max_res=8192)
    proxy=SimpleNamespace(datamanager=dm,config=cfg.pipeline,model=SimpleNamespace(freq_grid=grid),cached_freq_maps={},cached_freq_patch_sizes={},cached_freq_strides={},cached_freq_image_shapes={})
    for name in ('_read_frequency_metadata','_validate_frequency_schedule'):
        setattr(proxy,name,MethodType(getattr(LookCloserPipeline,name),proxy))
    proxy._expected_map_shape=LookCloserPipeline._expected_map_shape
    LookCloserPipeline._load_frequency_maps(proxy)
    assert set(proxy.cached_freq_maps)==set(range(63))
    for idx,arr in proxy.cached_freq_maps.items():
        check_map_shape(arr.shape,(2024,3840));assert proxy.cached_freq_image_shapes[idx]==(2024,3840)
        assert proxy.cached_freq_patch_sizes[idx]==proxy.cached_freq_strides[idx]==8
        assert torch.isfinite(arr).all() and arr.min()>=16 and arr.max()<=8192
    event('actual_pipeline_frequency_loader',maps=63,shape=[253,480],tail_pixels=0)
    # Stub has no parameters, forward, density, or optimizer. It supplies only the
    # configuration/get_loss_dict attributes required to install the real sampler.
    model_stub=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=lambda *a,**k:{})
    sampler=SamplingAdapter(SimpleNamespace(datamanager=dm,model=model_stub),root/'data','actor','body80',64)
    rows=[]
    for idx,row in enumerate(sampler.rows):
        pools={k:len(v)for k,v in row['pools'].items()};fas=[len(x)for x in row['fas']['cells']]
        assert pools['valid']>0 and sum(fas)>0
        rows.append(dict(camera=dm.train_dataset.image_filenames[idx].stem[6:],pools=pools,fas_cells_by_level=fas,proposal_means=row['proposal_means'].tolist()))
    event('actual_body80_initialized',cameras=len(rows),fas_cells=sum(sum(r['fas_cells_by_level'])for r in rows))
    lookup={int(c):i for i,c in enumerate(ids)};samples=[];clamps=[]
    for step in (0,4096,8192):
        sampler.set_step(step);batch=sampler.sample(train);indices=batch['indices'];arr=indices.numpy();assert len(arr)>0
        expected=body80_quotas(len(arr),step).tolist();assert sampler.last_counts==expected
        for camera in np.unique(arr[:,0]):
            take=np.flatnonzero(arr[:,0]==camera);yx=arr[take,1:];row=sampler.rows[camera]
            assert row['valid'][yx[:,0],yx[:,1]].all()
            assert np.array_equal(batch['actor_confidence'][take].numpy(),row['W64'][yx[:,0],yx[:,1]])
            assert torch.equal(batch['image'][take],train['image'][lookup[int(camera)],yx[:,0],yx[:,1]])
        assert np.isfinite(batch['actor_proposal_mean_weight']) and 0<batch['actor_proposal_mean_weight']<=1
        samples.append(dict(step=step,rays=len(arr),cameras_seen=len(np.unique(arr[:,0])),quotas=expected,proposal_mean=batch['actor_proposal_mean_weight'],indices_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),RGB_and_W64_gather_exact=True,valid_only=True))
        # Conservative geometry-only projection bounds, not rendered-depth update.
        rays=dm.train_ray_generator(indices[:1024]);o=rays.origins;d=rays.directions
        lower=(grid.aabb_min_buf-o)/d;upper=(grid.aabb_max_buf-o)/d
        near=torch.maximum(torch.minimum(lower,upper).amax(-1),torch.tensor(1e-6));far=torch.maximum(lower,upper).amin(-1);hit=far>near
        ix=indices[:1024];f2d=torch.stack([proxy.cached_freq_maps[int(c)][int(y)//8,int(x)//8]for c,y,x in ix]);focal=(dm.train_dataset.cameras.fx[:,0]+dm.train_dataset.cameras.fy[:,0])/2
        low=f2d*focal[ix[:,0]]/far;high=f2d*focal[ix[:,0]]/near
        clamps.append(dict(step=step,rays=len(ix),AABB_hits=int(hit.sum()),minimum_over_max_resolution_fraction=float((low[hit]>8192).float().mean()),maximum_over_max_resolution_fraction=float((high[hit]>8192).float().mean()),level15_at_far_fraction=float((grid.freq_to_level(low[hit])==15).float().mean()),basis='CPU ray/AABB near-far bounds only; actual rendered-depth update pending GPU technical gate'))
    event('actual_sampling_checked',samples=samples,projection_bounds=clamps)
    check_source_inventory(root/'data',original_inventory);check_source_inventory(root,map_inventory)
    for p,digest in proof['geometry_source_hashes'].items():assert sha(p)==digest
    assert not torch.cuda.is_initialized() and not (root/'data/ready.json').exists()
    result=dict(status='CPU_CACHE_AND_SAMPLING_PASS',time=time.time(),elapsed_seconds=time.time()-started,events=events,cameras=rows,samples=samples,projection_bounds=clamps,source_inventory_sha256=sha(campaign/'resolution_4k_train63_transfer/file_inventory.json'),map_inventory_sha256=sha(root/'map_inventory.json'),aggregate_sha256=sha(root/'aggregate.json'),derivation_receipt_sha256=sha(receipt),probe_sha256=sha(__file__),CUDA_initialized=False,radiance_model_instantiated=False,optimizer_instantiated=False,READY_created=False,actual_1024_frequency_update_tested=False,production_authorized=False,semantic_geometry_data=str(campaign/'data'),bundle='4K RGB/native confidence +4000-step maps; not pure-resolution causality',historical_MC_bag_semantic_error_preserved=True)
    (root/'cpu_complete.json').write_text(json.dumps(result,indent=2)+'\n');event('complete',result_sha256=sha(root/'cpu_complete.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.campaign.resolve())
