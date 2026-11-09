#!/usr/bin/env python3
"""Actual frozen PNG8k gradients for two packed coordinate modes; zero updates."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for directory in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',
                  ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',
                  ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:
    sys.path.insert(0,str(directory))

import torch
from nerfstudio.utils.eval_utils import eval_setup
from probe_marshawn_occupancy import digest_model
from marshawn_packed_distortion_candidate import (
    PackedDistortionCapture,load_frozen_formula,packet_terms,file_sha)


def snapshot_rng():
    return (torch.get_rng_state().clone(),[s.clone()for s in torch.cuda.get_rng_state_all()])


def same_rng(a,b):
    return torch.equal(a[0],b[0])and all(torch.equal(x,y)for x,y in zip(a[1],b[1]))


def grad_summary(value,named,scale,reference=None):
    gradients=torch.autograd.grad(value*scale,[p for _,p in named],retain_graph=True,allow_unused=True)
    saved=[];squared=0.;by_group={};dot=0.
    for (name,parameter),gradient in zip(named,gradients):
        grad=torch.zeros_like(parameter)if gradient is None else gradient.detach()/scale
        if not torch.isfinite(grad).all():raise ValueError('Nonfinite actual native gradient '+name)
        grad=grad.float();squared+=float(grad.double().square().sum())
        group='mlp_head'if'mlp_head'in name else'mlp_base'if'mlp_base'in name else'other'
        by_group[group]=by_group.get(group,0.)+float(grad.double().square().sum())
        if reference is not None:dot+=float((grad.double()*reference[len(saved)].double()).sum())
        saved.append(grad)
    norm=squared**.5
    record=dict(l2=norm,groups_l2={k:v**.5 for k,v in by_group.items()},finite=True)
    if reference is not None:
        ref_norm=sum(float(g.double().square().sum())for g in reference)**.5
        record['cosine_to_RGB']=dot/max(norm*ref_norm,1e-300)
        record['norm_ratio_to_RGB']=norm/max(ref_norm,1e-300)
    return record,saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rays',type=int,default=2048)
    p.add_argument('--batches',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(30*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    run=ROOT/'final_pair/radius0/marshawn_001660_final_radius0/instant-ngp-bounded/cold_seed42'
    checkpoint=run/'nerfstudio_models/step-000008000.ckpt'
    source=ROOT/'code_sfm/nerfstudio/models/lookcloser.py';source_sha=file_sha(source)
    formula=load_frozen_formula(source,source_sha)
    candidate=__import__('marshawn_packed_distortion_candidate',fromlist=['NativePackedDistortionPipeline']).NativePackedDistortionPipeline
    def configure(cfg):
        cfg.eval_checkpoint=checkpoint
        cfg.pipeline._target=candidate
        cfg.pipeline.packed_distortion_mode='linear'
        cfg.pipeline.packed_distortion_weight=.01
        cfg.pipeline.packed_distortion_source=str(source)
        cfg.pipeline.packed_distortion_source_sha256=source_sha
        cfg.pipeline.model.eval_num_rays_per_chunk=512
        return cfg
    config,pipe,loaded,step=eval_setup(run/'config.yml',test_mode='val',update_config_callback=configure)
    if loaded.resolve()!=checkpoint.resolve()or step!=8000:raise ValueError('Wrong exact checkpoint')
    model=pipe.model;before=digest_model(model);keys=set(pipe.state_dict())
    named=[(n,p)for n,p in model.named_parameters()if p.requires_grad]
    groups={k:[id(p)for p in v]for k,v in model.get_param_groups().items()}
    named_ids=[id(p)for _,p in named]
    hooks=(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
    rng_before=snapshot_rng();rows=[]
    # Fork all CUDA generators: even stratified native sampling stays local to probe.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);pipe.train()
        for number in range(a.batches):
            pipe.dynamic_num_rays_per_batch=a.rays;pipe._update_pixel_samplers()
            with PackedDistortionCapture(model)as capture:
                with torch.autocast(device_type='cuda',enabled=True):
                    outputs,losses,metrics=pipe.get_train_loss_dict(8000+number)
                if capture.calls!=1 or capture.packet is None:raise ValueError('Actual native forward not captured')
                packet=capture.packet
                terms=packet_terms(packet,formula,.01)
                torch.testing.assert_close(losses['packed_distortion'],terms['linear'],rtol=0,atol=0)
            if hooks!=(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks)):
                raise ValueError('Nested objective hook leaked')
            rgb,rgb_grads=grad_summary(losses['rgb_loss'],named,65536.)
            summaries={'RGB':rgb}
            for name,value in [('LOS',losses['observed_sfm_free_space']),
                               ('surface',losses['observed_sfm_surface_band']),
                               ('distortion_linear_001',terms['linear']),
                               ('distortion_inverse_depth_001',terms['inverse_depth'])]:
                summaries[name],_=grad_summary(value,named,65536.,rgb_grads)
            for mode in ['distortion_linear_001','distortion_inverse_depth_001']:
                summaries[mode]['norm_ratio_to_LOS']=summaries[mode]['l2']/max(summaries['LOS']['l2'],1e-300)
                summaries[mode]['norm_ratio_to_surface']=summaries[mode]['l2']/max(summaries['surface']['l2'],1e-300)
            indices=packet['ray_indices'];counts=torch.bincount(indices,minlength=packet['num_rays'])
            row=dict(batch=number,probe_schedule_step=8000+number,optimizer_updates=0,
                rays=packet['num_rays'],packed_samples=int(indices.numel()),
                empty_rays=int((counts==0).sum()),weights_requires_grad=packet['weights'].requires_grad,
                weights_grad_fn=type(packet['weights'].grad_fn).__name__,
                native_graph_same_tensor=True,AMP=True,gradient_unscale=65536.,
                near_range=[float(packet['near'].min()),float(packet['near'].max())],
                far_range=[float(packet['far'].min()),float(packet['far'].max())],
                packed_indices_sha256=hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest(),
                gradient_norms=summaries,output_RGB_finite=bool(torch.isfinite(outputs['rgb']).all()),
                source_primitive='Exact frozen FP64 per-ray sum, then mean over ALL rays including empty rays')
            rows.append(row)
            (a.output/f'batch_{number:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(batch=number,rays=row['rays'],samples=row['packed_samples'],gradient_norms=summaries)),flush=True)
            del rgb_grads,losses,outputs,terms,packet,capture
        pipe.eval()
    if not same_rng(rng_before,snapshot_rng()):raise ValueError('Probe changed global RNG')
    after=digest_model(model)
    if before!=after or keys!=set(pipe.state_dict()):raise ValueError('NN/grid/checkpoint keys changed')
    if groups!={k:[id(p)for p in v]for k,v in model.get_param_groups().items()}or named_ids!=[id(p)for _,p in model.named_parameters()if p.requires_grad]:
        raise ValueError('Native trainable groups changed')
    report=dict(passed=True,time=time.time(),optimizer_updates=0,training_launched=False,
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),step=step,
        original_state_sha256=before,final_state_sha256=after,
        RNG_unchanged=True,field_grid_unchanged=True,parameter_groups_unchanged=True,
        checkpoint_keyset_unchanged=True,hooks_removed=True,
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0)),
        source_hashes=dict(probe=file_sha(__file__),candidate=file_sha(Path(__file__).with_name('marshawn_packed_distortion_candidate.py')),
            frozen_formula=source_sha,native_model=file_sha(inspect.getsourcefile(type(model))),
            native_field=file_sha(inspect.getsourcefile(type(model.field)))),batches=rows)
    (a.output/'actual_cuda_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,output=str(a.output),optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
