"""Derived actor datasets and genuinely fitted frequency maps for recovery probes.

Inputs are symlinked/read-only; the source mesh bundle is never mutated. A subset
is a diagnosis, not the final dataset. The actual eval remains evaluation-only.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from mesh_distillation_background import BUNDLE, masks, read, write, sha, check
from nerfstudio.scripts.lookcloser_preprocess import train_progressive_and_estimate_frequency_map, save_frequency_metadata


def link(source, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(source.resolve())


def prepare(args):
    source=BUNDLE/args.domain; meta=read(source/'transforms.json')
    train=set(meta['train_filenames']); validation=set(meta['val_filenames'])
    rows=[r for r in meta['frames'] if r['file_path'] in train]
    chosen=rows if args.indices=='all' else [rows[int(i)] for i in args.indices.split(',')]
    vals=[r for r in meta['frames'] if r['file_path'] in validation][:args.eval_count]
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    request=dict(domain=args.domain,indices=args.indices,eval_count=args.eval_count,source_manifest_sha256=sha(BUNDLE/'dataset_hashes.json'))
    if (out/'request.json').exists() and read(out/'request.json')!=request:
        raise ValueError('Derived data request mismatch')
    all_masks=masks(BUNDLE) if args.domain=='real' else None
    for row in chosen+vals:
        stem=Path(row['file_path']).stem
        link(source/row['file_path'],out/row['file_path'])
        if args.domain=='synthetic':
            for key in ['mask_path','confidence_file_path','depth_file_path']:
                link(source/row[key],out/row[key])
            row['evaluation_mask_path']=row['mask_path']
        else:
            # Teacher holes never remove real supervision. Actual eval uses the
            # pre-existing, GT-only face ROI and is not frequency-preprocessed.
            (out/'masks').mkdir(exist_ok=True)
            if row['file_path'] in train:
                mask=(all_masks[row['physical_camera']]>0).astype('uint8')*255
            else:
                mask=np.zeros((row['h'],row['w']),np.uint8)
                for polygon in read(source/'face_roi.json')['include_polygons']:
                    cv2.fillPoly(mask,[np.array(polygon,np.int32)],255)
            Image.fromarray(mask).save(out/'masks'/f'{stem}.png')
            row['mask_path']=f'masks/{stem}.png';row['evaluation_mask_path']=row['mask_path']
            row.pop('depth_file_path',None)
    result={**meta,'frames':chosen+vals,'train_filenames':[r['file_path'] for r in chosen],
            'val_filenames':[r['file_path'] for r in vals],'test_filenames':[r['file_path'] for r in vals],
            'distillation':dict(actor_bounds=read(args.geometry)['actor_bounds'],source_bundle=str(BUNDLE),
                                domain=args.domain,diagnostic_subset=args.indices!='all')}
    write(out/'transforms.json',result);write(out/'request.json',request)


def frequencies(args):
    out=args.output;meta=read(out/'transforms.json');names=set(meta['train_filenames'])
    folder=out/'lookcloser_frequencies';folder.mkdir(exist_ok=True)
    torch.set_num_threads(2);started=time.monotonic()
    shard=getattr(args,'frequency_shard_index',0);shards=getattr(args,'frequency_shards',1)
    if not 0<=shard<shards:raise ValueError('Invalid frequency preprocessing shard')
    progress=out if shards==1 else out/f'frequency_worker_{shard}'
    progress.mkdir(exist_ok=True)
    for i,row in enumerate(r for r in meta['frames'] if r['file_path'] in names):
        if i%shards!=shard:continue
        stem=Path(row['file_path']).stem;dest=folder/f'{stem}.pt'
        request=dict(rgb_sha256=sha(out/row['file_path']),mask_sha256=sha(out/row['mask_path']),
                     per_level_steps=args.per_level_steps,seed=42,patch_size=8)
        receipt=folder/f'{stem}.receipt.json'
        if receipt.exists():
            old=read(receipt)
            if old['request']!=request or sha(dest)!=old['frequency_sha256']:
                raise ValueError('Frequency cache identity mismatch')
            continue
        torch.manual_seed(42);np.random.seed(42)
        rgb=torch.from_numpy(np.array(Image.open(out/row['file_path']))).cuda().float()/255
        valid=torch.from_numpy(np.array(Image.open(out/row['mask_path']))>0).cuda()
        if 'confidence_file_path' in row:
            valid &= torch.from_numpy(np.array(Image.open(out/row['confidence_file_path']))>0).cuda()
        model,freq,levels,_=train_progressive_and_estimate_frequency_map(rgb,steps=None,
            train_steps_per_level=args.per_level_steps,batch_size=8192,lr=.01,ssim_threshold=.95,
            patch_size=8,eval_patch_batch_size=1024,n_levels=16,n_features=2,min_res=16,max_res=8192,
            log2_hashmap_size=19,ssim_window_size=7,validity_mask=valid)
        patch_valid=F.avg_pool2d(valid[None,None].float(),8,8)[0,0]==1
        torch.save(freq.cpu(),dest);torch.save(patch_valid.cpu(),folder/f'{stem}.valid.pt')
        save_frequency_metadata(dest.with_suffix('.json'),stem,rgb.shape[:2],None,8,8,16,8192,16,2,19)
        write(receipt,dict(request=request,frequency_sha256=sha(dest),
            valid_patch_count=int(patch_valid.sum()),histogram=torch.bincount(levels[patch_valid].long(),minlength=16).tolist(),
            elapsed_seconds=time.monotonic()-started))
        if args.save_fit:
            from torchmetrics.functional import structural_similarity_index_measure
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            with torch.no_grad():
                h,w=rgb.shape[:2];yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
                uv=torch.stack([(xx.flatten()+.5)/w,(yy.flatten()+.5)/h],-1);pred=[]
                for start in range(0,len(uv),65536):pred.append(model.render_masked(uv[start:start+65536],max_active_level=15).float())
                pred=torch.cat(pred).reshape(h,w,3)
                x=torch.where(valid[...,None],pred,0).permute(2,0,1)[None];y=torch.where(valid[...,None],rgb,0).permute(2,0,1)[None]
                lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
                fit=dict(psnr=float(-10*torch.log10((pred[valid]-rgb[valid]).square().mean())),
                    ssim=float(structural_similarity_index_measure(x,y,data_range=1.)),lpips=float(lpips(x,y)))
                write(folder/f'{stem}.fit.json',fit)
                crop=torch.cat([rgb[256:960,640:1408],pred[256:960,640:1408]],1)
                Image.fromarray((crop.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{stem}.fit.png')
                del lpips,pred,x,y,uv
        check(progress,'actor_frequency_preprocess',i+1,started)
        del model,rgb,valid,freq,levels;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','frequencies','both']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',choices=['synthetic','real'],default='synthetic');p.add_argument('--indices',default='33')
    p.add_argument('--eval-count',type=int,default=2);p.add_argument('--per-level-steps',type=int,default=200)
    p.add_argument('--save-fit',action='store_true')
    p.add_argument('--frequency-shards',type=int,default=1)
    p.add_argument('--frequency-shard-index',type=int,default=0)
    p.add_argument('--geometry',type=Path,default=Path('/mnt/data/dec5_lookcloser_mesh_distillation_v1/geometry.json'))
    a=p.parse_args()
    if a.action!='frequencies':prepare(a)
    if a.action!='prepare':frequencies(a)


if __name__=='__main__':main()
