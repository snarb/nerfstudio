#!/usr/bin/env python3
"""Qualify train-only COLMAP camera-z depths using independent camera witnesses."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from build_colmap_patch_match_config import explicit_source_views
from import_colmap_mvs_depth_dataset import read_colmap_dense_array
from seed_colmap_from_nerfstudio import world_to_camera


def witness_residuals(depth,reference,source_depth,source):
    """Reproject camera-z measurements; reject occlusion and mismatched surfaces."""
    device=depth.device
    h,w=depth.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float32),
                         torch.arange(w,device=device,dtype=torch.float32),indexing='ij')
    camera=torch.stack([(xx+.5-reference['cx'])/reference['fl_x'],
                        (yy+.5-reference['cy'])/reference['fl_y'],torch.ones_like(xx)],-1)
    xyz=camera*depth[...,None]
    wr=torch.tensor(world_to_camera(reference),device=device,dtype=torch.float32)
    ws=torch.tensor(world_to_camera(source),device=device,dtype=torch.float32)
    world=(xyz-wr[:3,3])@wr[:3,:3]
    predicted=world@ws[:3,:3].T+ws[:3,3]
    z=predicted[...,2]
    u=source['fl_x']*predicted[...,0]/z.clamp_min(1e-8)+source['cx']-.5
    v=source['fl_y']*predicted[...,1]/z.clamp_min(1e-8)+source['cy']-.5
    sh,sw=source_depth.shape
    inside=(u>=0)&(v>=0)&(u<sw-1)&(v<sh-1)&(z>0)&(depth>0)
    # Nearest depth preserves discontinuities instead of blending foreground/background.
    ix=u.round().long().clamp(0,sw-1); iy=v.round().long().clamp(0,sh-1)
    observed=source_depth[iy,ix]
    inside &= torch.isfinite(observed)&(observed>0)
    relative=(observed-z).abs()/z.clamp_min(1e-8)
    # The observed value belongs to the rounded source pixel, not the
    # continuous projection. Backproject that actual measurement; otherwise
    # the round-trip check hides subpixel correspondence errors.
    source_camera=torch.stack([(ix+.5-source['cx'])/source['fl_x'],
                               (iy+.5-source['cy'])/source['fl_y'],torch.ones_like(u)],-1)*observed[...,None]
    back_world=(source_camera-ws[:3,3])@ws[:3,:3]
    back=back_world@wr[:3,:3].T+wr[:3,3]
    bu=reference['fl_x']*back[...,0]/back[...,2].clamp_min(1e-8)+reference['cx']-.5
    bv=reference['fl_y']*back[...,1]/back[...,2].clamp_min(1e-8)+reference['cy']-.5
    reprojection=torch.sqrt((bu-xx).square()+(bv-yy).square())
    absolute=(back[...,2]-depth).abs()
    return inside,relative,reprojection,absolute,observed>z*1.03


def build(data:Path,depth_maps:Path,output:Path,minimum_witnesses=3,source_count=12):
    if (output/'manifest.json').exists():
        raise FileExistsError('Use a new output for a confidence policy')
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/'transforms.json').read_text())
    train=set(meta['train_filenames'])
    frames=[r for r in meta['frames'] if r['file_path'] in train]
    by_name={r['file_path']:r for r in frames}
    mapping=explicit_source_views(frames,source_count=source_count)
    arrays={}
    hashes={}
    for name,row in by_name.items():
        path=depth_maps/(name+'.geometric.bin')
        if not path.exists():
            continue
        raw=read_colmap_dense_array(path)[...,0]
        if raw.shape!=(row['h'],row['w']):
            raise ValueError(f'Depth dimensions differ from calibrated HD input: {name}')
        arrays[name]=torch.from_numpy(np.where(np.isfinite(raw)&(raw>0),raw,0).copy()).cuda()
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not arrays:
        raise ValueError('No measured train depths found')
    receipts={}
    for name,depth in arrays.items():
        count=torch.zeros_like(depth,dtype=torch.int16)
        error=torch.zeros_like(depth)
        contradictions=torch.zeros_like(depth,dtype=torch.int16)
        for source in mapping[name]:
            if source not in arrays:
                continue
            visible,relative,pixel,absolute,farther=witness_residuals(depth,by_name[name],arrays[source],by_name[source])
            agree=visible&(relative<=.01)&(pixel<=1.)
            count+=agree.to(torch.int16)
            error+=torch.where(agree,absolute,0)
            # Observed nearer geometry is an occlusion, not negative evidence.
            # Large farther-depth disagreement remains a veto on strong priors.
            # A reprojection failure alone is left unknown.
            contradictions+=(visible&farther&(pixel<=1.)).to(torch.int16)
        uncertainty=(error/count.clamp_min(1)).clamp_min(depth*1e-4)
        valid=(count>=minimum_witnesses)&(contradictions==0)&(depth>0)
        # Do not bridge depth jumps or thin silhouette boundaries with hard priors.
        dh=(depth-torch.roll(depth,1,0)).abs()/depth.clamp_min(1e-8)
        dw=(depth-torch.roll(depth,1,1)).abs()/depth.clamp_min(1e-8)
        boundary=(dh>.02)|(dw>.02)|(torch.roll(dh,-1,0)>.02)|(torch.roll(dw,-1,1)>.02)
        valid &= ~boundary
        stem=Path(name).stem
        path=output/(stem+'.npz')
        np.savez_compressed(path,depth=depth.cpu().numpy(),uncertainty=uncertainty.cpu().numpy(),
                            valid=valid.cpu().numpy(),witnesses=count.cpu().numpy())
        preview=np.zeros((*depth.shape,3),dtype=np.uint8)
        array=depth.cpu().numpy(); good=valid.cpu().numpy()
        if good.any():
            low,high=np.quantile(array[good],[.01,.99])
            tone=np.rint(np.clip((array-low)/max(high-low,1e-6),0,1)*255).astype('uint8')
            preview[...,0]=tone;preview[...,1]=np.where(good,255,0)
        im=Image.fromarray(preview);im.thumbnail((480,480));im.save(output/(stem+'_confidence.jpg'))
        receipts[Path(name).name]=dict(path=path.name,source_sha256=hashes[name],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            valid_pixels=int(valid.sum()),valid_fraction=float(valid.float().mean()),
            measured_fraction=float((depth>0).float().mean()),maximum_witnesses=int(count.max()))
        print(json.dumps(dict(image=name,**receipts[Path(name).name])),flush=True)
    manifest=dict(train_only=True,confidence_policy_version=2,
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  depth_convention='camera_z',coordinate_system='train_normalized_opengl',
                  normalization_scale=1.,transforms_sha256=hashlib.sha256((data/'transforms.json').read_bytes()).hexdigest(),
                  minimum_witnesses=minimum_witnesses,max_reprojection_px=1.,max_relative_depth_error=.01,
                  source_count=source_count,maps=receipts)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--depth-maps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-witnesses',type=int,default=3)
    p.add_argument('--source-count',type=int,default=12)
    args=p.parse_args()
    torch.set_num_threads(2)
    build(args.data,args.depth_maps,args.output,args.minimum_witnesses,args.source_count)
