"""Train-overlap micro-registration of projective texture coordinates.

This opt-in diagnostic leaves mesh/cameras unchanged but deliberately adjusts
texture UVs. Each corrected warp samples its original train RGB once, not an
already warped image or a mixture. No held-out RGB or semantic mask is used.
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from audit_source_epipolar_residuals import peak_offset
from surface_color_field import solve_surface_field


def registration_observations(primary,source,primary_valid,source_valid,*,stride=32,radius=16,search_radius=5):
    """Spatially held-out NCC patch translations, expressed in target pixels."""
    if primary.shape!=source.shape or primary.ndim!=2:raise ValueError('Expected equal grayscale images')
    rows=[];height,width=primary.shape;margin=radius+search_radius
    for y in range(margin,height-margin,stride):
        for x in range(margin,width-margin,stride):
            if not primary_valid[y-radius:y+radius,x-radius:x+radius].all():continue
            if not source_valid[y-margin:y+margin,x-margin:x+margin].all():continue
            ref=np.ascontiguousarray(primary[y-radius:y+radius,x-radius:x+radius])
            if ref.std()<.008:continue
            region=np.ascontiguousarray(source[y-margin:y+margin,x-margin:x+margin])
            scores=cv2.matchTemplate(region,ref,cv2.TM_CCOEFF_NORMED)
            v,u=np.unravel_index(scores.argmax(),scores.shape)
            delta=peak_offset(scores,u,v)+[u-search_radius,v-search_radius]
            if scores[v,u]<.8 or np.linalg.norm(delta)>4:continue
            held=((x//128)*73856093^(y//128)*19349663)%5==0
            rows.append({'x':x,'y':y,'dx':float(delta[0]),'dy':float(delta[1]),'held':held,
                         'ncc_zero':float(scores[search_radius,search_radius]),'ncc_best':float(scores[v,u])})
    return rows


def register_surface_textures(warped,valid,depth,native,uv,sample):
    if not(len(warped)==len(valid)==len(native)==len(uv)):raise ValueError('Source inventory mismatch')
    gray=[(rgb.permute(1,2,0).detach().cpu().numpy()@np.array([.2126,.7152,.0722],np.float32)) for rgb in warped]
    masks=[v.cpu().numpy() for v in valid];stats=[];offsets=[]
    height,width=depth.shape;device=depth.device
    yy,xx=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing='ij')
    xx=xx.float();yy=yy.float();output=[warped[0]];new_valid=[valid[0]]
    for rank in range(1,len(warped)):
        rows=registration_observations(gray[0],gray[rank],masks[0],masks[rank])
        fit=[r for r in rows if not r['held']];held=[r for r in rows if r['held']]
        if len(fit)<12 or len(held)<3:
            output.append(warped[rank]);new_valid.append(valid[rank]);offsets.append(None)
            stats.append({'rank':rank,'corrected':False,'reason':'insufficient train-overlap patches','fit':len(fit),'held':len(held)});continue
        data=torch.zeros((1,2,height,width),device=device);weight=torch.zeros((1,1,height,width),device=device)
        for r in fit:
            x,y=r['x'],r['y'];data[0,:,y-2:y+3,x-2:x+3]=torch.tensor([r['dx'],r['dy']],device=device)[:,None,None]
            weight[0,0,y-2:y+3,x-2:x+3]=10*(r['ncc_best']-.8)/.2
        field,solver=solve_surface_field(data,weight,depth,smoothness=64,ridge=.001,max_iterations=1536,tolerance=1e-4)
        if not solver['converged']:raise RuntimeError(f'Texture registration failed to converge: {solver}')
        field=field[0];length=field.square().sum(0).sqrt();field*=torch.minimum(torch.ones_like(length),4/length.clamp_min(1e-8))
        errors=[]
        for r in held:
            shift=field[:,r['y'],r['x']].cpu().numpy()
            errors.append(float(np.linalg.norm(shift-[r['dx'],r['dy']])))
        before=float(np.median([np.hypot(r['dx'],r['dy']) for r in held]));after=float(np.median(errors))
        row={'rank':rank,'fit':len(fit),'held':len(held),'held_shift_length_before':before,
             'held_shift_residual_after':after,'solver':solver,'observations':rows}
        if after>=before:
            output.append(warped[rank]);new_valid.append(valid[rank]);offsets.append(None)
            stats.append({**row,'corrected':False,'reason':'held train registration did not improve'});continue
        tx=xx+field[0];ty=yy+field[1]
        sampled_uv=sample(torch.stack(uv[rank]),tx,ty)
        sampled_valid=sample(valid[rank].float()[None],tx,ty)[0]>.999
        neighbor_depth=sample(depth[None],tx,ty)[0]
        same_layer=(neighbor_depth>0)&((neighbor_depth.clamp_min(1e-7)/depth.clamp_min(1e-7)).log().abs()<.005)
        corrected=sample(native[rank],sampled_uv[0],sampled_uv[1])
        safe=valid[rank]&sampled_valid&same_layer
        # Unsafe coordinate adjustments revert to the original single source;
        # never expand visibility or remove a surface from the image.
        output.append(torch.where(safe[None],corrected,warped[rank]));new_valid.append(valid[rank])
        offset=torch.where(safe[None],sampled_uv-torch.stack(uv[rank]),0)
        offsets.append(offset.cpu().numpy())
        stats.append({**row,'corrected':True,'adjusted_pixels':int(safe.sum()),
                      'maximum_target_shift_pixels':float(field.square().sum(0).sqrt().max()),'native_resampling_count':1})
    return output,new_valid,offsets,{'enabled':True,'uses_eval_rgb':False,'uses_semantic_masks':False,
          'mesh_changed':False,'camera_matrices_changed':False,'texture_uvs_changed':True,
          'source_averaging':False,'primary_unchanged':True,'view_dependent':True,'sources':stats}
