#!/usr/bin/env python3
"""Resumable per-frame native/hybrid renderer, followed by pixel-exact GT crops.

Invoked by the durable campaign postprocessor, never by a Codex session timer.
Every neural frame loads its OWN cold-trained model, geometry and source instant.
No synthetic continuation, frame interpolation or resizing of GT is permitted.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from marshawn_video_path import sha, write, native_row


def read(path): return json.loads(Path(path).read_text())


def source_eligibility(arrays,rgb_valid):
    required={'W64','valid','source_ineligible','unknown'}
    if not required<=set(arrays):raise ValueError('Hybrid source lacks qualified actor/crew/UNKNOWN eligibility')
    eligible=(arrays['W64']>0)&arrays['valid']&~arrays['source_ineligible']&~arrays['unknown']&rgb_valid
    if eligible.shape!=rgb_valid.shape:raise ValueError('Hybrid source eligibility raster differs')
    if np.any(eligible&(arrays['source_ineligible']|arrays['unknown'])):raise AssertionError('Unsafe source eligibility')
    return eligible


def save(path, rgb):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(temp,format='PNG')
    temp.replace(path)


def legacy(spec):
    """Reuse the measured Misha pixel-transfer implementation, not mesh code."""
    root=Path(spec['legacy_hybrid_root'])
    for key in ['campaign','hybrid_reference']:
        sys.path.insert(0,str(root/key))
    module=importlib.import_module('misha50_hybrid')
    if spec.get('legacy_hybrid_sha256') and sha(module.__file__)!=spec['legacy_hybrid_sha256']:
        raise ValueError('Historical hybrid source changed')
    return module


def photograph_samples(raw, uv, row, fit):
    """Decode native Log3G10 taps BEFORE bilinear linear-light interpolation.

UV coordinates refer to already-undistorted COLMAP pixel centers as array
indices. Lens calibration is evaluated at native resolution. No HD texture is
upscaled. Frozen k4, fixed gain4 and train-only scalar EV are applied once.
"""
    from prepare_marshawn_color_variants import vendor
    from marshawn_actor_coordinates import raw_array_indices
    from marshawn_radial_photometry import radial_ev
    pq=vendor(); full=native_row(row)
    native_uv=(np.asarray(uv,np.float64)+.5)*[8192/row['w'],4320/row['h']]
    xy=raw_array_indices(native_uv,full)
    valid=np.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<[8191,4319]).all(-1)
    safe=np.where(valid[...,None],xy,0);ij=np.floor(safe).astype(np.int32);frac=safe-ij
    linear=np.zeros((*uv.shape[:-1],3),np.float32)
    matrix=pq.rgb_to_rgb_matrix('rwg','rec709')
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        weight=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
        values=raw[ij[...,1]+dy,ij[...,0]+dx].astype(np.float64)/65535
        values=np.maximum(np.maximum(pq.log3g10_decode(values),0)@matrix.T,0)
        linear+=values.astype(np.float32)*weight[...,None].astype(np.float32)
    evs={r['camera']:r['scalar_correction_EV'] for r in fit['camera_rows']}
    evs.update({n:r['ev'] for n,r in fit.get('eval_predictions',{}).items()})
    ev=radial_ev(full,4,native_uv)-fit['radial_global_center_EV']+evs[row['physical_camera']]
    linear*= (4*np.exp2(ev))[...,None].astype(np.float32)
    positive=np.maximum(linear,0);positive=positive/(1+positive)
    rgb=np.where(positive<=.0031308,12.92*positive,1.055*positive**(1/2.4)-.055)
    return np.clip(rgb,0,1).astype(np.float32),valid


def geometry(model,row,destination,legacy_module,checkpoint_sha,chunk):
    receipt=destination.with_suffix('.json')
    binding=dict(row=row,checkpoint_sha256=checkpoint_sha,quantiles='conditional-q10-q50-q90-camera-z')
    if destination.exists():
        old=read(receipt)
        if old['binding']!=binding or old['sha256']!=sha(destination):
            raise ValueError('Geometry cache belongs to a different cold model/camera')
        with np.load(destination) as cache:return {k:cache[k].copy() for k in cache.files}
    import torch
    from render_luster_hybrid import packed_quantiles
    h,w=row['h'],row['w'];cam=legacy_module.camera(row,model.device)
    out=dict(rgb=np.zeros((h*w,3),np.float32),alpha=np.zeros(h*w,np.float32),
        q10=np.full(h*w,np.nan,np.float32),q50=np.full(h*w,np.nan,np.float32),q90=np.full(h*w,np.nan,np.float32))
    with torch.inference_mode():
        for start in range(0,h*w,chunk):
            ids=torch.arange(start,min(start+chunk,h*w),device=model.device)
            coords=torch.stack([ids//w,ids%w],-1).float()+.5
            rays=model.collider(cam.generate_rays(camera_indices=0,coords=coords))
            values=model.get_outputs(rays);q=packed_quantiles(values,rays)
            q*=(-(rays.directions@cam.camera_to_worlds[0,:3,:3])[:,2])[:,None]
            out['rgb'][start:start+len(ids)]=values['rgb'].float().cpu().numpy()
            out['alpha'][start:start+len(ids)]=values['accumulation'].float().cpu().numpy().reshape(-1)
            for j,k in enumerate(['q10','q50','q90']):out[k][start:start+len(ids)]=q[:,j].cpu().numpy()
    for k,v in out.items():out[k]=v.reshape((h,w,3) if k=='rgb' else (h,w))
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**out);temp.replace(destination)
    write(receipt,dict(binding=binding,sha256=sha(destination),rays_per_chunk=chunk))
    return out


class Sources:
    def __init__(self,data,record,spec,model,old,checkpoint_sha,output):
        self.data=data;self.record=record;self.spec=spec;self.model=model;self.old=old
        self.checkpoint_sha=checkpoint_sha;self.output=output
        meta=read(data/'transforms.json')
        self.rows={r['physical_camera']:dict(r,camera_id=i) for i,r in enumerate(meta['frames'])}
        self.allowed={r['physical_camera'] for r in meta['frames'] if r['file_path'] in meta['train_filenames']}
        self.fit=read(spec['photometry_fit']);self.raw={};self.masks={};self.geometry={};self.inventory={}

    def get(self,name):
        import tifffile
        from marshawn_video_path import raw_file
        if name not in self.allowed:raise ValueError('Pixel transfer requested non-TRAIN source')
        row=self.rows[name]
        if name not in self.raw:
            path=raw_file(name,int(self.record['frame_id']),self.spec['raw_roots'])
            if not path:raise FileNotFoundError('Missing SAME-INSTANT native texture '+name)
            self.raw[name]=tifffile.memmap(path,mode='r')
            self.inventory[name]=dict(path=path,sha256=sha(path),frame_id=self.record['frame_id'],
                native_wh=[8192,4320],resize=False)
            mask=np.asarray(Image.open(self.data/row['mask_path']).convert('L'))>=254
            semantic=self.data/'semantics'/f'{name}.npz'
            with np.load(semantic) as z:
                eligible=source_eligibility(z,mask)
                if eligible.shape!=(row['h'],row['w']):raise ValueError('Hybrid source eligibility raster differs')
            self.masks[name]=eligible
            self.inventory[name].update(semantic_sha256=sha(semantic),
                eligibility_sha256=__import__('hashlib').sha256(np.packbits(eligible).tobytes()).hexdigest(),
                eligible_pixels=int(eligible.sum()),eligibility='W64>0 AND valid AND NOTsource_ineligible AND NOTunknown AND calibratedRGBvalid',
                hard_erosion=False,unaccepted_pixels='unchanged_neural_fallback')
        if name not in self.geometry:
            self.geometry[name]=geometry(self.model,row,self.output/'sources'/f'{name}.npz',
                self.old,self.checkpoint_sha,self.spec.get('ray_chunk',1024))
        return row,self.geometry[name],self.raw[name],self.masks[name]


def render_neural(spec,request,record,out):
    frame=record['frame_id'];root=Path(spec['frame_root_template'].format(frame_id=frame))
    # Campaign loader validates all immutable recipe/checkpoint bindings.
    module=importlib.import_module(spec.get('pipeline_module','marshawn_video_train'))
    import torch
    from marshawn_gpu_isolation import verify_cuda
    expected=os.environ.get('MARSHAWN_RENDER_GPU_UUID')
    if not expected:raise ValueError('Neural rendering requires the durable physical GPU lease dispatcher')
    gpu_proof=verify_cuda(expected,torch)
    loaded=module.load_pipeline(root,checkpoint_kind=spec.get('checkpoint_kind','actor_lpips'),device=spec.get('device','cuda:0'))
    if isinstance(loaded,tuple):
        pipeline,binding=loaded
    else:
        pipeline=loaded;binding=read(root/'frame_completion.json')
    checkpoint_sha=binding.get('checkpoint_sha256')
    if not checkpoint_sha:raise ValueError('Loader must return checkpoint_sha256 provenance')
    model=pipeline.model;model.eval();old=legacy(spec)
    target=geometry(model,record['camera'],out/'target.npz',old,checkpoint_sha,spec.get('ray_chunk',1024))
    save(out/'lookcloser.png',target['rgb'])
    data=Path(spec['data_template'].format(frame_id=frame))
    sources=Sources(data,record,spec,model,old,checkpoint_sha,out)
    old.native_sample=lambda raw,uv,row:photograph_samples(raw,uv,row,sources.fit)
    rgb,info,labels,weight=old.hybrid_image(out,sources,record['camera'],target,count=spec.get('source_count',16),
        mode='detail',sigma=spec.get('detail_sigma',8.),gain=1.,pixel_scale=request['output_wh'][0]/1920)
    save(out/'hybrid.png',rgb)
    np.savez_compressed(out/'transfer_audit.npz',source_labels=labels,transfer_weight=weight)
    return dict(kind='two_same_instant_neural_render_variants',checkpoint_sha256=checkpoint_sha,
        model_frame_id=frame,hybrid=info,native_sources=sources.inventory,checkpoint_binding=binding,
        source_transforms_sha256=sha(data/'transforms.json'),physical_GPU=gpu_proof)


def render_gt(spec,request,record,out):
    import tifffile
    from marshawn_video_path import raw_file
    name=request['endpoint_train_camera'];frame=int(record['frame_id'])
    path=raw_file(name,frame,spec['raw_roots'])
    if not path:raise FileNotFoundError('Missing actual continuation photograph '+str(frame))
    source=next(r for r in read(request['source_transforms'])['frames'] if r['physical_camera']==name)
    raw=tifffile.memmap(path,mode='r');fit=read(spec['photometry_fit']);x,y,w,h=record['gt_native_crop_xywh']
    # Equivalent to ONE native-resolution lens/radiometry conversion, then a
    # direct integer array crop. Native focal length is never scaled here.
    rgb=np.empty((h,w,3),np.float32)
    for top in range(0,h,64):
        yy,xx=np.mgrid[y+top:y+min(h,top+64),x:x+w]
        pixels=np.stack([xx,yy],-1)
        values,valid=photograph_samples(raw,pixels,native_row(source),fit)
        if not valid.all():raise ValueError('Native GTcrop asks outside calibrated RAW support')
        rgb[top:top+len(values)]=values
    save(out/'lookcloser.png',rgb)
    # Both deliveries use exactly the same actual photographs after t3.
    destination=out/'hybrid.png'
    if destination.exists():destination.unlink()
    os.link(out/'lookcloser.png',destination)
    return dict(kind='actual_gt',source_raw=path,source_raw_sha256=sha(path),integer_crop_xywh=[x,y,w,h],
        source_native_wh=[8192,4320],image_resize=False,source_time_interpolation=False,
        native_lens_correction=True,color='frozen Log3G10/RWG→Rec709linear→gain4+k4+TRAINscalar→Reinhard+sRGB')


def render_one(spec,path,index):
    out=Path(spec['output'])/'frames'/f'{index:04d}';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _render_one_locked(spec,path,index)


def _render_one_locked(spec,path,index):
    request=read(path);record=request['records'][index]
    out=Path(spec['output'])/'frames'/f'{index:04d}';out.mkdir(parents=True,exist_ok=True)
    binding=dict(path_sha256=sha(path),spec_sha256=sha(spec['_spec_path']),record=record)
    receipt=out/'complete.json'
    if receipt.exists():
        done=read(receipt)
        if done['binding']!=binding:raise ValueError('Completed render belongs to another request')
        for name,digest in done['images'].items():
            if sha(out/name)!=digest:raise ValueError('Completed render image changed')
        return
    detail=render_neural(spec,request,record,out) if record['kind']=='neural' else render_gt(spec,request,record,out)
    for name in ['lookcloser.png','hybrid.png']:
        if list(Image.open(out/name).size)!=request['output_wh']:raise ValueError('Output raster mismatch')
    write(receipt,dict(binding=binding,detail=detail,images={n:sha(out/n) for n in ['lookcloser.png','hybrid.png']},time=time.time()))


def encode(spec,path,neural_only=False):
    request=read(path);root=Path(spec['output']);receipts=[]
    count=72 if neural_only else 120
    for i in range(count):
        receipt=read(root/'frames'/f'{i:04d}'/'complete.json')
        if receipt['binding']['path_sha256']!=sha(path):raise ValueError('Mixed path campaign')
        for name,digest in receipt['images'].items():
            if sha(root/'frames'/f'{i:04d}'/name)!=digest:raise ValueError('Frame hash changed')
        receipts.append(receipt)
    for variant in ['lookcloser','hybrid']:
        suffix='neural3s' if neural_only else 'native_GT2s'
        output=root/f'Marshawn1660_3s_24fps_{variant}_{suffix}.mp4'
        temp=output.with_suffix('.tmp.mp4')
        subprocess.run([spec.get('ffmpeg','ffmpeg'),'-hide_banner','-loglevel','error','-y','-framerate','24','-start_number','0',
            '-i',str(root/'frames'/'%04d'/f'{variant}.png'),
            '-frames:v',str(count),'-r','24','-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True)
        temp.replace(output)
    write(root/('neural_3s_complete.json' if neural_only else 'complete.json'),dict(path_sha256=sha(path),frame_count=count,fps=24,seconds=count/24,
        variants=['lookcloser','hybrid'],GT_last48_identical=not neural_only,visual_review_required=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--path',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--encode',action='store_true')
    p.add_argument('--encode-neural',action='store_true')
    a=p.parse_args();spec=read(a.spec);spec['_spec_path']=str(a.spec)
    request=read(a.path)
    if sha(request['source_transforms'])!=request['source_transforms_sha256']:
        raise ValueError('Path source calibration changed; regenerate/review path before dispatch')
    if sha(spec['photometry_fit'])!=spec['photometry_fit_sha256']:
        raise ValueError('Frozen photometry fit changed')
    if spec.get('native_vendor_pq'):
        import prepare_marshawn_color_variants
        prepare_marshawn_color_variants.PQ=Path(spec['native_vendor_pq'])
    if spec.get('native_rig'):
        import marshawn_actor_coordinates
        marshawn_actor_coordinates.RIG=Path(spec['native_rig'])
    if a.encode:encode(spec,a.path)
    elif a.encode_neural:encode(spec,a.path,True)
    elif a.index is not None and 0<=a.index<120:render_one(spec,a.path,a.index)
    else:p.error('Select --index0..119 or --encode')


if __name__=='__main__':main()
