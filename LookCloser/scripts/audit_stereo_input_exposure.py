#!/usr/bin/env python3
"""Paired train-only NCC proxy for exposure correction before dense stereo.

No depth is recomputed here. Identical mesh-warped 11x11 patches are compared
with ordinary NCC, not asserted to reproduce COLMAP's bilateral CUDA cost.
Geometry-only eligibility is frozen across response variants. This diagnostic
cannot establish a surface/fly-through pass or identify hardware exposure.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256

HELD={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
METHODS=('original','cancel_ingest','scalar_fit')


def paired_ncc(reference,source,minimum_variance=1e-5):
    """Keep low-variance pairs as -1, never silently change the sample pool."""
    a=np.asarray(reference,np.float64);b=np.asarray(source,np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]<2 or not np.isfinite(np.r_[a,b]).all():
        raise ValueError('Need finite equal NxP patch arrays')
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    va=(a*a).mean(1);vb=(b*b).mean(1)
    usable=(va>=minimum_variance)&(vb>=minimum_variance)
    correlation=(a*b).mean(1)/np.sqrt(np.maximum(va*vb,1e-30))
    return np.where(usable,np.clip(correlation,-1,1),-1),usable


def effective_gains(ingest,corrections):
    ingest=np.asarray(ingest,np.float64);corrections=np.asarray(corrections,np.float64)
    if ingest.ndim!=1 or ingest.shape!=corrections.shape or not len(ingest):
        raise ValueError('Need matching nonempty train-camera gains')
    if not np.isfinite(np.r_[ingest,corrections]).all() or (ingest<=0).any() or (corrections<=0).any():
        raise ValueError('Gains must be finite and positive')
    return dict(original=ingest,cancel_ingest=np.full_like(ingest,np.exp(np.log(ingest).mean())),scalar_fit=ingest*corrections)


def main():
    import cv2
    import open3d as o3d
    from nerfstudio.data.utils.data_utils import load_exr_image
    from convert_exr_nerfstudio_to_jpeg import tone_map
    from render_patchmatch_camera_path import normalize_frame
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','conversion-manifest','mesh','mesh-metadata','color-calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--grid-step',type=int,default=32)
    p.add_argument('--neighbors',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing audit; no marker-only resume')
    if a.grid_step<12 or not 1<=a.neighbors<=12:p.error('Invalid bounded sample settings')
    payload=json.loads((a.data/'transforms.json').read_text());train=set(payload['train_filenames'])
    raw=[f for f in payload['frames'] if f['file_path'] in train]
    names=[f['physical_camera'] for f in raw]
    if len(names)!=len(set(names)) or len(names)!=62 or set(names)&HELD:
        raise ValueError('Need 62 unique train cameras without held RGB')
    if any(f.get('mask_path') for f in payload['frames']):raise ValueError('Semantic masks forbidden')
    meta=json.loads(a.mesh_metadata.read_text());color=json.loads(a.color_calibration.read_text())
    if meta['output_sha256']!=sha256(a.mesh) or color['mesh_sha256']!=sha256(a.mesh) or color['uses_eval_rgb']:
        raise ValueError('Mismatched mesh/color provenance or held RGB use')
    conversion=json.loads(a.conversion_manifest.read_text())
    if conversion['tone_map']['exposure_mode']!='per-image' or conversion['tone_map']['jpeg_quality']!=98:
        raise ValueError('Requires original per-image JPEG98 conversion')
    by_stem={Path(r['output']).stem:r for r in conversion['images']}
    ingest=np.array([by_stem[Path(f['file_path']).stem]['exposure_gain'] for f in raw])
    corrections=[]
    for name in names:
        gain=np.asarray(color['cameras'][name]['exposure_gain'])
        if gain.shape!=(3,) or not np.all(gain==gain[0]):raise ValueError('Need scalar, not RGB, fitted correction')
        corrections.append(gain[0])
    gains=effective_gains(ingest,corrections)
    files=[a.data/'transforms.json',a.conversion_manifest,a.mesh,a.mesh_metadata,a.color_calibration,Path(__file__),
        Path(__file__).with_name('convert_exr_nerfstudio_to_jpeg.py'),Path(__file__).with_name('render_patchmatch_camera_path.py')]
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json',dict(method='fixed_mesh_warped_train_patch_input_exposure_proxy',
        uses_eval_rgb=False,uses_semantic_masks=False,changes_geometry=False,changes_prediction=False,
        ordinary_NCC_not_exact_COLMAP_bilateral_cost=True,grid_step=a.grid_step,neighbors=a.neighbors,
        patch_radius=5,minimum_variance=1e-5,geometric_max_log_depth_range=.0075,
        thresholds=dict(minimum_pair_median_ncc_improvement=.005,minimum_improved_pair_fraction=.60),
        upstream_reference='https://github.com/colmap/colmap/blob/5509fffe/src/colmap/mvs/patch_match_cuda.cu',
        input_hashes={str(path.resolve()):sha256(path) for path in files},camera_order=names,
        exposure_gains={k:v.tolist() for k,v in gains.items()}))
    images={method:[] for method in METHODS};image_rows=[]
    for i,frame in enumerate(raw):
        receipt=by_stem[Path(frame['file_path']).stem];path=a.data/frame['file_path'];source=Path(receipt['input'])
        if sha256(path)!=receipt['sha256'] or sha256(source)!=receipt['source_sha256']:
            raise ValueError('Changed source/ingest RGB')
        exr=load_exr_image(source)
        row=dict(physical_camera=names[i],source=str(source),source_sha256=sha256(source),original=str(path),original_sha256=sha256(path),outputs={})
        for method in METHODS:
            stream=io.BytesIO();Image.fromarray(tone_map(exr,float(gains[method][i]))).save(stream,format='JPEG',quality=98,subsampling=0,optimize=True)
            content=stream.getvalue();digest=hashlib.sha256(content).hexdigest()
            if method=='original':
                if digest!=receipt['sha256']:raise ValueError('Original EXR-to-JPEG replay is not byte identical')
            else:
                target=a.output/method/frame['file_path'];target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content)
                row['outputs'][method]=dict(path=str(target),sha256=digest,gain=float(gains[method][i]))
            images[method].append(np.asarray(Image.open(io.BytesIO(content)).convert('L'),np.float32)/255)
        image_rows.append(row)
        atomic_json(a.output/'ingest_state.json',dict(completed=len(image_rows),images=image_rows))
        if (i+1)%10==0 or i==61:print(f'ingest={i+1}/62 original_replay_exact',flush=True)
    frames=[normalize_frame(f,payload,meta) for f in raw]
    centers=np.array([np.asarray(f['transform_matrix'])[:3,3] for f in frames])
    neighbors=np.argsort(np.linalg.norm(centers[:,None]-centers[None],axis=-1),axis=1)[:,1:a.neighbors+1]
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    offsets=np.stack(np.meshgrid(np.arange(-5,6),np.arange(-5,6)),axis=-1).reshape(-1,2)
    rows=[];observations=[]
    for i,frame in enumerate(frames):
        yy,xx=np.mgrid[8:frame['h']-8:a.grid_step,8:frame['w']-8:a.grid_step]
        xy=np.c_[xx.ravel(),yy.ravel()];pixels=xy[:,None]+offsets[None]
        q=np.stack(((pixels[...,0]+.5-frame['cx'])/frame['fl_x'],-(pixels[...,1]+.5-frame['cy'])/frame['fl_y'],-np.ones(pixels.shape[:2])),-1)
        rotation=np.asarray(frame['transform_matrix'])[:3,:3];directions=q@rotation.T
        rays=np.concatenate((np.broadcast_to(centers[i],directions.shape),directions),-1).astype(np.float32)
        z=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        good=np.isfinite(z).all(1)&(z>0).all(1)
        good&=np.ptp(np.log(np.maximum(np.where(np.isfinite(z),z,1),1e-9)),axis=1)<.0075
        xy=xy[good];pixels=pixels[good];z=z[good];world=centers[i]+directions[good]*z[...,None]
        ref={method:images[method][i][pixels[...,1],pixels[...,0]] for method in METHODS}
        for j in neighbors[i]:
            source_frame=frames[j];pose=np.asarray(source_frame['transform_matrix']);local=(world-centers[j])@pose[:3,:3]
            depth=-local[...,2];u=source_frame['fl_x']*local[...,0]/depth+source_frame['cx']-.5
            v=-source_frame['fl_y']*local[...,1]/depth+source_frame['cy']-.5
            in_bounds=(depth>0)&(u>=1)&(u<source_frame['w']-2)&(v>=1)&(v<source_frame['h']-2)
            target_rays=np.concatenate((np.broadcast_to(centers[j],world.shape),world-centers[j]),-1).astype(np.float32)
            hit=scene.cast_rays(o3d.core.Tensor(target_rays))['t_hit'].numpy()
            seen=in_bounds&np.isfinite(hit)&(np.abs(hit-1)*np.linalg.norm(world-centers[j],axis=-1)<=.000025)
            keep=seen.all(1)
            if not keep.any():continue
            values=[];validity=[]
            for method in METHODS:
                warped=cv2.remap(images[method][j],u[keep].astype(np.float32),v[keep].astype(np.float32),cv2.INTER_LINEAR)
                ncc,usable=paired_ncc(ref[method][keep],warped);values.append(ncc);validity.append(usable)
            values=np.stack(values,1);validity=np.stack(validity,1)
            observations.append(np.c_[np.full(keep.sum(),i),np.full(keep.sum(),j),xy[keep],values,validity.astype(int)])
            rows.append(dict(reference=names[i],source=names[j],patches=int(keep.sum()),
                ncc_median={m:float(np.median(values[:,k])) for k,m in enumerate(METHODS)},
                ncc_p10={m:float(np.quantile(values[:,k],.1)) for k,m in enumerate(METHODS)},
                usable_fraction={m:float(validity[:,k].mean()) for k,m in enumerate(METHODS)}))
        print(f'patch_reference={i+1}/62 pairs={len(rows)}',flush=True)
    if not observations:raise RuntimeError('No common geometric patch inventory')
    all_rows=np.concatenate(observations);np.savez_compressed(a.output/'observations.npz',rows=all_rows)
    summaries=[]
    for method in METHODS:
        changes=np.array([r['ncc_median'][method]-r['ncc_median']['original'] for r in rows])
        eligible=method!='original' and np.median(changes)>=.005 and (changes>0).mean()>=.60
        summaries.append(dict(method=method,pair_median_ncc=float(np.median([r['ncc_median'][method] for r in rows])),
            median_paired_improvement=float(np.median(changes)),improved_pair_fraction=float((changes>0).mean()),
            eligible_for_dense_control=bool(eligible)))
    atomic_json(a.output/'findings.json',dict(status='paired_input_response_proxy_complete',uses_eval_rgb=False,uses_semantic_masks=False,
        accepted_surface_recipe=False,changes_geometry=False,changes_prediction=False,request_sha256=sha256(a.output/'request.json'),
        observations_sha256=sha256(a.output/'observations.npz'),ingest_state_sha256=sha256(a.output/'ingest_state.json'),
        patch_pairs=len(all_rows),camera_pairs=len(rows),identical_geometry_inventory=True,original_62_jpeg_replays_byte_identical=True,
        summaries=summaries,pairs=rows,columns=['reference_index','source_index','x','y',*[m+'_ncc' for m in METHODS],*[m+'_usable' for m in METHODS]]))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
