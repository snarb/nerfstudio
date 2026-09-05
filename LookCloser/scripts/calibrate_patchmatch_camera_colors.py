#!/usr/bin/env python3
"""Fit scalar exposure and diagonal RGB corrections from train-only 3D overlaps.

The reference scale is the geometric mean of the train-camera responses, never
the held-out image. Spatially held-out surface samples audit the fit. JPEG ingest
gains are recorded separately to distinguish processing from pre-ingest response.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_patchmatch_camera_path import normalize_frame
from render_mesh_image_blend import load_depth,load_rgb,grid_sample
from patchmatch_color_calibration import decode_exposed_linear,encode_exposed_linear,solve_relative_gains,fit_spatial_exposure

HELD_OUT={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}


def sample_correspondences(data,mesh,metadata,depth_root,samples):
    payload=json.loads((data/'transforms.json').read_text())
    train=set(payload['train_filenames'])
    frames=[normalize_frame(f,payload,metadata) for f in payload['frames'] if f['file_path'] in train]
    if len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62:
        raise ValueError('Expected all 62 distinct fixed train cameras')
    if any(f.get('mask_path') for f in frames) or HELD_OUT&{f['physical_camera'] for f in frames}:
        raise ValueError('Masks or held-out RGB in calibration inputs')
    vertices=np.asarray(o3d.io.read_triangle_mesh(str(mesh)).vertices)
    rng=np.random.default_rng(0);vertices=vertices[rng.choice(len(vertices),min(samples,len(vertices)),replace=False)]
    points=torch.tensor(vertices,device='cuda',dtype=torch.float32)
    colors=[];vis=[];centers=[];uvs=[]
    with torch.inference_mode():
        for f in frames:
            pose=torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float32)
            q=(points-pose[:3,3])@pose[:3,:3]
            z=-q[:,2];u=f['fl_x']*q[:,0]/z+f['cx'];v=-f['fl_y']*q[:,1]/z+f['cy']
            rgb=load_rgb(data/f['file_path'],torch.device('cuda'))
            depth=torch.tensor(load_depth(depth_root/(Path(f['file_path']).stem+'.npy.gz')),device='cuda')*metadata['dataparser_scale']
            sampled=grid_sample(rgb,u[None],v[None])[:,0].T
            observed=grid_sample(depth[None],u[None],v[None])[0,0]
            valid=(z>0)&(u>=2)&(u<f['w']-3)&(v>=2)&(v<f['h']-3)&(observed>0)
            valid&=torch.abs(torch.log(z.clamp_min(1e-6)/observed.clamp_min(1e-6)))<.001
            valid&=(sampled>.1).all(-1)&(sampled<.9).all(-1)
            colors.append(sampled.cpu().numpy());vis.append(valid.cpu().numpy());centers.append(pose[:3,3].cpu().numpy())
            uvs.append(torch.stack((u,v),-1).cpu().numpy())
    return frames,np.asarray(colors),np.asarray(vis),np.asarray(centers),vertices,np.asarray(uvs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--mesh-metadata',type=Path,required=True);p.add_argument('--mesh-depth',type=Path,required=True)
    p.add_argument('--conversion-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=60000)
    p.add_argument('--spatial-grid',type=int,nargs=2,default=None)
    p.add_argument('--spatial-smoothness',type=float,default=10.)
    p.add_argument('--spatial-max-multiplier',type=float,default=1.25)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; preserve earlier fits')
    metadata=json.loads(a.mesh_metadata.read_text())
    depth_manifest_path=a.mesh_depth/'mesh_depth_manifest.json'
    depth_manifest=json.loads(depth_manifest_path.read_text())
    if depth_manifest['mesh_sha256']!=sha256(a.mesh) or depth_manifest.get('masks') is not False:
        raise ValueError('Raycasts must be from the exact unmasked calibration mesh')
    if not np.isclose(depth_manifest['dataparser_scale'],metadata['dataparser_scale'],rtol=1e-7):
        raise ValueError('Raycast and mesh normalization scales differ')
    if not np.allclose(depth_manifest['dataparser_transform'],metadata['dataparser_transform'],atol=1e-6):
        raise ValueError('Raycast and mesh normalization transforms differ')
    conversion=json.loads(a.conversion_manifest.read_text())
    if conversion['tone_map']['curve']!='global_exposure_then_reinhard_then_srgb':
        raise ValueError('Unknown ingest curve; cannot calibrate linear response')
    frames,rgb,valid,centers,vertices,uv=sample_correspondences(a.data,a.mesh,metadata,a.mesh_depth,a.samples)
    # Spatial blocks, not random individual pixels, are excluded from fitting.
    blocks=np.floor(vertices/.008).astype(np.int64)
    held=((blocks[:,0]*73856093)^(blocks[:,1]*19349663)^(blocks[:,2]*83492791))%5==0
    distance=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    near=np.argsort(distance,axis=1)[:,1:9]
    pairs=sorted({tuple(sorted((i,int(j)))) for i,js in enumerate(near) for j in js})
    exposed=decode_exposed_linear(rgb);log_rgb=np.log(exposed.clip(1e-7))
    lum=exposed@np.array([.2126,.7152,.0722]);log_lum=np.log(lum.clip(1e-7))
    kept=[];rgb_delta=[];scalar_delta=[];weights=[];rows=[]
    for i,j in pairs:
        overlap=valid[i]&valid[j];fit=overlap&~held;check=overlap&held
        if fit.sum()<200 or check.sum()<50:continue
        d=log_rgb[j,fit]-log_rgb[i,fit];s=log_lum[j,fit]-log_lum[i,fit]
        delta=np.median(d,axis=0)
        kept.append((i,j));rgb_delta.append(delta);scalar_delta.append(np.median(s))
        noise=max(float(np.median(np.abs(d-delta))),.05)
        weights.append(np.sqrt(min(int(fit.sum()),4000))/noise)
        rows.append({'i':frames[i]['physical_camera'],'j':frames[j]['physical_camera'],
                     'fit_samples':int(fit.sum()),'validation_samples':int(check.sum()),'log_rgb_j_over_i':delta.tolist()})
    rgb_gains=solve_relative_gains(len(frames),kept,rgb_delta,weights)
    scalar_gains=solve_relative_gains(len(frames),kept,scalar_delta,weights).repeat(3,axis=1)
    by_stem={Path(row['output']).stem:row for row in conversion['images']}
    ingest=np.array([by_stem[Path(f['file_path']).stem]['exposure_gain'] for f in frames])
    ingest_gains=(np.exp(np.log(ingest).mean())/ingest)[:,None].repeat(3,axis=1)
    models={'none':np.ones_like(rgb_gains),'ingest':ingest_gains,'exposure':scalar_gains,'rgb':rgb_gains}
    spatial_grid=spatial_values=spatial_stats=None
    if a.spatial_grid is not None:
        spatial_grid,spatial_values,spatial_stats=fit_spatial_exposure(frames,uv,valid,held,log_lum,scalar_gains,kept,*a.spatial_grid,
                smoothness_weight=a.spatial_smoothness,max_multiplier=a.spatial_max_multiplier)
        models['spatial']=scalar_gains[:,None,:]*np.exp(spatial_values[:,:,None])
    residuals={}
    for name,gains in models.items():
        corrected=encode_exposed_linear(exposed*(gains[:,None,:] if gains.ndim==2 else gains))
        values=[]
        for row,(i,j) in zip(rows,kept):
            check=valid[i]&valid[j]&held
            errors=np.abs(corrected[i,check]-corrected[j,check]).mean(-1)
            row[f'{name}_validation_display_l1_median']=float(np.median(errors));values.extend(errors.tolist())
        residuals[name]={'median':float(np.median(values)),'p90':float(np.quantile(values,.9)),'count':len(values)}
    raw_gain=rgb_gains*ingest[:,None];raw_gain/=np.exp(np.log(raw_gain).mean(0))
    camera_rows={}
    for i,f in enumerate(frames):
        path=(a.data/f['file_path']).resolve(strict=True)
        expected=by_stem[path.stem]['sha256']
        if sha256(path)!=expected:raise ValueError('Train JPEG differs from its conversion receipt')
        chroma=rgb_gains[i]/np.exp(np.log(rgb_gains[i]).mean())
        camera_rows[f['physical_camera']]={'image_sha256':expected,'image':str(path),'ingest_gain':float(ingest[i]),
                  'ingest_gain_correction':ingest_gains[i].tolist(),
                  'exposure_gain':scalar_gains[i].tolist(),'rgb_gain':rgb_gains[i].tolist(),'chromatic_gain':chroma.tolist(),
                  'pre_ingest_relative_correction':raw_gain[i].tolist()}
        if spatial_grid is not None:camera_rows[f['physical_camera']]['spatial_log_gain_grid']=spatial_grid[i].tolist()
    output={'schema_version':1,'method':'train_only_geometric_overlap_radiometric_calibration',
            'domain':'inverse_srgb_inverse_reinhard_exposed_linear','gauge':'geometric_mean_train_gain_one',
            'uses_eval_rgb':False,'uses_semantic_masks':False,'source_averaging':False,
            'identifiability_note':'Pre-ingest response combines camera exposure, white balance, processing, sensitivity and view-dependent lighting; hardware sensitivity is not identifiable from these observations alone.',
            'data_sha256':sha256(a.data/'transforms.json'),'mesh_sha256':sha256(a.mesh),
            'conversion_manifest_sha256':sha256(a.conversion_manifest),'script_sha256':sha256(Path(__file__)),
            'mesh_depth_manifest_sha256':sha256(depth_manifest_path),
            'helper_sha256':sha256(Path(__file__).with_name('patchmatch_color_calibration.py')),
            'spatial_holdout':{'block_size_normalized':.008,'modulus':5,'held_samples':int(held.sum()),'total_samples':len(held)},
            'validation_display_pair_l1':residuals,'cameras':camera_rows,'pairs':rows}
    output['spatial_fit']=spatial_stats
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,output)
    print(json.dumps({'validation_display_pair_l1':residuals,'rgb_gain_min':rgb_gains.min(0).tolist(),'rgb_gain_max':rgb_gains.max(0).tolist()}))


if __name__=='__main__':main()
