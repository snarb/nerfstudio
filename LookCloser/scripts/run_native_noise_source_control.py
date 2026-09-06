#!/usr/bin/env python3
"""Native train-image filtering with frozen mesh, UVs, visibility and hard labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from audit_source_detail_control import decode_eight_source_labels
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from native_source_noise_filter import high_frequency_floor,filter_native_rgb


def main():
    from render_mesh_image_blend import (NerfstudioDataParserConfig,camera_parameters,load_depth,
        fill_small_consistent_depth_holes,load_rgb,project_target_to_source,grid_sample,write_exr_image)
    from patchmatch_color_calibration import apply_camera_gain
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['frozen-control','output','native-cache']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--frame-id',required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing control output')
    if len(a.frame_id)!=6 or not a.frame_id.isdigit():p.error('Need six-digit frame identity')
    old_req_path=a.frozen_control/'request.json';old_man_path=a.frozen_control/'control_manifest.json'
    old=json.loads(old_req_path.read_text());man=json.loads(old_man_path.read_text())
    if man['state']!='complete' or man['request_sha256']!=sha256(old_req_path):raise ValueError('Invalid frozen input')
    hashes={k:v for k,v in old['input_hashes'].items() if Path(k).suffix!='.py'}
    for path,digest in hashes.items():
        if sha256(Path(path))!=digest:raise ValueError('Changed frozen input: '+path)
    for name in ['baseline_pred_0000.png','source_selection.png']:
        path=a.frozen_control/name
        if sha256(path)!=man['outputs'][name]:raise ValueError('Changed frozen source control')
        hashes[str(path)]=sha256(path)
    audit_path=next(Path(k) for k in hashes if Path(k).name=='reprojection_audit.json')
    audit=json.loads(audit_path.read_text());data_path=Path(audit['data']);payload=json.loads((data_path/'transforms.json').read_text())
    if (audit['uses_eval_rgb_for_prediction'] is not False or audit['eval_rgb_use']!='not_read' or audit['uses_masks']
            or not audit['exact_mesh_visibility'] or audit['pixel_center_offset']!=.5 or audit['dataparser_scale']!=1.
            or audit['source_rgb_depth_aware_sampling'] or audit['surface_texture_registration']
            or audit['angular_surface_color'] or audit['mesh_camera_color'] or audit['surface_color_field']):
        raise ValueError('Need original normalized native/exact source-warp control')
    calibration_path=Path(audit['camera_color_calibration']['path']);calibration=json.loads(calibration_path.read_text())
    if sha256(calibration_path)!=audit['camera_color_calibration']['sha256'] or audit['camera_color_calibration']['model']!='spatial':
        raise ValueError('Changed/unsupported frozen camera response')
    frames={str((data_path/f['file_path']).resolve()):f for f in payload['frames']}
    source_paths=[Path(s['source_image']).resolve() for s in audit['sources']]
    train_paths={str((data_path/f).resolve()) for f in payload['train_filenames']}
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if (len(train_paths)!=62 or len({frames[q]['physical_camera'] for q in train_paths})!=62
            or forbidden&{frames[q]['physical_camera'] for q in train_paths}
            or len(source_paths)!=8 or any(str(q) not in train_paths for q in source_paths)):
        raise ValueError('Need 62 unique train cameras and eight train-only texture sources')
    config=NerfstudioDataParserConfig(data=data_path,eval_mode='filename',orientation_method='none',center_method='none',
        auto_scale_poses=False,scale_factor=1.,downscale_factor=1,load_3D_points=False)
    train=config.setup().get_dataparser_outputs(split='train');target=config.setup().get_dataparser_outputs(split='val')
    if len(target.image_filenames)!=1 or train.mask_filenames is not None or target.mask_filenames is not None:
        raise ValueError('Unexpected eval inventory or semantic masks')
    if train.dataparser_scale!=1 or target.dataparser_scale!=1:raise ValueError('Unexpected normalization')
    indices={Path(q).resolve():i for i,q in enumerate(train.image_filenames)}
    dm_path=Path(audit['mesh_depth_manifest']);dm=json.loads(dm_path.read_text())
    if sha256(dm_path)!=audit['mesh_depth_manifest_sha256'] or sha256(Path(dm['mesh']))!=dm['mesh_sha256']:
        raise ValueError('Changed mesh/depth manifest')
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    selection=np.asarray(Image.open(a.frozen_control/'source_selection.png').convert('RGB'))
    labels=decode_eight_source_labels(selection);baseline=np.asarray(Image.open(a.frozen_control/'baseline_pred_0000.png').convert('RGB'))
    valid=np.stack([np.asarray(Image.open(audit_path.parent/'source_warps'/f'valid_{i:02d}.png'))>0 for i in range(8)])
    yy,xx=np.indices(labels.shape)
    if not valid[np.maximum(labels,0),yy,xx][labels>=0].all():raise ValueError('Invisible frozen source')
    scripts=[Path(__file__),Path(__file__).with_name('native_source_noise_filter.py'),Path(__file__).with_name('render_mesh_image_blend.py'),
        Path(__file__).with_name('patchmatch_color_calibration.py'),Path(__file__).with_name('audit_source_detail_control.py'),
        Path(__file__).resolve().parents[2]/'nerfstudio/data/dataparsers/nerfstudio_dataparser.py']
    hashes.update({str(q):sha256(q) for q in [old_req_path,old_man_path,calibration_path,*scripts]})
    a.output.mkdir(parents=True);a.native_cache.mkdir(parents=True,exist_ok=True)
    request=dict(frame_id=a.frame_id,train_camera_count=62,source_camera_count=8,sources=old['sources'],input_hashes=hashes,
        strengths=[1.,2.],source_rgb_averaging=False,filters_native_train_rgb=True,hard_labels_unchanged=True,
        geometry_unchanged=True,visibility_unchanged=True,texture_uvs_unchanged=True,uses_eval_rgb=False,uses_semantic_masks=False,
        opencv_version=cv2.__version__,torch_version=torch.__version__,filter_stage='Native RGB8 before frozen camera-response correction',
        historical_warps='Input data hashes verified; old scripts not re-executed; a fresh unfiltered native replay is checked')
    atomic_json(a.output/'request.json',request)
    Image.fromarray(baseline).save(a.output/'baseline_pred_0000.png');Image.fromarray(selection).save(a.output/'source_selection.png')
    for variant in ['native_off','nlm1','nlm2']:(a.output/variant/'source_warps').mkdir(parents=True)
    all_warps={k:[] for k in ['native_off','nlm1','nlm2']};sources=[]
    device=torch.device('cuda');target_c2w,target_k=camera_parameters(target.cameras,0,device);target_k['pixel_center_offset']=.5
    depth_t=torch.from_numpy(depth).to(device)
    with torch.inference_mode():
        for rank,path in enumerate(source_paths):
            physical=frames[str(path)]['physical_camera'];row=calibration['cameras'][physical]
            if sha256(path)!=row['image_sha256']:raise ValueError('Changed calibrated native image')
            image=np.asarray(Image.open(path).convert('RGB'));key=physical+'_'+sha256(path)[:12]
            cache=a.native_cache/key;cache_manifest=cache/'manifest.json'
            helper_hash=sha256(Path(__file__).with_name('native_source_noise_filter.py'))
            if cache.exists():
                cm=json.loads(cache_manifest.read_text())
                if cm['source_sha256']!=sha256(path) or cm['helper_sha256']!=helper_hash:raise ValueError('Stale native filter cache')
                for q,h in cm['outputs'].items():
                    if sha256(cache/q)!=h:raise ValueError('Changed native filter cache')
            else:
                cache.mkdir();floor=high_frequency_floor(image);stats={}
                for strength in [1.,2.]:
                    filtered,st=filter_native_rgb(image,strength,floor=floor);name=f'nlm{strength:g}'
                    Image.fromarray(filtered).save(cache/(name+'.png'));stats[name]=st
                cm=dict(state='complete',source=str(path),source_sha256=sha256(path),helper_sha256=helper_hash,
                    physical_camera=physical,high_frequency_floor=floor,filters=stats,
                    outputs={q.name:sha256(q) for q in cache.iterdir() if q.is_file()})
                atomic_json(cache_manifest,cm)
            source_c2w,source_k=camera_parameters(train.cameras,indices[path],device);source_k['pixel_center_offset']=.5
            u,v,z=project_target_to_source(depth_t,target_c2w,target_k,source_c2w,source_k)
            for variant,native_path in [('native_off',path),('nlm1',cache/'nlm1.png'),('nlm2',cache/'nlm2.png')]:
                corrected=apply_camera_gain(load_rgb(native_path,device),row['exposure_gain'],row['spatial_log_gain_grid'])
                warped=grid_sample(corrected,u,v).clamp(0,1)
                quantized=np.rint(warped.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                Image.fromarray(quantized).save(a.output/variant/'source_warps'/f'source_{rank:02d}.png')
                all_warps[variant].append(quantized)
            old_warp=np.asarray(Image.open(audit_path.parent/'source_warps'/f'source_{rank:02d}.png').convert('RGB'))
            delta=np.abs(all_warps['native_off'][-1].astype(np.int16)-old_warp.astype(np.int16))
            if delta[valid[rank]].max()>1:raise ValueError('Fresh calibrated UV/RGB replay differs from frozen warp')
            sources.append(dict(rank=rank,physical_camera=physical,native_cache_manifest=str(cache_manifest),
                native_cache_manifest_sha256=sha256(cache_manifest),native_replay_max_valid_rgb8_error=int(delta[valid[rank]].max()),
                high_frequency_floor={k:v for k,v in cm['high_frequency_floor'].items() if k!='tiles'},filters=cm['filters']))
            print(physical+': high-frequency floor='+str(cm['high_frequency_floor']['sigma_rgb8']),flush=True)
        for variant,warps in all_warps.items():
            result=np.stack(warps)[np.maximum(labels,0),yy,xx].copy();result[labels<0]=0
            if variant=='native_off' and np.abs(result.astype(np.int16)-baseline.astype(np.int16)).max()>1:
                raise ValueError('Unfiltered selected-source replay failed')
            Image.fromarray(result).save(a.output/variant/'eval_pred_0000.png')
            write_exr_image(a.output/variant/'eval_pred_0000.exr',torch.from_numpy(result.astype(np.float32)/255))
    atomic_json(a.output/'source_filter_audit.json',dict(sources=sources,uses_eval_rgb=False,uses_semantic_masks=False,
        no_physical_noise_or_psf_claim=True,selection_unchanged=True,uvs_unchanged=True))
    outputs={str(q.relative_to(a.output)):sha256(q) for q in a.output.rglob('*') if q.is_file() and q.name!='request.json'}
    atomic_json(a.output/'control_manifest.json',dict(state='complete',request_sha256=sha256(a.output/'request.json'),outputs=outputs))
    print('Native-image filter controls complete',flush=True)


if __name__=='__main__':main()
