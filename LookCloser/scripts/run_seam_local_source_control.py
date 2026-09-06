#!/usr/bin/env python3
"""Local depth-connected seam blend of verified train-source warps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from audit_source_detail_control import decode_eight_source_labels
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from seam_local_source_blending import blend_seam_sources


def main():
    from render_mesh_image_blend import load_depth, fill_small_consistent_depth_holes, write_exr_image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame-id',required=True)
    p.add_argument('--radius',type=int,default=32)
    p.add_argument('--visibility-feather',type=float,default=8.)
    p.add_argument('--base-smoothness',type=float,default=64.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing diagnostic output')
    if not a.frame_id.isdigit() or len(a.frame_id)!=6:p.error('Need six-digit frame identity')
    if (not np.isfinite([a.visibility_feather,a.base_smoothness]).all()
            or min(a.visibility_feather,a.base_smoothness)<=0 or not 1<=a.radius<=1024):
        p.error('Need positive finite scales and integer radius in 1..1024')
    req_path=a.frozen_control/'request.json';man_path=a.frozen_control/'control_manifest.json'
    req=json.loads(req_path.read_text());man=json.loads(man_path.read_text())
    if man['state']!='complete' or man['request_sha256']!=sha256(req_path):raise ValueError('Invalid frozen control')
    data_hashes={k:v for k,v in req['input_hashes'].items() if Path(k).suffix!='.py'}
    for name,digest in data_hashes.items():
        if sha256(Path(name))!=digest:raise ValueError('Changed frozen input: '+name)
    for name in ['baseline_pred_0000.png','source_selection.png']:
        if sha256(a.frozen_control/name)!=man['outputs'][name]:raise ValueError('Changed frozen '+name)
    audits=[Path(k) for k in data_hashes if Path(k).name=='reprojection_audit.json']
    if len(audits)!=1:raise ValueError('Ambiguous source-warp audit')
    audit=json.loads(audits[0].read_text());render=audits[0].parent
    if (audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False
            or audit['eval_rgb_use']!='not_read' or not audit['exact_mesh_visibility'] or audit['pixel_center_offset']!=.5):
        raise ValueError('Need native/exact train-only unmasked source warps')
    if (req['train_camera_count']!=62 or req['source_camera_count']!=8 or len(set(req['sources']))!=8
            or set(req['sources'])&{'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}):
        raise ValueError('Invalid frozen train-source inventory')
    dm_path=Path(audit['mesh_depth_manifest']);dm=json.loads(dm_path.read_text())
    if sha256(dm_path)!=audit['mesh_depth_manifest_sha256'] or sha256(Path(dm['mesh']))!=dm['mesh_sha256']:
        raise ValueError('Changed mesh/depth manifest')
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    if str(depth_path) not in data_hashes:raise ValueError('Unbound target depth')
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    rgb_paths=[render/'source_warps'/f'source_{i:02d}.png' for i in range(8)]
    valid_paths=[render/'source_warps'/f'valid_{i:02d}.png' for i in range(8)]
    if any(str(x) not in data_hashes for x in [*rgb_paths,*valid_paths]):raise ValueError('Unbound source warps')
    rgb=np.stack([np.asarray(Image.open(x).convert('RGB')) for x in rgb_paths])
    valid=np.stack([np.asarray(Image.open(x))>0 for x in valid_paths])
    selection=np.asarray(Image.open(a.frozen_control/'source_selection.png').convert('RGB'))
    labels=decode_eight_source_labels(selection);yy,xx=np.indices(labels.shape)
    baseline=rgb[np.maximum(labels,0),yy,xx].copy();baseline[labels<0]=0
    if not np.array_equal(baseline,np.asarray(Image.open(a.frozen_control/'baseline_pred_0000.png').convert('RGB'))):
        raise ValueError('Frozen source-warp replay differs from paired baseline')
    scripts=[Path(__file__),Path(__file__).with_name('seam_local_source_blending.py'),Path(__file__).with_name('visible_source_blending.py'),
        Path(__file__).with_name('surface_color_field.py'),Path(__file__).with_name('render_mesh_image_blend.py'),
        Path(__file__).with_name('audit_source_detail_control.py'),Path(__file__).with_name('colmap_patchmatch_tsdf_campaign_common.py')]
    request=dict(frame_id=a.frame_id,source_rgb_averaging=True,uses_eval_rgb=False,uses_semantic_masks=False,
        train_camera_count=62,source_camera_count=8,sources=req['sources'],geometry_unchanged=True,visibility_unchanged=True,
        radius=a.radius,visibility_feather=a.visibility_feather,base_smoothness=a.base_smoothness,
        historical_scripts='Bound by frozen request hash; not re-executed to produce cached warps',
        input_hashes={**data_hashes,**{str(x):sha256(x) for x in [req_path,man_path,*scripts]}})
    a.output.mkdir(parents=True);atomic_json(a.output/'request.json',request)
    Image.fromarray(baseline).save(a.output/'baseline_pred_0000.png');Image.fromarray(selection).save(a.output/'source_selection.png')
    with torch.inference_mode():
        warped=torch.as_tensor(rgb,device='cuda',dtype=torch.float32).permute(0,3,1,2)/255
        print('Local full-RGB and low-band seam blending started',flush=True)
        outputs,weights,band,stats=blend_seam_sources(list(warped),torch.as_tensor(valid,device='cuda'),
            torch.as_tensor(labels,device='cuda'),torch.as_tensor(depth,device='cuda'),
            radius=a.radius,visibility_feather=a.visibility_feather,base_smoothness=a.base_smoothness)
        for name,result in outputs.items():
            if not bool(torch.isfinite(result).all()):raise ValueError('Non-finite blended render')
            out=a.output/name;out.mkdir();hwc=result.permute(1,2,0).float().cpu()
            Image.fromarray(np.rint(hwc.numpy()*255).astype(np.uint8)).save(out/'eval_pred_0000.png')
            write_exr_image(out/'eval_pred_0000.exr',hwc)
        np.savez_compressed(a.output/'source_weights.npz',weights=weights.cpu().numpy())
        Image.fromarray(band.cpu().numpy().astype(np.uint8)*255).save(a.output/'seam_band.png')
        atomic_json(a.output/'stats.json',stats)
    hashes={str(x.relative_to(a.output)):sha256(x) for x in a.output.rglob('*') if x.is_file() and x.name!='request.json'}
    atomic_json(a.output/'control_manifest.json',dict(state='complete',request_sha256=sha256(a.output/'request.json'),outputs=hashes))
    print('Local seam blends complete; mixed pixels='+str(stats['mixed_pixels']),flush=True)


if __name__=='__main__':main()
