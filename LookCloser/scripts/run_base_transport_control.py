#!/usr/bin/env python3
"""Frozen RGB8 source-warp control for harmonic low-frequency color transport."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from audit_source_detail_control import decode_eight_source_labels
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from hard_source_base_transport import transport_source_bases


def main():
    from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes,write_exr_image
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['render','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--base-smoothness',type=float,nargs='+',default=[16.,64.])
    p.add_argument('--max-color-jump',type=float,default=None)
    p.add_argument('--max-seed-chroma-difference',type=float,default=None)
    p.add_argument('--luminance-only',action='store_true')
    p.add_argument('--bounded-rgb',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing diagnostic output')
    if a.luminance_only and a.bounded_rgb:p.error('Luminance-only and bounded-RGB are separate controls')
    if (not np.isfinite(a.base_smoothness).all() or min(a.base_smoothness)<=0
            or len(set(a.base_smoothness))!=len(a.base_smoothness)):p.error('Need unique finite positive base scales')
    if a.max_color_jump is not None and (not np.isfinite(a.max_color_jump) or a.max_color_jump<=0):p.error('Color-edge gate must be finite and positive')
    if a.max_seed_chroma_difference is not None and (not np.isfinite(a.max_seed_chroma_difference) or a.max_seed_chroma_difference<=0):
        p.error('Seed chromaticity gate must be finite and positive')
    audit_path=a.render/'reprojection_audit.json';audit=json.loads(audit_path.read_text())
    if (audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False
            or audit['eval_rgb_use']!='not_read' or not audit['exact_mesh_visibility'] or audit['pixel_center_offset']!=.5):
        raise ValueError('Need native/exact train-only unmasked source warps')
    transforms=Path(audit['data'])/'transforms.json';data=json.loads(transforms.read_text())
    frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    train={str(Path(f).resolve()) for f in data['train_filenames']}
    sources=[Path(s['source_image']).resolve() for s in audit['sources']]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if (len(train)!=62 or len({frames[s]['physical_camera'] for s in train})!=62
            or forbidden&{frames[s]['physical_camera'] for s in train}
            or len(sources)!=8 or len(set(sources))!=8 or any(str(s) not in train for s in sources)):
        raise ValueError('Require exactly 62 unique train cameras and eight train-only source warps')
    dm_path=Path(audit['mesh_depth_manifest'])
    if sha256(dm_path)!=audit['mesh_depth_manifest_sha256']:raise ValueError('Changed depth manifest')
    dm=json.loads(dm_path.read_text());mesh=Path(dm['mesh'])
    if sha256(mesh)!=dm['mesh_sha256']:raise ValueError('Changed mesh')
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    rgb_paths=[a.render/'source_warps'/f'source_{i:02d}.png' for i in range(8)]
    valid_paths=[a.render/'source_warps'/f'valid_{i:02d}.png' for i in range(8)]
    rgb=np.stack([np.asarray(Image.open(path).convert('RGB')) for path in rgb_paths])
    valid=np.stack([np.asarray(Image.open(path))>0 for path in valid_paths])
    selection_path=a.render/'seam_cut8/source_selection.png';selection=np.asarray(Image.open(selection_path).convert('RGB'))
    labels=decode_eight_source_labels(selection);yy,xx=np.indices(labels.shape)
    if not valid[np.maximum(labels,0),yy,xx][labels>=0].all():raise ValueError('Invisible source selected')
    baseline=rgb[np.maximum(labels,0),yy,xx].copy();baseline[labels<0]=0
    native_path=a.render/'seam_cut8/eval_pred_0000.png';native=np.asarray(Image.open(native_path).convert('RGB'))
    max_delta=int(np.abs(native.astype(np.int16)-baseline.astype(np.int16)).max())
    if max_delta>1:raise ValueError('Quantized source control is not within one RGB8 level of the native render')
    hashes={str(path):sha256(path) for path in [audit_path,transforms,dm_path,mesh,depth_path,*sources,*rgb_paths,*valid_paths,
        selection_path,native_path,Path(__file__),Path(__file__).with_name('hard_source_base_transport.py'),
        Path(__file__).with_name('surface_color_field.py'),Path(__file__).with_name('render_mesh_image_blend.py')]}
    request=dict(base_smoothness=a.base_smoothness,uses_eval_rgb=False,uses_semantic_masks=False,
        train_camera_count=62,source_camera_count=8,sources=[frames[str(s)]['physical_camera'] for s in sources],
        native_max_rgb8_difference=max_delta,source_rgb_averaging=False,
        source_labels_unchanged=True,geometry_unchanged=True,visibility_unchanged=True,input_hashes=hashes)
    if a.max_color_jump is not None:request['max_color_jump']=a.max_color_jump
    if a.max_seed_chroma_difference is not None:request['max_seed_chroma_difference']=a.max_seed_chroma_difference
    if a.luminance_only:request['luminance_only']=True
    if a.bounded_rgb:request['bounded_rgb']=True
    a.output.mkdir(parents=True);atomic_json(a.output/'request.json',request)
    Image.fromarray(baseline).save(a.output/'baseline_pred_0000.png');Image.fromarray(selection).save(a.output/'source_selection.png')
    outputs={}
    with torch.inference_mode():
        warped=torch.as_tensor(rgb,device='cuda',dtype=torch.float32).permute(0,3,1,2)/255
        prediction=torch.as_tensor(baseline,device='cuda',dtype=torch.float32).permute(2,0,1)/255
        labels_t=torch.as_tensor(labels,device='cuda');valid_t=torch.as_tensor(valid,device='cuda');depth_t=torch.as_tensor(depth,device='cuda')
        for scale in a.base_smoothness:
            name=f'base{scale:g}';out=a.output/name;out.mkdir()
            print(name+' started',flush=True)
            corrected,offset,stats=transport_source_bases(prediction,labels_t,list(warped),list(valid_t),depth_t,
                base_smoothness=scale,max_color_jump=a.max_color_jump,max_seed_chroma_difference=a.max_seed_chroma_difference,
                luminance_only=a.luminance_only,bounded_rgb=a.bounded_rgb)
            if not bool(torch.isfinite(corrected).all()):raise ValueError('Non-finite corrected render')
            assert torch.equal(corrected[:,labels_t<=0],prediction[:,labels_t<=0])
            result=corrected.permute(1,2,0).cpu().numpy()
            Image.fromarray(np.rint(result*255).astype(np.uint8)).save(out/'eval_pred_0000.png')
            write_exr_image(out/'eval_pred_0000.exr',corrected.permute(1,2,0).float().cpu())
            np.savez_compressed(out/'display_offset.npz',offset=offset.permute(1,2,0).cpu().numpy())
            atomic_json(out/'stats.json',stats)
            outputs[name]={path.name:sha256(path) for path in out.iterdir() if path.is_file()}
            print(name+' complete; max_offset='+str(stats['correction_abs_max']),flush=True)
    outputs['baseline_pred_0000.png']=sha256(a.output/'baseline_pred_0000.png')
    outputs['source_selection.png']=sha256(a.output/'source_selection.png')
    atomic_json(a.output/'control_manifest.json',dict(state='complete',request_sha256=sha256(a.output/'request.json'),outputs=outputs))


if __name__=='__main__':main()
