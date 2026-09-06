#!/usr/bin/env python3
"""Matched hard-label control for single-camera seam-gradient color correction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_source_detail_control import decode_eight_source_labels
from hard_source_gradient_leveling import level_source_gradients


def main():
    from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['render','warp-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection-variant',choices=['nearest_fill8','seam_cut8'],required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous outputs')
    saved=json.loads(a.warp_audit.read_text());hashes=dict(saved['input_hashes'])
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed audited input: {file}')
    audit_path=a.render/'reprojection_audit.json'
    if str(audit_path) not in hashes:raise ValueError('Warp audit belongs to another render')
    audit=json.loads(audit_path.read_text())
    if audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False:
        raise ValueError('Need train-only unmasked source warps')
    data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    train={str(Path(f).resolve()) for f in data['train_filenames']}
    sources=[Path(s['source_image']).resolve() for s in audit['sources']]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if (len(sources)!=8 or len(set(sources))!=8 or any(str(s) not in train for s in sources)
            or forbidden&{frames[str(s)]['physical_camera'] for s in sources}):raise ValueError('Need eight permitted train cameras')
    dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    rgb=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'source_{i:02d}.png').convert('RGB')) for i in range(8)])
    valid=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'valid_{i:02d}.png'))>0 for i in range(8)])
    selection_path=a.render/a.selection_variant/'source_selection.png'
    selection=np.asarray(Image.open(selection_path).convert('RGB'));labels=decode_eight_source_labels(selection)
    yy,xx=np.indices(labels.shape)
    if not valid[np.maximum(labels,0),yy,xx][labels>=0].all():raise ValueError('Invisible source selected')
    baseline=rgb[np.maximum(labels,0),yy,xx].copy();baseline[labels<0]=0
    native_path=a.render/a.selection_variant/'eval_pred_0000.png'
    native=np.asarray(Image.open(native_path).convert('RGB'))
    if np.abs(native.astype(np.int16)-baseline.astype(np.int16)).max()>1:raise ValueError('Native/quantized control differs by more than one RGB8 level')
    print('inputs_verified; solver_started',flush=True)
    with torch.inference_mode():
        warped=torch.as_tensor(rgb,device='cuda',dtype=torch.float32).permute(0,3,1,2)/255
        prediction=torch.as_tensor(baseline,device='cuda',dtype=torch.float32).permute(2,0,1)/255
        output,offset,stats=level_source_gradients(prediction,torch.as_tensor(labels,device='cuda'),list(warped),
            list(torch.as_tensor(valid,device='cuda')),torch.as_tensor(depth,device='cuda'))
        result=output.permute(1,2,0).cpu().numpy();correction=offset.permute(1,2,0).cpu().numpy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite render')
    a.output.mkdir(parents=True)
    Image.fromarray(baseline).save(a.output/'baseline_pred_0000.png')
    Image.fromarray(np.rint(result*255).astype(np.uint8)).save(a.output/'eval_pred_0000.png')
    (a.output/'source_selection.png').write_bytes(selection_path.read_bytes())
    np.savez_compressed(a.output/'display_offset.npz',offset=correction)
    for path in [a.warp_audit,selection_path,native_path,Path(__file__),Path(__file__).with_name('hard_source_gradient_leveling.py')]:hashes[str(path)]=sha256(path)
    manifest=dict(method=stats['method'],source_labels_unchanged=True,geometry_unchanged=True,visibility_unchanged=True,
        uses_eval_rgb=False,uses_semantic_masks=False,source_rgb_averaging=False,native_reprojection_control=False,
        rgb_domain='additive display RGB correction on identical RGB8 source warps',selection_variant=a.selection_variant,
        sources=[frames[str(s)]['physical_camera'] for s in sources],stats=stats,input_hashes=hashes,
        output_hashes={p.name:sha256(p) for p in a.output.iterdir() if p.is_file()})
    atomic_json(a.output/'control_manifest.json',manifest)
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
