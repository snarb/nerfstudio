#!/usr/bin/env python3
"""Paired diagnostic hard selections from immutable quantized train-source warps.

This does not replace native camera reprojection. Both controls use the SAME
8-bit audited warps, visibility and graph-cut code; only the fitted depth order
differs. RGB comes from one existing source pixel, without filtering or blending.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_depth_conditioned_bandwidth import source_projection_geometry,predict_relative_variance
from hard_texture_seam_cut import optimize_source_labels


def save_control_images(output,prediction,labels,count):
    import torch
    from render_mesh_image_blend import write_source_selection
    output.mkdir(parents=True)
    Image.fromarray(prediction).save(output/'eval_pred_0000.png')
    write_source_selection(output,'',torch.as_tensor(labels),count)


def main():
    from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',type=Path,required=True);p.add_argument('--model-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--degree',type=int,choices=[0,1],required=True)
    p.add_argument('--penalty',type=float,default=.01);p.add_argument('--rank-penalty',type=float,default=.001)
    p.add_argument('--seam-leveling',action='store_true',help='Matched existing one-sided gain control; not RGB averaging')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous controls')
    if not np.isfinite(a.penalty) or a.penalty<=0 or not np.isfinite(a.rank_penalty) or a.rank_penalty<0:
        p.error('Require finite nonnegative penalties and a positive bandwidth weight')
    model_audit=json.loads(a.model_audit.read_text())
    hashes=dict(model_audit['input_hashes'])
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed audited input: {file}')
    audit_path=a.render/'reprojection_audit.json'
    if str(audit_path) not in hashes:raise ValueError('Model and render provenance differ')
    model=next(m for m in model_audit['models'] if m['degree']==a.degree)
    audit=json.loads(audit_path.read_text());data_path=Path(audit['data'])/'transforms.json'
    data=json.loads(data_path.read_text());frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    target=frames[str(Path(audit['target_image']).resolve())]
    sources=[frames[str(Path(s['source_image']).resolve())] for s in audit['sources']]
    if [f['physical_camera'] for f in sources]!=model_audit['sources']:raise ValueError('Source order differs')
    if audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False:raise ValueError('Need train-only unmasked warps')
    dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    rgb=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'source_{i:02d}.png').convert('RGB')) for i in range(len(sources))])
    valid=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'valid_{i:02d}.png'))>0 for i in range(len(sources))])
    support=valid.any(0)&(depth>0);support[[0,-1]]=False;support[:,[0,-1]]=False
    yy,xx=np.where(support);xy=np.c_[xx,yy];costs=np.zeros(valid.shape,np.float32)
    clipped_total=np.zeros(len(sources),np.int64);good_count=0
    for start in range(0,len(xy),8192):
        points=xy[start:start+8192]
        native,scale,condition=source_projection_geometry(points,depth,target,sources,audit['pixel_center_offset'])
        good=(native>0).all(1)&np.isfinite(scale).all(1)&(scale>0).all(1)&(condition<10).all(1)
        values,clipped=predict_relative_variance(model,native[good],scale[good])
        values-=values.min(1,keepdims=True)
        x,y=points[good].T;costs[:,y,x]=values.T*a.penalty
        clipped_total+=clipped.sum(0);good_count+=int(good.sum())
    labels,optimization=optimize_source_labels(rgb.astype(np.float32)/255,valid,rank_penalty=a.rank_penalty,source_costs=costs)
    y,x=np.indices(labels.shape);prediction=rgb[np.maximum(labels,0),y,x].copy();prediction[labels<0]=0
    assert valid[np.maximum(labels,0),y,x][labels>=0].all()
    leveling=None
    if a.seam_leveling:
        import torch
        from hard_source_seam_leveling import level_hard_source_seams
        warped=torch.as_tensor(rgb,device='cuda',dtype=torch.float32).permute(0,3,1,2)/255
        image=torch.as_tensor(prediction,device='cuda',dtype=torch.float32).permute(2,0,1)/255
        image,_,leveling=level_hard_source_seams(image,torch.as_tensor(labels,device='cuda'),
            list(warped),list(torch.as_tensor(valid,device='cuda')),torch.as_tensor(depth,device='cuda'))
        prediction=np.rint(image.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    save_control_images(a.output,prediction,labels,len(sources))
    pred_path=a.output/'eval_pred_0000.png'
    hashes[str(a.model_audit)]=sha256(a.model_audit)
    for name in ['render_bandwidth_model_control.py','audit_depth_conditioned_bandwidth.py','hard_texture_seam_cut.py']:
        path=Path(__file__).with_name(name);hashes[str(path)]=sha256(path)
    if a.seam_leveling:
        for name in ['hard_source_seam_leveling.py','surface_color_field.py','patchmatch_color_calibration.py']:
            path=Path(__file__).with_name(name);hashes[str(path)]=sha256(path)
    result={'method':'paired_hard_selection_from_identical_quantized_train_warps','degree':a.degree,
        'penalty':a.penalty,'rank_penalty':a.rank_penalty,'source_rgb_averaging':False,'uses_eval_rgb':False,
        'uses_semantic_masks':False,'native_reprojection_control':False,'rgb_domain':'display RGB8 source warps',
        'geometry_unchanged':True,'visibility_unchanged':True,
        'source_identity':'one source plus one-sided harmonic gain' if leveling else 'exact single-source RGB8 index lookup',
        'seam_leveling':leveling,
        'sources':model_audit['sources'],'model_geometry_pixels':good_count,
        'source_inverse_depth_clipped_pixels':clipped_total.tolist(),'optimization':optimization,
        'source_selected_pixels':[int((labels==i).sum()) for i in range(len(sources))],
        'input_hashes':hashes,'render_sha256':sha256(pred_path),'selection_sha256':sha256(a.output/'source_selection.png')}
    atomic_json(a.output/'control_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','optimization']}))


if __name__=='__main__':main()
