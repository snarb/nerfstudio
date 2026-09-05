#!/usr/bin/env python3
"""Fixed-label, train-only within-source detail restoration diagnostic.

Uses immutable RGB8 warps, not fresh native reprojection. The fitted native-depth
bandwidth model supplies amounts; no target RGB, semantic mask, source mixture,
geometry change, relabelling, or per-anatomy exception is permitted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_depth_conditioned_bandwidth import source_projection_geometry,predict_relative_variance
from source_detail_restoration import relative_restoration_amount,restore_source_detail


def decode_eight_source_labels(image):
    palette=np.array([[230,25,75],[60,180,75],[255,225,25],[0,130,200],
                      [245,130,48],[145,30,180],[70,240,240],[240,50,230]],np.uint8)
    labels=np.full(image.shape[:2],-1,np.int32)
    for rank,color in enumerate(palette):labels[(image==color).all(-1)]=rank
    if ((labels<0)&image.any(-1)).any():raise ValueError('Unexpected categorical source palette')
    return labels


def held_patch_check(before,after,observations):
    """Fixed original correspondence; block-balanced held scores, no rematching."""
    luma=np.array([.2126,.7152,.0722],np.float32)
    gray=[np.ascontiguousarray(images@luma,dtype=np.float32) for images in (before,after)]
    half=observations['patch_size']//2
    grouped={}
    def ncc(a,b):
        a=a[6:-6,6:-6].astype(float);b=b[6:-6,6:-6].astype(float)
        a-=a.mean();b-=b.mean()
        return float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    for r in observations['observations']:
        if not r['held']:continue
        x,y,i,j=r['x'],r['y'],r['primary_rank'],r['source_rank']
        centers=[(x-.5-r['dx']/2,y-.5-r['dy']/2),(x-.5+r['dx']/2,y-.5+r['dy']/2)]
        values=[]
        for group in gray:
            a=cv2.getRectSubPix(group[i],(half*2,half*2),centers[0])
            b=cv2.getRectSubPix(group[j],(half*2,half*2),centers[1])
            values.append(ncc(a,b))
        grouped.setdefault((i,j,*r['block']),[]).append(values)
    rows=[]
    for key,values in sorted(grouped.items()):
        if len(values)<3:continue
        values=np.asarray(values)
        rows.append(dict(pair=list(key[:2]),block=list(key[2:]),patches=len(values),
                         before=float(np.median(values[:,0])),after=float(np.median(values[:,1])),
                         delta=float(np.median(values[:,1]-values[:,0]))))
    if not rows:raise ValueError('No held pair blocks')
    return dict(held_pair_blocks=len(rows),held_spatial_blocks=len({tuple(r['block']) for r in rows}),
                patches=sum(r['patches'] for r in rows),median_ncc_before=float(np.median([r['before'] for r in rows])),
                median_ncc_after=float(np.median([r['after'] for r in rows])),
                median_block_delta=float(np.median([r['delta'] for r in rows])),
                improved_block_fraction=float(np.mean([r['delta']>0 for r in rows])),blocks=rows,
                registration_reestimated=False,held_rgb_used_for_filter_parameters=False)


def main():
    from render_mesh_image_blend import load_depth,fill_small_consistent_depth_holes
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['render','model-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--selection-variant',choices=['nearest_fill8','seam_cut8'],default='seam_cut8')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing diagnostic')
    model_audit=json.loads(a.model_audit.read_text());hashes=dict(model_audit['input_hashes'])
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed audited input: {file}')
    audit_path=a.render/'reprojection_audit.json'
    if str(audit_path) not in hashes:raise ValueError('Model/render provenance differs')
    audit=json.loads(audit_path.read_text());data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    target=frames[str(Path(audit['target_image']).resolve())]
    sources=[frames[str(Path(s['source_image']).resolve())] for s in audit['sources']]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if (len(sources)!=8 or [s['physical_camera'] for s in sources]!=model_audit['sources']
            or forbidden&set(model_audit['sources']) or audit['uses_eval_rgb_for_prediction'] is not False
            or audit['uses_masks'] is not False):raise ValueError('Need eight matching train-only sources, no masks')
    model=next(m for m in model_audit['models'] if m['degree']==1)
    observations_path=a.render/'local_bandwidth_all_pairs24.json'
    if str(observations_path) not in hashes:raise ValueError('Missing immutable observations')
    observations=json.loads(observations_path.read_text())
    dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    rgb8=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'source_{i:02d}.png').convert('RGB')) for i in range(8)])
    rgb=rgb8.astype(np.float32)/255
    valid=np.stack([np.asarray(Image.open(a.render/'source_warps'/f'valid_{i:02d}.png'))>0 for i in range(8)])
    selection_path=a.render/a.selection_variant/'source_selection.png'
    selection=np.asarray(Image.open(selection_path).convert('RGB'));labels=decode_eight_source_labels(selection)
    hashes[str(selection_path)]=sha256(selection_path)
    yy,xx=np.indices(labels.shape)
    if not valid[np.maximum(labels,0),yy,xx][labels>=0].all():raise ValueError('Existing labels select invisible source')
    baseline=rgb8[np.maximum(labels,0),yy,xx].copy();baseline[labels<0]=0
    native_pred_path=a.render/a.selection_variant/'eval_pred_0000.png'
    native_pred=np.asarray(Image.open(native_pred_path).convert('RGB'))
    delta=np.abs(native_pred.astype(np.int16)-baseline.astype(np.int16))
    if delta.max()>1:raise ValueError('Frozen labels do not reproduce native render within one RGB8 level')
    hashes[str(native_pred_path)]=sha256(native_pred_path)
    support=valid.any(0)&(depth>0);support[[0,-1]]=False;support[:,[0,-1]]=False
    y,x=np.where(support);xy=np.c_[x,y];amount=np.zeros(valid.shape,np.float32)
    for start in range(0,len(xy),8192):
        points=xy[start:start+8192]
        native,scale,condition=source_projection_geometry(points,depth,target,sources,audit['pixel_center_offset'])
        good=(native>0).all(1)&np.isfinite(scale).all(1)&(scale>0).all(1)&(condition<10).all(1)
        variance,clipped=predict_relative_variance(model,native[good],scale[good])
        x,y=points[good].T
        available=valid[:,y,x]&~clipped.T
        amount[:,y,x]=relative_restoration_amount(variance.T,available,model['fit_absolute_error_median'])
    restored=[];rows=[]
    for rank in range(8):
        image,applied,stats=restore_source_detail(rgb[rank],valid[rank],depth,amount[rank])
        restored.append(image);amount[rank]=applied;rows.append(dict(rank=rank,**stats))
    restored=np.stack(restored)
    held=held_patch_check(rgb,restored,observations)
    restored8=np.rint(restored.clip(0,1)*255).astype(np.uint8)
    prediction=restored8[np.maximum(labels,0),yy,xx].copy();prediction[labels<0]=0
    selected_amount=amount[np.maximum(labels,0),yy,xx];selected_amount[labels<0]=0
    a.output.mkdir(parents=True)
    Image.fromarray(baseline).save(a.output/'baseline_pred_0000.png')
    Image.fromarray(prediction).save(a.output/'eval_pred_0000.png')
    # Identical encoded source map, not a newly optimized selection.
    (a.output/'source_selection.png').write_bytes(selection_path.read_bytes())
    np.savez_compressed(a.output/'applied_amount.npz',amount=amount)
    (a.output/'restored_sources').mkdir()
    for rank in range(8):Image.fromarray(restored8[rank]).save(a.output/'restored_sources'/f'source_{rank:02d}.png')
    hashes[str(a.model_audit)]=sha256(a.model_audit)
    for name in ['audit_source_detail_control.py','source_detail_restoration.py','patchmatch_color_calibration.py']:
        path=Path(__file__).with_name(name);hashes[str(path)]=sha256(path)
    result=dict(method='fixed_label_bounded_within_source_detail_control',uses_eval_rgb=False,uses_semantic_masks=False,
        source_rgb_averaging=False,filters_single_source_rgb=True,geometry_unchanged=True,visibility_unchanged=True,
        labels_unchanged=True,native_reprojection_control=False,parameter_selection='fixed max amount .125; threshold twice fit-only median variance error; no held RGB tuning',
        sources=model_audit['sources'],source_stats=rows,held_validation=held,selection_variant=a.selection_variant,
        native_baseline_max_rgb8_difference=int(delta.max()),selected_filtered_pixels=int((selected_amount>0).sum()),
        changed_prediction_pixels=int((prediction!=baseline).any(-1).sum()),input_hashes=hashes,
        output_hashes={str(path.relative_to(a.output)):sha256(path) for path in a.output.rglob('*') if path.is_file()})
    atomic_json(a.output/'control_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','output_hashes','source_stats','held_validation']}|
                     {'held_validation':{k:v for k,v in held.items() if k!='blocks'}}))


if __name__=='__main__':main()
