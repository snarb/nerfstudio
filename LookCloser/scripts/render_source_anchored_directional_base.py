#!/usr/bin/env python3
"""Relative directional-base correction with exact source-view identity.

Use B(query)-B(source) from the SAME directional model, not an independently
smoothed source base. This isolates residual model error from angular transfer.
No fit is changed; the original float RGB and categorical source map are fixed.
"""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from canonical_surface_base import CanonicalSurfaceBase,HELD
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_view_conditioned_surface_base import evaluate_base,validate_directional_manifest
from render_canonical_surface_base_control import unproject_mesh_depth,apply_selected_offsets
from audit_source_detail_control import decode_eight_source_labels


def relative_base_offset(coefficients,world,query_center,source_center,degree):
    return evaluate_base(coefficients,world,query_center,degree)-evaluate_base(coefficients,world,source_center,degree)


def main():
    from render_mesh_image_blend import load_depth,load_rgb,fill_small_consistent_depth_holes,write_prediction
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['control','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous control')
    previous=json.loads((a.control/'control_manifest.json').read_text());hashes={}
    def verify(path,digest=None):
        path=Path(path).resolve();actual=sha256(path)
        if digest is not None and actual!=digest:raise ValueError('Changed input: '+str(path))
        hashes[str(path)]=actual
    for path,digest in previous['input_hashes'].items():verify(path,digest)
    for path,digest in previous['output_hashes'].items():verify(a.control/path,digest)
    verify(a.control/'control_manifest.json')
    if not previous['original_png_byte_identical'] or previous['uses_eval_rgb'] or previous['uses_semantic_masks']:
        raise ValueError('Need original audited train-only control')
    def unique(name):
        paths=[Path(f) for f in previous['input_hashes'] if Path(f).name==name]
        if len(paths)!=1:raise ValueError('Ambiguous input '+name)
        return paths[0]
    model_path=unique('directional_base_manifest.json');model=json.loads(model_path.read_text())
    base_path=Path(model['source_base_manifest'])
    audit=json.loads(unique('reprojection_audit.json').read_text());dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={str((Path(audit['data'])/f['file_path']).resolve()):f for f in data['frames']}
    target=frames[audit['target_image']];sources=[frames[r['source_image']] for r in audit['sources']]
    if len(sources)!=8 or any(f['physical_camera'] in HELD for f in sources):raise ValueError('Invalid train sources')
    validate_directional_manifest(model,base_path,Path(dm['mesh']))
    coefficients_path=model_path.parent/model['coefficients'];verify(coefficients_path,model['coefficients_sha256'])
    with np.load(coefficients_path) as z:coefficients=z['coefficients']
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    world=unproject_mesh_depth(depth,target,dm)
    labels=decode_eight_source_labels(np.asarray(Image.open(a.control/'off/source_selection.png')))
    baseline=load_rgb(a.control/'off/eval_pred_0000.exr',torch.device('cpu')).permute(1,2,0).numpy()
    field=CanonicalSurfaceBase(base_path,Path(dm['mesh']),Path(audit['camera_color_calibration']['path']))
    field.bind(world,(depth>0)&(labels>=0));triangle=field.triangles[field.ids]
    interpolated=(coefficients[triangle]*field.weights[:,:,None,None]).sum(1)
    valid_labels=labels[field.support];points=world[field.support];offset=np.zeros((len(points),3),np.float32)
    query_center=np.asarray(target['transform_matrix'])[:3,3]
    for rank,source in enumerate(sources):
        selected=(valid_labels==rank)&field.near
        if selected.any():
            offset[selected]=relative_base_offset(interpolated[selected],points[selected],query_center,
                np.asarray(source['transform_matrix'])[:3,3],model['degree'])
    offsets=np.zeros_like(baseline);offsets[field.support]=offset
    for name in ['render_source_anchored_directional_base.py','render_view_conditioned_surface_base.py']:
        verify(Path(__file__).with_name(name))
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json',dict(input_hashes=hashes,uses_eval_rgb=False,uses_semantic_masks=False,
        query_source_same_model=True,fit_unchanged=True,degree=model['degree']))
    rows=[]
    for variant,strength in [('off',0.),('source_anchored',1.)]:
        prediction,stats=apply_selected_offsets(baseline,labels,offsets,strength=strength)
        write_prediction(a.output,variant,torch.from_numpy(prediction).permute(2,0,1),None)
        shutil.copyfile(a.control/'off/source_selection.png',a.output/variant/'source_selection.png')
        rows.append(dict(name=variant,**stats))
    if sha256(a.output/'off/eval_pred_0000.png')!=sha256(a.control/'off/eval_pred_0000.png'):
        raise RuntimeError('Original off replay differs')
    np.savez_compressed(a.output/'applied_offsets.npz',offsets=offsets)
    result=dict(method='source_anchored_directional_display_base_transfer',degree=model['degree'],
        uses_eval_rgb=False,uses_semantic_masks=False,geometry_unchanged=True,visibility_unchanged=True,labels_unchanged=True,
        original_png_byte_identical=True,source_view_identity_exact=True,fit_unchanged=True,variants=rows,
        supported_pixels=int(field.support.sum()),far_mesh_pixels=int((~field.near).sum()),
        input_hashes=hashes,output_hashes={str(f.relative_to(a.output)):sha256(f) for f in sorted(a.output.rglob('*')) if f.is_file()},
        accepted_surface_recipe=False)
    atomic_json(a.output/'control_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','output_hashes']}),flush=True)


if __name__=='__main__':main()
