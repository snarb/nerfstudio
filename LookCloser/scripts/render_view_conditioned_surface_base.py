#!/usr/bin/env python3
"""Apply a train-validated directional base to an audited frozen-label control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image
from canonical_surface_base import CanonicalSurfaceBase
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from view_conditioned_surface_base import direction_features
from render_canonical_surface_base_control import unproject_mesh_depth,apply_selected_offsets
from audit_source_detail_control import decode_eight_source_labels


def evaluate_base(coefficients, world, camera_center, degree):
    if (coefficients.shape != (*world.shape[:-1], [1,4,9][degree], 3)
            or world.shape[-1] != 3 or np.asarray(camera_center).shape != (3,)
            or not np.isfinite(coefficients).all() or not np.isfinite(world).all()
            or not np.isfinite(camera_center).all()):
        raise ValueError('Finite directional coefficient/point/center inventory mismatch')
    feature=direction_features(torch.as_tensor(np.asarray(camera_center)-world),degree).numpy()
    return np.einsum('...k,...kc->...c',feature,coefficients).astype(np.float32)


def validate_directional_manifest(manifest, source_base_path, mesh_path):
    if (manifest['uses_eval_rgb'] is not False or manifest['uses_semantic_masks'] is not False
            or manifest['query_dependent_base'] is not True or manifest['fit']['converged'] is not True
            or manifest['degree'] not in (1,2) or manifest['mesh_sha256'] != sha256(mesh_path)
            or manifest['source_base_manifest_sha256'] != sha256(source_base_path)):
        raise ValueError('Directional model provenance or solver mismatch')


def main():
    from render_mesh_image_blend import load_depth,load_rgb,fill_small_consistent_depth_holes,write_prediction
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','directional-base-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous renders; output must not exist')
    old=json.loads((a.control/'control_manifest.json').read_text())
    hashes={}
    def verify(path,expected=None):
        path=Path(path).resolve();actual=sha256(path)
        if expected is not None and actual!=expected:raise ValueError('Changed input: '+str(path))
        hashes[str(path)]=actual
    for path,digest in old['input_hashes'].items():verify(path,digest)
    for path,digest in old['output_hashes'].items():verify(a.control/path,digest)
    verify(a.control/'control_manifest.json')
    if not old['original_png_byte_identical'] or not old['labels_unchanged'] or old['uses_eval_rgb'] or old['uses_semantic_masks']:
        raise ValueError('Need validated train-only frozen-label control')
    audit_paths=[Path(f) for f in old['input_hashes'] if Path(f).name=='reprojection_audit.json']
    if len(audit_paths)!=1:raise ValueError('Ambiguous original render')
    audit=json.loads(audit_paths[0].read_text());dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    target=next(f for f in data['frames'] if str((Path(audit['data'])/f['file_path']).resolve())==audit['target_image'])
    mesh_path=Path(dm['mesh']);color_path=Path(audit['camera_color_calibration']['path'])
    manifest=json.loads(a.directional_base_manifest.read_text());model_root=a.directional_base_manifest.parent
    base_path=Path(manifest['source_base_manifest']);base=json.loads(base_path.read_text())
    validate_directional_manifest(manifest,base_path,mesh_path)
    if manifest['source_cameras']!=base['source_cameras']:raise ValueError('Directional source inventory mismatch')
    verify(a.directional_base_manifest);verify(base_path,manifest['source_base_manifest_sha256'])
    verify(model_root/'request.json',manifest['request_sha256'])
    verify(model_root/'selection_before_render.json',manifest['selection_sha256'])
    selection=json.loads((model_root/'selection_before_render.json').read_text())
    if not selection['eligible_for_render'] or selection['selected_degree']!=manifest['degree']:
        raise ValueError('No matching held-train eligibility receipt')
    request=json.loads((model_root/'request.json').read_text())
    for path,digest in request['input_hashes'].items():verify(path,digest)
    coeff_path=model_root/manifest['coefficients'];verify(coeff_path,manifest['coefficients_sha256'])
    with np.load(coeff_path) as z:coefficients=z['coefficients']
    with np.load(base_path.parent/base['coefficients']) as z:common=z['common_base']
    if coefficients.shape!=(len(common),[1,4,9][manifest['degree']],3) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid directional coefficient inventory')
    depth_path=Path(next(row['depth'] for row in dm['images'] if row['image']==audit['target_image']))
    depth=fill_small_consistent_depth_holes(load_depth(depth_path),max_area=1000,boundary_radius=4,max_relative_plane_rmse=.015)[0]
    world=unproject_mesh_depth(depth,target,dm)
    labels=decode_eight_source_labels(np.asarray(Image.open(a.control/'off/source_selection.png')))
    baseline=load_rgb(a.control/'off/eval_pred_0000.exr',torch.device('cpu')).permute(1,2,0).numpy()
    with np.load(a.control/'applied_offsets.npz') as z:offsets=z['offsets'].copy()
    field=CanonicalSurfaceBase(base_path,mesh_path,color_path);field.bind(world,(depth>0)&(labels>=0))
    triangle=field.triangles[field.ids];weights=field.weights
    interpolated=(coefficients[triangle]*weights[:,:,None,None]).sum(1)
    query=evaluate_base(interpolated,world[field.support],np.asarray(target['transform_matrix'])[:3,3],manifest['degree'])
    common_sample=(common[triangle]*weights[:,:,None]).sum(1)
    replacement=query-common_sample;replacement[~field.near]=0
    offsets[field.support]+=replacement
    for name in ('render_view_conditioned_surface_base.py','view_conditioned_surface_base.py','render_canonical_surface_base_control.py'):
        verify(Path(__file__).with_name(name))
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json',dict(input_hashes=hashes,uses_eval_rgb=False,uses_semantic_masks=False,
        degree=manifest['degree'],query_dependent_base=True,geometry_unchanged=True,labels_unchanged=True))
    rows=[]
    for variant,strength in [('off',0.),('directional_base',1.)]:
        pred,stats=apply_selected_offsets(baseline,labels,offsets,strength=strength)
        write_prediction(a.output,variant,torch.from_numpy(pred).permute(2,0,1),None)
        shutil.copyfile(a.control/'off/source_selection.png',a.output/variant/'source_selection.png')
        rows.append(dict(name=variant,**stats))
    if sha256(a.output/'off/eval_pred_0000.png')!=sha256(a.control/'off/eval_pred_0000.png'):
        raise RuntimeError('Original byte-identical off replay failed')
    np.savez_compressed(a.output/'applied_offsets.npz',offsets=offsets)
    result=dict(method='direction_conditioned_mesh_base_plus_hard_source_detail',degree=manifest['degree'],
        uses_eval_rgb=False,uses_semantic_masks=False,query_dependent_base=True,geometry_unchanged=True,
        labels_unchanged=True,visibility_unchanged=True,original_png_byte_identical=True,
        supported_pixels=int(field.support.sum()),far_mesh_pixels=int((~field.near).sum()),
        query_base_min=float(query.min()),query_base_max=float(query.max()),variants=rows,
        input_hashes=hashes,output_hashes={str(f.relative_to(a.output)):sha256(f) for f in sorted(a.output.rglob('*')) if f.is_file()},
        accepted_surface_recipe=False)
    atomic_json(a.output/'control_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','output_hashes')}),flush=True)


if __name__=='__main__':main()
