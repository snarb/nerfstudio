#!/usr/bin/env python3
"""Compare radiometric fits on identical held train-surface samples, without eval RGB."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_patchmatch_camera_colors import sample_correspondences
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from patchmatch_color_calibration import decode_exposed_linear,encode_exposed_linear,grid_basis


def held_pair_residuals(corrected,valid,held,pairs,physical_cameras):
    """Score only the predeclared common held correspondences, never fit pixels."""
    if corrected.shape[:2]!=valid.shape or corrected.shape[-1]!=3 or held.shape!=(valid.shape[1],):
        raise ValueError('Invalid shared correspondence shapes')
    errors=[];rows=[]
    for i,j in pairs:
        check=valid[i]&valid[j]&held
        e=np.abs(corrected[i,check]-corrected[j,check]).mean(-1)
        if not len(e) or not np.isfinite(e).all():raise ValueError('Empty or nonfinite held comparison')
        errors.extend(e.tolist())
        rows.append({'camera_i':physical_cameras[i],'camera_j':physical_cameras[j],
                     'count':len(e),'display_l1_median':float(np.median(e))})
    if not errors:raise ValueError('No common held camera pairs')
    return {'count':len(errors),'display_l1_median':float(np.median(errors)),
            'display_l1_p90':float(np.quantile(errors,.9)),'pairs':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','mesh','mesh-metadata','mesh-depth','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration',type=Path,action='append',required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Audit output already exists')
    metadata=json.loads(a.mesh_metadata.read_text())
    if metadata['output_sha256']!=sha256(a.mesh):raise ValueError('Mesh receipt hash mismatch')
    dm=json.loads((a.mesh_depth/'mesh_depth_manifest.json').read_text())
    if dm['mesh_sha256']!=sha256(a.mesh) or dm.get('masks') is not False:
        raise ValueError('Wrong or masked raycasts')
    if not np.isclose(dm['dataparser_scale'],metadata['dataparser_scale'],rtol=1e-7):
        raise ValueError('Raycast scale mismatch')
    frames,rgb,valid,centers,vertices,uv,_=sample_correspondences(
        a.data,a.mesh,metadata,a.mesh_depth,60000,pixel_center_offset=.5,exact_visibility=True)
    blocks=np.floor(vertices/.008).astype(np.int64)
    held=((blocks[:,0]*73856093)^(blocks[:,1]*19349663)^(blocks[:,2]*83492791))%5==0
    distance=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
    near=np.argsort(distance,axis=1)[:,1:9]
    pairs=sorted({tuple(sorted((i,int(j)))) for i,js in enumerate(near) for j in js})
    pairs=[(i,j) for i,j in pairs if (valid[i]&valid[j]&~held).sum()>=200 and (valid[i]&valid[j]&held).sum()>=50]
    exposed=decode_exposed_linear(rgb);results=[]
    for path in a.calibration:
        calibration=json.loads(path.read_text())
        if calibration.get('uses_eval_rgb') is not False or calibration.get('uses_semantic_masks') is not False:
            raise ValueError('Only train-only unmasked fits may be compared')
        corrected=[]
        for i,frame in enumerate(frames):
            camera=calibration['cameras'][frame['physical_camera']]
            if camera['image_sha256']!=sha256(a.data/frame['file_path']):
                raise ValueError('Fit input RGB identity differs from audited RGB')
            grid=np.asarray(camera['spatial_log_gain_grid'])
            ids,weights=grid_basis(uv[i],frame['w'],frame['h'],grid.shape[1],grid.shape[0])
            log_gain=(grid.ravel()[ids]*weights).sum(-1)
            corrected.append(encode_exposed_linear(exposed[i]*np.asarray(camera['exposure_gain'])[None]*np.exp(log_gain[:,None])))
        stats=held_pair_residuals(np.asarray(corrected),valid,held,pairs,[f['physical_camera'] for f in frames])
        results.append({'calibration':str(path),'calibration_sha256':sha256(path),**stats})
    result={'uses_eval_rgb':False,'uses_semantic_masks':False,'metric_scope':'train_overlap_color_consistency_not_face_quality',
            'identical_samples':True,'pixel_center_offset':.5,'exact_mesh_visibility':True,
            'held_surface_samples':int(held.sum()),'surface_samples':len(held),
            'surface_samples_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),
            'validity_sha256':hashlib.sha256(valid.tobytes()).hexdigest(),
            'mesh_sha256':sha256(a.mesh),'mesh_metadata_sha256':sha256(a.mesh_metadata),
            'depth_manifest_sha256':sha256(a.mesh_depth/'mesh_depth_manifest.json'),
            'data_sha256':sha256(a.data/'transforms.json'),'script_sha256':sha256(Path(__file__)),
            'sampler_sha256':sha256(Path(__file__).with_name('calibrate_patchmatch_camera_colors.py')),
            'results':results}
    atomic_json(a.output,result)
    print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in results]),flush=True)


if __name__=='__main__':main()
