#!/usr/bin/env python3
"""Test train-camera bandwidth versus native camera depth on disjoint blocks.

The model predicts measured relative projected blur variance, not optical focus
or a physical PSF. Native source depth and the local projection Jacobian avoid
using target-camera depth as a rig-specific lookup. This is an audit only; no
predictions, geometry, source labels, masks or held-out RGB are changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def source_projection_geometry(xy,depth,target,sources,pixel_center_offset=.5):
    """Native source z and isotropic source-to-target variance scale at each point."""
    xy=np.asarray(xy,np.int64)
    pose=np.asarray(target['transform_matrix'],np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2:raise ValueError('Expected Nx2 native raster indices')
    if (xy<1).any() or (xy[:,0]>=depth.shape[1]-1).any() or (xy[:,1]>=depth.shape[0]-1).any():
        raise ValueError('Jacobian samples need valid raster neighbors')
    def world_at(delta):
        x,y=(xy+delta).T;z=depth[y,x]
        q=np.stack(((x+pixel_center_offset-target['cx'])/target['fl_x']*z,
                   -(y+pixel_center_offset-target['cy'])/target['fl_y']*z,-z),-1)
        return q@pose[:3,:3].T+pose[:3,3]
    world=[world_at(np.array(d)) for d in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]]
    depths=[];scales=[];conditions=[]
    for frame in sources:
        c2w=np.asarray(frame['transform_matrix'],np.float64)
        points=[(p-c2w[:3,3])@c2w[:3,:3] for p in world]
        source_z=-points[0][:,2]
        def uv(q):
            z=-q[:,2]
            return np.stack((frame['fl_x']*q[:,0]/z+frame['cx'],
                             -frame['fl_y']*q[:,1]/z+frame['cy']),-1)
        projections=[uv(q) for q in points]
        jacobian=np.stack(((projections[2]-projections[1])*.5,(projections[4]-projections[3])*.5),-1)
        singular=np.linalg.svd(jacobian,compute_uv=False)
        scale=np.mean(1/np.maximum(singular,1e-9)**2,axis=-1)
        depths.append(source_z);scales.append(scale);conditions.append(singular[:,0]/np.maximum(singular[:,1],1e-9))
    return np.stack(depths,-1),np.stack(scales,-1),np.stack(conditions,-1)


def fit_depth_bandwidth(rows,native_depth,variance_scale,*,degree=0,ridge=.01):
    """Block-balanced robust fit; all centering/scaling/fitting uses fit blocks only."""
    if degree not in (0,1,2) or ridge<=0:raise ValueError('Invalid polynomial/ridge')
    count=native_depth.shape[1];n=len(rows)
    if native_depth.shape!=(n,count) or variance_scale.shape!=native_depth.shape:
        raise ValueError('Native geometry dimensions differ from observations')
    if not np.isfinite(native_depth).all() or (native_depth<=0).any() or not np.isfinite(variance_scale).all() or (variance_scale<=0).any():
        raise ValueError('Need positive finite native camera depth and Jacobian scales')
    fit=np.array([not r['held'] for r in rows]);held=~fit
    if not fit.any() or not held.any():raise ValueError('Need disjoint fit and held blocks')
    block_sets=[{tuple(r['block']) for r in rows if bool(r['held'])==v} for v in [False,True]]
    if block_sets[0]&block_sets[1]:raise ValueError('Spatial fit/held blocks overlap')
    inverse=1/native_depth
    center=np.median(inverse[fit],axis=0)
    width=np.maximum(np.quantile(inverse[fit],.9,axis=0)-np.quantile(inverse[fit],.1,axis=0),1e-6)
    coordinates=(inverse-center)/width
    features=np.stack([coordinates**k for k in range(degree+1)],-1)*variance_scale[...,None]
    design=np.zeros((n,count,degree+1),np.float64)
    primary=np.array([r['primary_rank'] for r in rows]);secondary=np.array([r['source_rank'] for r in rows])
    if (primary<0).any() or (secondary>=count).any() or (primary>=secondary).any():raise ValueError('Invalid ordered camera pair')
    index=np.arange(n);design[index,secondary]=features[index,secondary];design[index,primary]=-features[index,primary]
    design=design.reshape(n,-1)
    target=np.array([r['relative_blur_variance'] for r in rows],np.float64)
    if not np.isfinite(target).all():raise ValueError('Nonfinite observed relative bandwidth')
    groups={}
    for k,r in enumerate(rows):groups.setdefault((r['primary_rank'],r['source_rank'],*r['block']),[]).append(k)
    # Many overlapping patches are one pair/block observation, not independent votes.
    blocks=[];x=[];y=[];training=[]
    for key,ids in sorted(groups.items()):
        if len(ids)<3:continue
        blocks.append({'pair':list(key[:2]),'block':list(key[2:]),'patches':len(ids),'held':bool(held[ids[0]])})
        x.append(np.median(design[ids],axis=0));y.append(np.median(target[ids]));training.append(bool(fit[ids[0]]))
    x=np.asarray(x);y=np.asarray(y);training=np.asarray(training)
    if training.sum()<count*(degree+1) or (~training).sum()<3:
        raise ValueError('Insufficient independent camera-pair/block observations')
    a=x[training];b=y[training];weights=np.ones(len(a))
    regularizer=np.eye(a.shape[1])*np.sqrt(ridge)
    for _ in range(8):
        coefficients=np.linalg.lstsq(np.r_[a*np.sqrt(weights[:,None]),regularizer],np.r_[b*np.sqrt(weights),np.zeros(a.shape[1])],rcond=None)[0]
        error=a@coefficients-b
        weights=np.minimum(1,.35/np.maximum(abs(error),1e-9))
    predicted=x@coefficients
    for r,actual,pred in zip(blocks,y,predicted):r.update(observed=float(actual),predicted=float(pred),absolute_error=float(abs(actual-pred)))
    held_errors=abs(predicted[~training]-y[~training])
    return {'degree':degree,'ridge':ridge,'coefficients':coefficients.reshape(count,degree+1).tolist(),
            'inverse_depth_center_fit_only':center.tolist(),'inverse_depth_scale_fit_only':width.tolist(),
            'inverse_depth_bounds_fit_only':[[float(inverse[fit&((primary==c)|(secondary==c)),c].min()),
                                              float(inverse[fit&((primary==c)|(secondary==c)),c].max())] for c in range(count)],
            'fit_pair_blocks':int(training.sum()),'held_pair_blocks':int((~training).sum()),
            'fit_spatial_blocks':len(block_sets[0]),'held_spatial_blocks':len(block_sets[1]),
            'held_absolute_error_median':float(np.median(held_errors)),'held_absolute_error_p90':float(np.quantile(held_errors,.9)),
            'fit_absolute_error_median':float(np.median(abs(predicted[training]-y[training]))),
            'blocks':blocks,'fitted_matrix_rank':int(np.linalg.matrix_rank(a)),'coefficient_count':a.shape[1],
            'identifiability':'Regularized relative bandwidth regression; not a calibrated physical defocus model.'}


def predict_relative_variance(model,native_depth,variance_scale):
    """Bound inverse-depth evaluation to measured fit support; never filter RGB."""
    native_depth=np.asarray(native_depth);variance_scale=np.asarray(variance_scale)
    if (native_depth.shape!=variance_scale.shape or not np.isfinite(native_depth).all()
            or (native_depth<=0).any() or not np.isfinite(variance_scale).all() or (variance_scale<=0).any()):
        raise ValueError('Need matching positive finite native depth and Jacobian scale')
    bounds=np.asarray(model['inverse_depth_bounds_fit_only']);q=1/native_depth
    clipped=(q<bounds[:,0])|(q>bounds[:,1]);q=np.clip(q,bounds[:,0],bounds[:,1])
    q=(q-np.asarray(model['inverse_depth_center_fit_only']))/np.asarray(model['inverse_depth_scale_fit_only'])
    coefficients=np.asarray(model['coefficients'])
    variance=sum(coefficients[:,k]*q**k for k in range(model['degree']+1))*variance_scale
    # Only differences are source costs. These regressions are not positive PSFs.
    return np.clip(variance,-6.25,6.25),clipped


def main():
    from render_mesh_image_blend import load_depth
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous audits')
    observations=json.loads(a.observations.read_text());audit_path=a.render/'reprojection_audit.json'
    if observations['uses_eval_rgb'] is not False or observations['uses_semantic_masks'] is not False:
        raise ValueError('Only train-only, unmasked evidence is permitted')
    hashes=dict(observations['input_hashes'])
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed input: {file}')
    if str(audit_path) not in hashes:raise ValueError('Observation/render provenance mismatch')
    audit=json.loads(audit_path.read_text());data=json.loads((Path(audit['data'])/'transforms.json').read_text())
    frames={str(Path(f['file_path']).resolve()):f for f in data['frames']}
    target=frames[str(Path(audit['target_image']).resolve())]
    source_frames=[frames[str(Path(s['source_image']).resolve())] for s in audit['sources']]
    if [f['physical_camera'] for f in source_frames]!=observations['sources']:raise ValueError('Camera order mismatch')
    dm=json.loads(Path(audit['mesh_depth_manifest']).read_text())
    depth_path=Path(next(r['depth'] for r in dm['images'] if r['image']==audit['target_image']))
    depth=load_depth(depth_path)
    rows=observations['observations'];xy=np.array([[r['x'],r['y']] for r in rows])
    native,scale,condition=source_projection_geometry(xy,depth,target,source_frames,audit['pixel_center_offset'])
    good=np.isfinite(native).all(1)&(native>0).all(1)&np.isfinite(scale).all(1)&(scale>0).all(1)&(condition<10).all(1)
    selected=[r for r,use in zip(rows,good) if use]
    results=[fit_depth_bandwidth(selected,native[good],scale[good],degree=k) for k in [0,1,2]]
    hashes[str(a.observations)]=sha256(a.observations);hashes[str(Path(__file__))]=sha256(Path(__file__))
    atomic_json(a.output,{'uses_eval_rgb':False,'uses_semantic_masks':False,'changes_prediction':False,
        'metric_scope':'train_camera_projected_bandwidth_diagnostic_not_face_quality',
        'observations':len(rows),'valid_jacobian_observations':int(good.sum()),'sources':observations['sources'],
        'source_variance_scale_range':[float(scale[good].min()),float(scale[good].max())],
        'geometry_convention':'Native source camera inverse-z, isotropic source-to-target covariance scale trace(invJ invJ^T)/2',
        'models':results,'input_hashes':hashes,'interpretation':__doc__})
    print(json.dumps([{k:v for k,v in r.items() if k not in ['blocks','coefficients','inverse_depth_center_fit_only','inverse_depth_scale_fit_only']} for r in results]))


if __name__=='__main__':main()
