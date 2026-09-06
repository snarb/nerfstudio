#!/usr/bin/env python3
"""Separate spatial and motion-dependent epipolar residuals on held times.

The spatial term is a low-order image-coordinate residual field, NOT a valid
replacement camera calibration. The motion coefficient is NOT timestamp truth.
All models are diagnostic and never affect source frames or predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256


def block_weights(groups):
    groups=[tuple(g) for g in groups];counts={g:groups.count(g) for g in set(groups)}
    return np.array([1/counts[g] for g in groups],np.float64)


def fit_residual_model(xy,speed,residual,groups,*,spatial=False,timing=False,ridge=.01):
    xy=np.asarray(xy,np.float64);speed=np.asarray(speed,np.float64);residual=np.asarray(residual,np.float64)
    if (xy.ndim!=2 or xy.shape[1]!=2 or speed.shape!=(len(xy),) or residual.shape!=speed.shape
            or len(groups)!=len(xy) or len(xy)<10 or not np.isfinite(xy).all()
            or not np.isfinite(speed).all() or not np.isfinite(residual).all() or ridge<=0):
        raise ValueError('Invalid finite fit-only correspondence data')
    base=block_weights(groups)
    center=np.average(xy,weights=base,axis=0)
    scale=np.maximum(np.sqrt(np.average((xy-center)**2,weights=base,axis=0)),1.)
    columns=[np.ones(len(xy))] if spatial else []
    if spatial:columns.extend(((xy-center)/scale).T)
    if timing:columns.append(speed)
    if not columns:raise ValueError('Select spatial and/or timing model')
    design=np.stack(columns,-1);weights=base.copy();coefficients=np.zeros(design.shape[1])
    # Fixed regularization applies to coordinate slopes, not the intercept/time.
    penalty=np.full(design.shape[1],ridge)
    if spatial:penalty[0]=1e-10
    if timing:penalty[-1]=1e-10
    for _ in range(12):
        matrix=design.T@(design*weights[:,None])+np.diag(penalty)
        rhs=design.T@(weights*residual)
        coefficients=np.linalg.solve(matrix,rhs)
        error=design@coefficients-residual
        weights=base*np.minimum(1.,.5/np.maximum(abs(error),1e-12))
    independent_motion=None
    if timing and spatial:
        space=design[:,:-1]
        projected=space@np.linalg.lstsq(space*np.sqrt(base[:,None]),speed*np.sqrt(base),rcond=None)[0]
        independent_motion=float(np.sum(base*(speed-projected)**2)/max(np.sum(base*speed**2),1e-12))
    return dict(spatial=spatial,timing=timing,center_fit_only=center.tolist(),scale_fit_only=scale.tolist(),
        coefficients=coefficients.tolist(),ridge=ridge,fit_points=len(xy),fit_blocks=len(set(map(tuple,groups))),
        motion_coefficient_available_frames=float(coefficients[-1]) if timing else None,
        suggested_zeroing_time_shift_available_frames=float(-coefficients[-1]) if timing else None,
        motion_energy_not_explained_by_spatial_design=independent_motion)


def predict_residual(model,xy,speed):
    xy=np.asarray(xy,float);speed=np.asarray(speed,float)
    columns=[np.ones(len(xy))] if model['spatial'] else []
    if model['spatial']:columns.extend(((xy-model['center_fit_only'])/model['scale_fit_only']).T)
    if model['timing']:columns.append(speed)
    return np.stack(columns,-1)@np.asarray(model['coefficients'])


def held_block_summary(error,groups):
    grouped={}
    for value,group in zip(error,groups):grouped.setdefault(tuple(group),[]).append(abs(float(value)))
    rows=[dict(group=list(group),points=len(values),median_absolute_error=float(np.median(values))) for group,values in sorted(grouped.items())]
    return dict(points=len(error),spatial_blocks=len(rows),point_median_absolute_error=float(np.median(abs(error))),
        block_median_absolute_error=float(np.median([r['median_absolute_error'] for r in rows])),blocks=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve existing evidence')
    audit=json.loads(a.audit.read_text())
    if audit['uses_eval_rgb'] or audit['changes_calibration'] or audit['changes_source_frame_identity']:
        raise ValueError('Only frozen train-only temporal observations may be used')
    hashes=dict(audit['input_hashes'])
    for file,digest in hashes.items():
        if sha256(Path(file))!=digest:raise ValueError(f'Changed source evidence: {file}')
    dataset={}
    for row in audit['results']:
        for record in row['records']:
            matches=record['matches']
            if record['group']!='moving' or not {'-1','0','1'}<=set(matches):continue
            # The predictor does not use the zero-offset response being modeled.
            speed=(matches['1']['residual']-matches['-1']['residual'])/2
            if abs(speed)<.1:continue
            dataset.setdefault(row['secondary_camera'],[]).append(dict(frame_id=row['frame_id'],
                reference_index=record['reference_index'],xy=record['reference_xy'],
                speed=speed,residual=matches['0']['residual'],group=[row['frame_id'],*record['spatial_block']]))
    results=[]
    for camera,records in dataset.items():
        xy=np.array([r['xy'] for r in records]);speed=np.array([r['speed'] for r in records]);residual=np.array([r['residual'] for r in records])
        groups=[r['group'] for r in records];frame_ids=np.array([r['frame_id'] for r in records])
        for held_frame in sorted(set(frame_ids)):
            fit=frame_ids!=held_frame;held=~fit
            if fit.sum()<20 or held.sum()<10:raise ValueError('Insufficient disjoint-time fit/held data')
            train_groups=[g for g,use in zip(groups,fit) if use];held_groups=[g for g,use in zip(groups,held) if use]
            models=[]
            for name,space,time in [('timing_only',False,True),('spatial_only',True,False),('spatial_plus_timing',True,True)]:
                model=fit_residual_model(xy[fit],speed[fit],residual[fit],train_groups,spatial=space,timing=time)
                predicted=predict_residual(model,xy[held],speed[held]);model['name']=name
                model['held']=held_block_summary(residual[held]-predicted,held_groups)
                models.append(model)
            result=dict(secondary_camera=camera,held_frame=str(held_frame),fit_frames=sorted(set(frame_ids[fit])),
                baseline=held_block_summary(residual[held],held_groups),models=models)
            results.append(result)
            print(json.dumps(dict(secondary_camera=camera,held_frame=str(held_frame),baseline=result['baseline']['block_median_absolute_error'],
                models=[{k:m[k] for k in ['name','motion_coefficient_available_frames','motion_energy_not_explained_by_spatial_design']}|
                        {'held_block_median_absolute_error':m['held']['block_median_absolute_error']} for m in models])),flush=True)
    hashes[str(a.audit)]=sha256(a.audit);hashes[str(Path(__file__))]=sha256(Path(__file__))
    atomic_json(a.output,dict(uses_eval_rgb=False,changes_calibration=False,changes_source_frame_identity=False,changes_prediction=False,
        method='leave_one_time_out_spatial_vs_motion_epipolar_residual_models',interpretation=__doc__,
        target='signed zero-offset source epipolar residual',motion_predictor='(e at +1 minus e at -1)/2; e at zero is excluded',
        spatial_predictors='affine reference-camera image xy ONLY; no zero-offset secondary-camera coordinates; fit-only weighted normalization',
        weighting='one total weight per time/spatial block; Huber .5 pixels',records=dataset,results=results,input_hashes=hashes))


if __name__=='__main__':main()
