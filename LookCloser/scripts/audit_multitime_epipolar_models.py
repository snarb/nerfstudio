#!/usr/bin/env python3
"""Held-time audit of epipolar models; never export/apply a new calibration.

An essential-matrix fit keeps supplied intrinsics; a fundamental-matrix fit is
less constrained. Both are diagnostics of correspondences, not new rig poses.
Held matches are never filtered by a fitted model's inlier mask.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_source_epipolar_residuals import fundamental,projection
from audit_fixed_camera_feature_geometry import epipolar_distance,farthest_points
from audit_spatial_temporal_residuals import held_block_summary


def normalize_pixels(points,k):
    points=np.asarray(points,float);q=np.c_[points,np.ones(len(points))]@np.linalg.inv(k).T
    return q[:,:2]/q[:,2:]


def fit_epipolar_model(a,b,ka,kb,kind):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<20 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Need at least twenty finite fit-only correspondences')
    cv2.setRNGSeed(0)
    if kind=='fundamental':
        matrix,inliers=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,.75,.999,10000)
        extra={}
    elif kind=='essential':
        na,nb=normalize_pixels(a,ka),normalize_pixels(b,kb)
        focal=np.mean([ka[0,0],ka[1,1],kb[0,0],kb[1,1]])
        essential,inliers=cv2.findEssentialMat(na,nb,np.eye(3),method=cv2.USAC_MAGSAC,prob=.999,threshold=.75/focal,maxIters=10000)
        if essential is None or essential.shape!=(3,3):raise RuntimeError('No unique essential fit')
        matrix=np.linalg.inv(kb).T@essential@np.linalg.inv(ka)
        extra={'essential_singular_values':np.linalg.svd(essential,compute_uv=False).tolist(),
               'normalized_ransac_threshold':float(.75/focal)}
    else:raise ValueError('Unknown epipolar model')
    if matrix is None or matrix.shape!=(3,3) or not np.isfinite(matrix).all():raise RuntimeError('Invalid epipolar fit')
    return matrix,dict(kind=kind,fit_points=len(a),fit_inliers=int((inliers>0).sum()),
                       ransac_threshold_pixels=.75,confidence=.999,max_iterations=10000,**extra)


def balanced_indices(points,groups,max_per_block=8):
    blocks={}
    for i,g in enumerate(groups):blocks.setdefault(tuple(g),[]).append(i)
    chosen=[]
    for ids in blocks.values():
        # SIFT can emit multiple orientations at identical xy. Select unique
        # locations before farthest-point sampling; otherwise zero-distance
        # ties can select the same record repeatedly and silently overweight it.
        unique=np.sort(np.unique(points[ids],axis=0,return_index=True)[1])
        ids=np.asarray(ids)[unique]
        chosen.extend(ids[k] for k in farthest_points(points[ids],min(len(ids),max_per_block)))
    return np.asarray(chosen,np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing diagnostic')
    audit=json.loads(a.audit.read_text());hashes=dict(audit['input_hashes'])
    if audit['uses_eval_rgb'] or audit['changes_calibration']:raise ValueError('Frozen train-only inputs required')
    for path,digest in hashes.items():
        if sha256(Path(path))!=digest:raise ValueError(f'Changed input: {path}')
    if str(a.calibration) not in hashes:raise ValueError('Calibration provenance differs')
    calibration=json.loads(a.calibration.read_text());cameras={f['physical_camera']:{**calibration,**f} for f in calibration['frames']}
    reference=cameras[audit['reference_camera']];ka,_=projection(reference,.5)
    dataset={}
    for row in audit['results']:
        for r in row['records']:
            dataset.setdefault(row['secondary_camera'],[]).append(dict(frame_id=row['frame_id'],
                a=r['reference_xy'],b=r['matches']['0']['point'],reference_index=r['reference_index'],
                group=[row['frame_id'],*r['spatial_block']]))
    results=[]
    for camera,records in dataset.items():
        pa=np.array([r['a'] for r in records]);pb=np.array([r['b'] for r in records]);ids=np.array([r['frame_id'] for r in records])
        groups=[r['group'] for r in records];kb,_=projection(cameras[camera],.5);frozen=fundamental(reference,cameras[camera],.5)
        for held_frame in sorted(set(ids)):
            fit=np.where(ids!=held_frame)[0];held=np.where(ids==held_frame)[0]
            fit=fit[balanced_indices(pa[fit],[groups[i] for i in fit])]
            held_groups=[groups[i] for i in held]
            models=[]
            for kind in ['essential','fundamental']:
                matrix,stats=fit_epipolar_model(pa[fit],pb[fit],ka,kb,kind)
                error=epipolar_distance(matrix,pa[held],pb[held])
                models.append(dict(**stats,matrix=matrix.tolist(),held=held_block_summary(error,held_groups),
                                   held_signed_residuals=error.tolist()))
            frozen_error=epipolar_distance(frozen,pa[held],pb[held])
            row=dict(secondary_camera=camera,held_frame=str(held_frame),fit_frames=sorted(set(ids[fit])),
                fit_indices=fit.tolist(),held_indices=held.tolist(),
                frozen=held_block_summary(frozen_error,held_groups),models=models)
            results.append(row)
            print(json.dumps(dict(secondary_camera=camera,held_frame=str(held_frame),
                baseline=row['frozen']['block_median_absolute_error'],models=[dict(kind=m['kind'],fit_points=m['fit_points'],fit_inliers=m['fit_inliers'],
                held_block_median_absolute_error=m['held']['block_median_absolute_error']) for m in models])),flush=True)
    hashes[str(a.audit)]=sha256(a.audit);hashes[str(Path(__file__))]=sha256(Path(__file__))
    helper=Path(__file__).with_name('audit_spatial_temporal_residuals.py');hashes[str(helper)]=sha256(helper)
    atomic_json(a.output,dict(uses_eval_rgb=False,uses_mesh=False,changes_calibration=False,changes_prediction=False,
        changes_source_frame_identity=False,method='leave_one_time_out_essential_and_fundamental_diagnostic',
        reference_camera=audit['reference_camera'],interpretation=__doc__,records=dataset,results=results,input_hashes=hashes,
        fit_selection='At most eight distinct reference-xy locations per time/128px block; spatially spread; no model-residual selection',
        held_selection='All preexisting temporal-audit records, without new epipolar inlier filtering'))


if __name__=='__main__':main()
