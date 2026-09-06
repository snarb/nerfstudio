#!/usr/bin/env python3
"""Fit prior-time epipolar diagnostics and evaluate a completely new time window."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from audit_source_epipolar_residuals import fundamental,projection
from audit_fixed_camera_feature_geometry import epipolar_distance
from audit_spatial_temporal_residuals import held_block_summary
from audit_multitime_epipolar_models import fit_epipolar_model,balanced_indices


def temporal_windows(audit):
    return {frame for row in audit['results'] for frame in row['window'].values()}


def records_for(audit,camera):
    return [dict(frame_id=row['frame_id'],a=r['reference_xy'],b=r['matches']['0']['point'],
        reference_index=r['reference_index'],group=[row['frame_id'],*r['spatial_block']])
        for row in audit['results'] if row['secondary_camera']==camera for r in row['records']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit-audit','held-audit','calibration','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing audit')
    fit_audit=json.loads(a.fit_audit.read_text());held_audit=json.loads(a.held_audit.read_text())
    if temporal_windows(fit_audit)&temporal_windows(held_audit):raise ValueError('Fit/held temporal windows overlap')
    if fit_audit['reference_camera']!=held_audit['reference_camera']:raise ValueError('Different reference camera')
    hashes={}
    for audit in [fit_audit,held_audit]:
        if audit['uses_eval_rgb'] or audit['changes_calibration']:raise ValueError('Need frozen train-only observations')
        for file,digest in audit['input_hashes'].items():
            if file in hashes and hashes[file]!=digest:raise ValueError('Fit/held input provenance differs')
            if sha256(Path(file))!=digest:raise ValueError(f'Changed input: {file}')
            hashes[file]=digest
    if str(a.calibration) not in hashes:raise ValueError('Missing immutable calibration')
    calibration=json.loads(a.calibration.read_text());cameras={f['physical_camera']:{**calibration,**f} for f in calibration['frames']}
    reference=cameras[fit_audit['reference_camera']];ka,_=projection(reference,.5);results=[]
    for camera in sorted({row['secondary_camera'] for row in held_audit['results']}):
        fit=records_for(fit_audit,camera);held=records_for(held_audit,camera)
        if not fit or not held:raise ValueError('Missing camera correspondences')
        fa=np.array([r['a'] for r in fit]);fb=np.array([r['b'] for r in fit]);ha=np.array([r['a'] for r in held]);hb=np.array([r['b'] for r in held])
        selected=balanced_indices(fa,[r['group'] for r in fit]);groups=[r['group'] for r in held]
        kb,_=projection(cameras[camera],.5);frozen=fundamental(reference,cameras[camera],.5)
        frozen_error=epipolar_distance(frozen,ha,hb);models=[]
        for kind in ['essential','fundamental']:
            matrix,stats=fit_epipolar_model(fa[selected],fb[selected],ka,kb,kind)
            error=epipolar_distance(matrix,ha,hb)
            models.append(dict(**stats,matrix=matrix.tolist(),held=held_block_summary(error,groups),held_signed_residuals=error.tolist()))
        results.append(dict(secondary_camera=camera,fit_frames=fit_audit['frames'],held_frames=held_audit['frames'],
            fit_indices=selected.tolist(),held_records=held,frozen_matrix=frozen.tolist(),
            frozen=held_block_summary(frozen_error,groups),frozen_signed_residuals=frozen_error.tolist(),models=models))
        print(json.dumps(dict(secondary_camera=camera,held_points=len(held),baseline=results[-1]['frozen']['block_median_absolute_error'],
            models=[dict(kind=m['kind'],fit_inliers=m['fit_inliers'],fit_points=m['fit_points'],
                         held_block_median_absolute_error=m['held']['block_median_absolute_error']) for m in models])),flush=True)
    for path in [a.fit_audit,a.held_audit,Path(__file__),Path(__file__).with_name('audit_multitime_epipolar_models.py')]:hashes[str(path)]=sha256(path)
    atomic_json(a.output,dict(uses_eval_rgb=False,uses_mesh=False,changes_calibration=False,changes_prediction=False,
        changes_source_frame_identity=False,fit_time_windows=sorted(temporal_windows(fit_audit)),held_time_windows=sorted(temporal_windows(held_audit)),
        results=results,input_hashes=hashes,interpretation='Completely new temporal window; no held inlier rejection. Models are diagnostics, not new camera calibration.'))


if __name__=='__main__':main()
