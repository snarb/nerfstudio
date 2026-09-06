#!/usr/bin/env python3
"""Check whether native train-depth planes support moving a traced mesh point.

This is a read-only geometry feasibility test, not a repair or an RGB predictor.
Diagnostic query pixels do not enter any reconstruction rule. Missing depths
remain unknown. Each 5x5 footprint fits inverse depth, the pinhole plane model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from carve_patchmatch_mesh_free_space import train_frames
from render_patchmatch_camera_path import normalize_frame


def fit_native_plane(taps, center_xy, camera, *, max_depth_residual=.0005):
    """Return a world-space unit plane n.x = offset, or explicit rejection.

    Fit only positive finite taps within .005 of the median. Require 20/25
    surviving taps; prune by actual depth residual and refit twice. Pixel
    indices are converted to camera rays with the explicit +.5 convention.
    """
    values = np.asarray(taps, dtype=np.float64)
    if values.shape != (5, 5):
        return {'valid': False, 'reason': 'missing_5x5_footprint'}
    if not np.isfinite(max_depth_residual) or max_depth_residual <= 0:
        raise ValueError('Need a positive finite plane residual tolerance')
    pose = np.asarray(camera['transform_matrix'], dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    fx, fy, cx, cy = (float(camera[k]) for k in ('fl_x', 'fl_y', 'cx', 'cy'))
    if (pose.shape != (4, 4) or center.shape != (2,) or not np.isfinite(pose).all()
            or not np.isfinite(center).all() or not np.isfinite([fx,fy,cx,cy]).all()
            or min(fx,fy) <= 0):
        raise ValueError('Invalid pinhole camera/footprint center')
    yy, xx = np.mgrid[-2:3, -2:3]
    design = np.c_[xx.ravel(), yy.ravel(), np.ones(25)]
    z = values.ravel()
    positive = np.isfinite(z) & (z > 0)
    if positive.sum() < 20:
        return {'valid': False, 'reason': 'fewer_than_20_positive_taps'}
    use = positive & (np.abs(z - np.median(z[positive])) <= .005)
    for _ in range(3):
        if use.sum() < 20:
            return {'valid': False, 'reason': 'mixed_or_nonplanar_footprint'}
        coeff, _, rank, _ = np.linalg.lstsq(design[use], 1 / z[use], rcond=None)
        if rank != 3:
            return {'valid': False, 'reason': 'rank_deficient_footprint'}
        inverse = design @ coeff
        predicted = np.divide(1., inverse, out=np.full(25,np.inf), where=inverse>0)
        residual = np.abs(predicted - np.where(positive,z,np.inf))
        # Monotone exclusion: rejected native taps never re-enter the fit.
        retained = use & np.isfinite(residual) & (residual <= max_depth_residual)
        if np.array_equal(retained,use):
            break
        use = retained
    if use.sum() < 20:
        return {'valid': False, 'reason': 'mixed_or_nonplanar_footprint'}
    coeff, _, _, _ = np.linalg.lstsq(design[use], 1 / z[use], rcond=None)
    aa, bb, cc = coeff
    normal_cv = np.array([aa*fx, bb*fy,
        cc + aa*(cx-.5-center[0]) + bb*(cy-.5-center[1])])
    length = np.linalg.norm(normal_cv)
    normal_world = pose[:3,:3] @ (normal_cv * [1,-1,-1]) / length
    offset = 1/length + normal_world @ pose[:3,3]
    final_depth = 1/(design[use] @ coeff)
    return dict(valid=True, normal=normal_world.tolist(), offset=float(offset),
        taps_used=int(use.sum()), depth_rmse=float(np.sqrt(np.mean((final_depth-z[use])**2))))


def normal_displacement(plane, point, normal, *, max_shift=.004, minimum_cosine=.5):
    point, normal = np.asarray(point,float), np.asarray(normal,float)
    if (point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all()
            or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1) > 1e-6
            or not np.isfinite([max_shift, minimum_cosine]).all()
            or max_shift <= 0 or not 0 < minimum_cosine <= 1):
        raise ValueError('Need finite point, unit mesh normal and valid bounds')
    if not plane['valid']:
        return dict(valid=False, reason=plane['reason'])
    pn = np.asarray(plane['normal'])
    cosine = float(pn @ normal)
    distance = float(plane['offset'] - pn @ point)
    if abs(cosine) < minimum_cosine:
        return dict(valid=False, reason='plane_normal_disagreement', cosine=cosine,
                    signed_plane_distance=distance)
    displacement = distance/cosine
    return dict(valid=abs(displacement) <= max_shift,
        reason='near_plane' if abs(displacement) <= max_shift else 'different_surface_layer',
        displacement=displacement, cosine=cosine, signed_plane_distance=distance)


def displacement_consensus(values, *, width=.001, minimum_views=3, minimum_fraction=.6):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or width <= 0
            or minimum_views < 2 or not 0 < minimum_fraction <= 1):
        raise ValueError('Invalid displacement votes or consensus settings')
    if not len(values):
        return dict(eligible=False, votes=0, agreeing_views=0, displacement=None)
    ordered = np.sort(values)
    right = np.searchsorted(ordered, ordered+width, side='right')
    counts = right-np.arange(len(values))
    left = int(np.argmax(counts)); cluster = ordered[left:right[left]]
    return dict(eligible=bool(len(cluster)>=minimum_views and len(cluster)/len(values)>=minimum_fraction),
        votes=len(values), agreeing_views=len(cluster), agreeing_fraction=len(cluster)/len(values),
        displacement=float(np.median(cluster)), cluster_span=float(np.ptp(cluster)),
        all_vote_min=float(ordered[0]), all_vote_max=float(ordered[-1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-audit','depth-data','mesh','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    evidence=json.loads(a.native_audit.read_text())
    data=json.loads((a.depth_data/'transforms.json').read_text())
    meta=json.loads(a.mesh_metadata.read_text())
    if sha256(a.mesh) != meta['output_sha256'] or evidence['input_hashes'][str(a.mesh)] != sha256(a.mesh):
        raise ValueError('Wrong input mesh')
    for path,digest in evidence['input_hashes'].items():
        # Historical helper code can change, but stored numerical inputs cannot.
        if Path(path).suffix != '.py' and sha256(Path(path)) != digest:
            raise ValueError('Changed audit input: '+path)
    frames={f['physical_camera']:normalize_frame(f,data,meta) for f in train_frames(data)}
    rows=[]
    for query in evidence['points']:
        vertices=np.asarray(query['triangle_vertices'])
        normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0])
        normal/=np.linalg.norm(normal)
        sources=[]; votes=[]
        if len(query['sources'])!=62 or {s['physical_camera'] for s in query['sources']} != set(frames):
            raise ValueError('Incomplete native train observation inventory')
        for source in query['sources']:
            camera=frames[source['physical_camera']]
            plane=fit_native_plane(source['native_taps'],source.get('center_xy',[0,0]),camera)
            move=normal_displacement(plane,query['world'],normal)
            if move['valid']:
                votes.append(move['displacement'])
            sources.append(dict(physical_camera=source['physical_camera'],plane=plane,**move))
        rows.append(dict(index=query['index'],pixel=query['pixel'],world=query['world'],
            triangle_id=query['triangle_id'],mesh_normal=normal.tolist(),
            consensus=displacement_consensus(votes),sources=sources))
    result=dict(uses_rgb=False,uses_eval_rgb=False,changes_geometry=False,
        diagnostic_queries_only=True,train_camera_count=62,
        settings=dict(native_footprint=5,minimum_taps=20,depth_residual=.0005,
            maximum_normal_shift=.004,minimum_normal_cosine=.5,
            consensus_interval_width=.001,minimum_views=3,minimum_fraction=.6),
        input_hashes={str(f):sha256(f) for f in [a.native_audit,a.depth_data/'transforms.json',a.mesh,
            a.mesh_metadata,Path(__file__),Path(__file__).with_name('render_patchmatch_camera_path.py')]},
        points=rows,accepted_surface_recipe=False)
    atomic_json(a.output,result)
    print(json.dumps([dict(index=r['index'],**r['consensus']) for r in rows]),flush=True)


if __name__=='__main__':
    main()
