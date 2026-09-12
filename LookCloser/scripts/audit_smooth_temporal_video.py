"""Record actual human/LLM reviews separately from immutable render receipts.

An artifact-tolerant viewing decision is never reported as artifact-free geometry.
The audit requires every time instant, every final checksum and reviewed evidence.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime,timezone
from pathlib import Path
import os
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,HELD_CAMERAS,CALIBRATION
from review_encode_smooth_temporal_video import completed
from render_smooth_temporal_mesh_video import calibration_pose
from scipy.spatial.transform import Rotation


def record_review(output,groups,notes,catastrophic=False):
    if not groups or not notes or not notes.strip():raise ValueError('Actual reviewed groups and findings are required')
    evidence=[];frames=[]
    for group in groups:
        folder=output/'contact_sheets'/group;manifest=read(folder/'manifest.json')
        for name in ['face.png','ear_hair.png','lipstick_hand.png']:
            path=folder/name
            evidence.append({'path':str(path),'sha256':sha(path)})
        frames.extend(manifest['render_hashes'])
    if len(set(frames))!=len(frames):raise ValueError('Duplicate reviewed instant')
    review={'reviewer':'Codex LLM','utc':datetime.now(timezone.utc).isoformat(),
            'frame_ids':frames,'evidence':evidence,'notes':notes,'catastrophic_geometry':catastrophic,
            'status':'fail' if catastrophic else 'accepted_with_known_artifacts',
            'strict_artifact_free':False,'reference':'Nearest train RGB is context, NOT novel-view ground truth'}
    directory=output/'visual_reviews';directory.mkdir(exist_ok=True)
    path=directory/f'{frames[0]}_{frames[-1]}.json'
    if path.exists() and read(path)['notes']!=notes:raise ValueError('Preserve earlier review; use separate reviewed groups')
    atomic_json(path,review)


def validate_review_inventory(ids,reviews):
    mapped={}
    for path,review in reviews:
        if review['status'] not in ['accepted_with_known_artifacts','fail']:raise ValueError('Unfinished review')
        for frame in review['frame_ids']:
            if frame in mapped:raise ValueError('Conflicting duplicate review')
            mapped[frame]=(path,review)
    if set(mapped)!=set(ids):raise ValueError('Visual review inventory incomplete or unexpected')
    return mapped


def audit(output):
    request,ready=completed(output);ids=request['ordered_frame_ids']
    if len(ids)!=150 or len(set(ids))!=150 or [r['frame_id'] for r,_ in ready]!=ids:raise ValueError('Incomplete temporal inventory')
    reviews=[(p,read(p)) for p in sorted((output/'visual_reviews').glob('*.json'))]
    mapped=validate_review_inventory(ids,reviews)
    for _,review in reviews:
        for item in review['evidence']:
            if sha(item['path'])!=item['sha256']:raise ValueError('Reviewed pixels changed')
    video=read(output/'video_manifest.json')
    if video['source_frames']!=ids or sha(video['video'])!=video['video_sha256']:raise ValueError('Video changed')
    encoded=read(output/'encoded_review.json');temporal=read(output/'temporal_visual_review.json')
    if encoded['video_sha256']!=video['video_sha256'] or encoded['encoded_temporal_overview_sha256']!=sha(output/'encoded_temporal_overview.png'):
        raise ValueError('Encoded review changed')
    if temporal['video_sha256']!=video['video_sha256'] or temporal['status']!='accepted_with_known_artifacts':
        raise ValueError('Temporal visual review incomplete or failed')
    notable=set(temporal.get('notable_geometry_frames',[]))
    if not notable<=set(ids):raise ValueError('Unknown notable geometry frame')
    gains=[];seconds=[];rows=[];poses=[];calibration=read(CALIBRATION)
    for record,result in ready:
        frame=record['frame_id'];directory=output/'frames'/frame
        if sha(record['mesh'])!=record['mesh_sha256'] or sha(record['metadata'])!=record['metadata_sha256']:raise ValueError('Geometry changed')
        rgb=np.asarray(Image.open(directory/'frame.png'))
        if rgb.shape!=(1920,1080,3):raise ValueError('Unexpected portrait dimensions')
        source=np.asarray(Image.open(directory/'source_ids.png'))
        if not np.isin(source,np.r_[np.arange(62),255]).all():raise ValueError('Invalid source camera label')
        if len(result['source_cameras'])!=62 or len(set(result['source_cameras']))!=62 or set(result['source_cameras'])&HELD_CAMERAS:
            raise ValueError('Source split changed')
        if result['camera']!=record['camera'] or result['target_rgb_read'] or result['rgb_averaging']:raise ValueError('Recipe violation')
        poses.append(calibration_pose(record['camera'],calibration,read(record['metadata']))['transform_matrix'])
        for field in ['rgb_coverage','mesh_hit_fraction','elapsed_seconds']:
            if not np.isfinite(result[field]) or result[field]<=0:raise ValueError('Nonfinite or empty output')
        depth=np.load(directory/'target_depth.npz')['depth']
        if depth.shape!=(1080,1920) or not np.isfinite(depth).all() or (depth<0).any():raise ValueError('Invalid retained depth')
        path,review=mapped[frame];gains.append(result['fixed_exposure']);seconds.append(result['elapsed_seconds'])
        rows.append({'frame_id':frame,'index':record['index'],'render_sha256':result['render_sha256'],
                     'mesh_sha256':result['mesh_sha256'],'visual_status':'fail_local_geometry' if frame in notable else review['status'],
                     'catastrophic_geometry':review['catastrophic_geometry'],'notable_geometry_artifact':frame in notable,
                     'visual_notes':review['notes'],'visual_review':str(path)})
    if len(set(gains))!=1:raise ValueError('Time-varying exposure')
    poses=np.asarray(poses);nxt=np.roll(poses,-1,axis=0)
    angular=Rotation.from_matrix(poses[:,:3,:3].transpose(0,2,1)@nxt[:,:3,:3]).magnitude()*30*180/np.pi
    speed=np.linalg.norm(nxt[:,:3,3]-poses[:,:3,3],axis=1)*30
    if angular.max()>5 or speed.min()<=0 or speed.max()/speed.min()>1.02:
        raise ValueError('Fast, discontinuous or nonuniform camera path after mesh normalization')
    with (output/'frames_audit.csv.partial').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(output/'frames_audit.csv.partial',output/'frames_audit.csv')
    failures=sum(r['catastrophic_geometry'] for r in rows)
    atomic_json(output/'audit.json',{'integrity':'pass','visual_acceptance':'fail' if failures else 'accepted_with_known_artifacts',
        'strict_artifact_free':False,'source_instant_count':150,'reviewed_frame_count':150,'catastrophic_frame_count':failures,
        'notable_geometry_frames':sorted(notable),'notable_local_geometry_failure_count':len(notable),
        'goal_fully_achieved_claimed':False,
        'fixed_exposure':gains[0],'render_seconds_min_median_max':np.quantile(seconds,[0,.5,1]).tolist(),
        'angular_speed_degrees_per_second_min_median_max':np.quantile(angular,[0,.5,1]).tolist(),
        'calibration_space_speed_max_min_ratio':float(speed.max()/speed.min()),
        'no_full_frame_image_quality_metrics':True,'face_metric_control':'Separate two-time heldout control, not 150-view GT metrics',
        'video_sha256':video['video_sha256'],'request_sha256':sha(output/'request.json'),
        'temporal_visual_review_sha256':sha(output/'temporal_visual_review.json'),
        'review_hashes':{str(p):sha(p) for p,_ in reviews},'csv_sha256':sha(output/'frames_audit.csv'),
        'audit_script_sha256':sha(__file__)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['record','audit','pending'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',nargs='+')
    p.add_argument('--notes');p.add_argument('--catastrophic',action='store_true');a=p.parse_args()
    if a.action=='record':record_review(a.output,a.groups,a.notes,a.catastrophic)
    elif a.action=='audit':audit(a.output)
    else:
        seen={f for p in (a.output/'visual_reviews').glob('*.json') for f in read(p)['frame_ids']}
        groups=[p.parent.name for p in sorted((a.output/'contact_sheets').glob('*/manifest.json'))
                if not set(read(p)['render_hashes'])<=seen]
        print(json.dumps({'reviewed_frames':len(seen),'ready_unreviewed_groups':groups}))


if __name__=='__main__':main()
