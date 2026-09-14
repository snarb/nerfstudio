"""Verify actor-time preservation and a cyclic permutation of calibrated camera poses."""
import argparse
from pathlib import Path
import numpy as np
from joint_temporal_texture import CALIBRATION,read,sha,atomic_json,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from finalize_local_mesh_repair import verify_hashes
from probe_temporal_camera_phase import PARENT


def audit(root, require_complete=False):
    parent=verify_request(PARENT);request=verify_request(root);cal=read(CALIBRATION)
    if request['ordered_frame_ids']!=parent['ordered_frame_ids'] or len(request['inventory'])!=150:
        raise ValueError('Actor time inventory changed')
    phase=request['recipe']['camera_phase_frames'];rows=[];positions=[]
    for i,r in enumerate(request['inventory']):
        old=parent['inventory'][i]; shifted=parent['inventory'][(i+phase)%150]
        if r['frame_id']!=old['frame_id'] or r['mesh_sha256']!=old['mesh_sha256'] or r['source_transforms_sha256']!=old['source_transforms_sha256']:
            raise ValueError('Actor frame or geometry changed')
        a=np.array(calibration_pose(r['camera'],cal,read(r['metadata']))['transform_matrix'])
        b=np.array(calibration_pose(shifted['camera'],cal,read(shifted['metadata']))['transform_matrix'])
        if not np.allclose(a,b,atol=1e-7):raise ValueError('Camera phase transfer mismatch')
        if any(r['camera'][k]!=old['camera'][k] for k in ['fl_x','fl_y','cx','cy','w','h']):
            raise ValueError('Fixed intrinsics changed')
        positions.append(a[:3,3]);path=root/'frames'/r['frame_id']
        if not (path/'complete.json').exists():
            if require_complete:raise ValueError('Incomplete full video inventory')
            continue
        receipt=read(path/'complete.json')
        if receipt['request_sha256']!=sha(root/'request.json'):raise ValueError('Wrong render request')
        verify_hashes(path,receipt['hashes']);result=read(path/'result.json')
        if result['source_time_frame_count']!=1 or result['target_rgb_read'] or set(result['source_cameras'])&HELD_CAMERAS:
            raise ValueError('Invalid source provenance')
        if result['camera']!=r['camera']:raise ValueError('Render camera differs from request')
        rows.append(dict(frame=r['frame_id'],complete_sha256=sha(path/'complete.json')))
    positions=np.array(positions);steps=np.linalg.norm(np.roll(positions,-1,axis=0)-positions,axis=1)
    if len(np.unique(positions,axis=0))!=150 or steps.min()<=0:raise ValueError('Static or duplicate camera')
    atomic_json(root/'phase_audit.json',dict(rows=rows,rendered_count=len(rows),requested_count=150,
        actual_actor_times_preserved=True,identical_periodic_path_phase_shift=phase,
        distinct_calibrated_camera_positions=150,minimum_calibration_step=float(steps.min()),
        maximum_calibration_step=float(steps.max()),full_sequence_rendered=len(rows)==150,
        request_sha256=sha(root/'request.json'),script_sha256=sha(__file__),visual_acceptance=False))
    print(f'Phase audit: {len(rows)}/150 renders; camera and actor provenance verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_phase30_dynamic_150'))
    p.add_argument('--require-complete',action='store_true');a=p.parse_args();audit(a.root,a.require_complete)
