"""Replay and freeze the rejected hand correspondence controls, not a mesh repair."""
from pathlib import Path
import argparse
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
from temporal_rigid_patch import project_native
from triangulate_hand_landmarks import triangulate

BASE=Path('/mnt/data/dec5_hand_landmark_triangulation')
PRIOR=Path('/mnt/data/dec5_hand_landmark_prior')
OBS=Path('/mnt/data/dec5_wrist_observations')
ROOTS=[BASE,PRIOR,Path('/mnt/data/dec5_temporal_wrist_stage_diagnosis'),Path('/mnt/data/dec5_geometry_reset_wrist_tracking')]


def run(check):
    inventory=BASE/'artifact_manifest.json'
    if check:
        frozen=read(inventory)
        for path,expected in frozen['files'].items():
            if sha(path)!=expected:raise ValueError('Changed artifact: '+path)
        print('Verified',len(frozen['files']),'retained/input hashes');return
    if inventory.exists():raise ValueError('Already frozen; use --check')
    req=read(BASE/'request.json');result=read(BASE/'result.json');inference=read(PRIOR/'result.json');ireq=read(PRIOR/'request.json')
    assert result['request_sha256']==sha(BASE/'request.json')
    assert req['inference_result_sha256']==sha(PRIOR/'result.json')
    assert req['inference_request_sha256']==sha(PRIOR/'request.json')==inference['request_sha256']
    assert sha(ireq['model'])==ireq['model_sha256']
    external=[Path(ireq['model']),Path(__file__),Path(__file__).parents[1]/'experiments/dec5_hand_landmark_consistency.md']
    for root,script in zip(ROOTS,['triangulate_hand_landmarks.py','study_hand_landmark_prior.py','diagnose_temporal_wrist_stages.py','study_geometry_reset_wrist_tracking.py']):
        request=read(root/'request.json'); path=Path(__file__).with_name(script)
        assert sha(path)==request['script_sha256'];external.append(path)
        if 'helper_sha256' in request:
            helper=Path(__file__).with_name('temporal_rigid_patch.py');assert sha(helper)==request['helper_sha256'];external.append(helper)
        for path,digest in request.get('flow_hashes',{}).items():
            assert sha(path)==digest;external.append(Path(path))
    for record in inference['records']:
        assert sha(record['source'])==record['source_sha256'];external.append(Path(record['source']))
    replayed=0
    for frame_result in result['frames']:
        frame=frame_result['frame']; receipt=OBS/frame/'result.json';external.append(receipt)
        assert sha(receipt)==req['observations'][frame]
        rows={r['camera']['physical_camera']:r['camera'] for r in read(receipt)['records']}
        xy={r['camera']:np.array(r['hands'][0]['portrait_xy']) for r in inference['records'] if r['frame']==frame and r['detected']==1}
        uv={name:np.stack([1919-p[:,1],p[:,0]],axis=1) for name,p in xy.items()}
        evidence=np.load(BASE/frame/'evidence.npz');fit_errors=[];validation_errors=[]
        for joint in frame_result['joints']:
            j=joint['joint']
            if joint['status']!='triangulated':
                assert not evidence['good'][j];continue
            names=joint['fit_views']; assert req['validation_camera'] not in names
            point,errors=triangulate([rows[n] for n in names],[uv[n][j] for n in names])
            np.testing.assert_allclose(point,evidence['points'][j],rtol=0,atol=1e-10)
            np.testing.assert_allclose(errors,joint['fit_errors'],rtol=0,atol=1e-7);fit_errors.extend(errors);replayed+=1
            if 'validation_error' in joint:
                n=req['validation_camera'];error=float(np.linalg.norm(project_native(point[None],rows[n])[0][0]-uv[n][j]))
                np.testing.assert_allclose(error,joint['validation_error'],atol=1e-7,rtol=0);validation_errors.append(error)
        np.testing.assert_allclose(np.median(fit_errors),frame_result['fit_median'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(np.median(validation_errors),frame_result['validation_median'],atol=1e-7,rtol=0)
    viewed=[BASE/f/'H004_C005_1210SZ_reprojection.png' for f in ireq['times']]
    viewed += [PRIOR/f/(n+'_landmarks.png') for f in ['001029','001037'] for n in ['G004_B005_1210FG','H004_A005_1210M6','H004_C005_1210SZ']]
    viewed += [ROOTS[2]/f/'H004_A005_1210M6_reprojection.png' for f in ['001031','001037']]
    viewed += [ROOTS[3]/(n+'_independent_reprojection.png') for n in ['H004_A005_1210M6','H004_C005_1210SZ']]
    atomic_json(BASE/'visual_review.json',dict(status='fail_for_geometry_transfer',reviewer='main_agent',
        inspected={str(p):sha(p) for p in viewed},notes='View-dependent joint assignments; large 001035 mismatch. Flow reset does not improve fixed independent endpoint.',
        scope='15 listed native correspondence panels, not a review of every saved overlay or a new video'))
    atomic_json(BASE/'audit.json',dict(replayed_joints=replayed,heldout_rgb_used=False,geometry_changed=False,
        inference_rerun=False,neural_anatomical_accuracy_verified=False,status='replay_pass_candidate_rejected'))
    files={str(p):sha(p) for root in ROOTS for p in root.rglob('*') if p.is_file() and p!=inventory}
    files.update({str(p):sha(p) for p in external})
    atomic_json(inventory,dict(files=files,status='rejected_controls_frozen',geometry_changed=False))
    print('Replayed',replayed,'joints; froze',len(files),'files')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
