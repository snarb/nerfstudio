"""Freeze measured TSDF controls and calibrated learned-depth canary evidence."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.ndimage import map_coordinates
from joint_temporal_texture import read,sha,atomic_json,cameras
from calibrated_stereo_rectification import portrait_calibration,disparity_to_world

ROOT=Path('/mnt/data/dec5_foundation_hand_geometry')
STEREO=[Path('/mnt/data/dec5_foundation_hand_stereo/001037'),Path('/mnt/data/dec5_foundation_wrist_stereo/001037')]
TSDF=Path('/mnt/data/dec5_forearm_tsdf_scale/001037')
SCRIPTS=['study_forearm_tsdf_scale.py','review_forearm_tsdf_scale.py','download_foundation_stereo_research.py',
    'download_foundation_stereo_mirror.py','calibrated_stereo_rectification.py','stage_foundation_hand_stereo.py',
    'infer_foundation_hand_stereo.py','study_foundation_wrist_pair.py','review_foundation_hand_geometry.py',
    'freeze_foundation_hand_canary.py']


def run(check):
    manifest=ROOT/'artifact_manifest.json'
    if check:
        q=read(manifest)
        for p,h in q['files'].items():
            if sha(p)!=h:raise ValueError('Changed retained/input artifact '+p)
        print('Verified',len(q['files']),'retained/input hashes',flush=True);return
    if manifest.exists():raise ValueError('Already frozen; use --check')
    deps={};rectification=[];rows,_,_=cameras('001037');lookup={r['physical_camera']:r for r in rows}
    for root in STEREO:
        q=read(root/'request.json');inference=read(root/'inference/request.json');done=read(root/'inference/complete.json')
        assert inference['staged_request_sha256']==sha(root/'request.json')
        assert done['request_sha256']==sha(root/'inference/request.json')
        assert q['script_sha256']==sha(Path(__file__).with_name('stage_foundation_hand_stereo.py'))
        assert q['rectification_sha256']==sha(Path(__file__).with_name('calibrated_stereo_rectification.py'))
        assert inference['script_sha256']==sha(Path(__file__).with_name('infer_foundation_hand_stereo.py'))
        deps.update(q['source_hashes']);deps.update(done['dinov2_source_hashes'])
        for r in q['pairs']:
            source=Path(r['directory']);cal=np.load(source/'calibration.npz');path=root/'inference'/source.name/'prediction.npz'
            result=read(path.parent/'complete.json');assert sha(path)==result['prediction_sha256']
            for n,h in r['hashes'].items():assert sha(source/n)==h
            data=np.load(path);yy,xx=np.nonzero(data['consistent']&cal['left_mask'].astype(bool));yy,xx=yy[::32],xx[::32]
            dl=data['left_disparity'][yy,xx]
            xyz=disparity_to_world(xx,yy,dl,cal['cropped_intrinsic'],cal['rectified_extrinsic'],float(cal['baseline']),float(cal['disparity_offset']))
            errors=[]
            for side,n,u in [('left',r['left'],xx),('right',r['right'],xx-dl)]:
                k,e=portrait_calibration(lookup[n]);p=xyz@e[:3,:3].T+e[:3,3];uv=p@k.T;uv=uv[:,:2]/uv[:,2:]
                expected=np.column_stack([map_coordinates(cal[side+'_map_'+axis],[yy,u],order=1,mode='nearest') for axis in ['x','y']])
                error=np.linalg.norm(uv-expected,axis=1);assert np.percentile(error,99)<.02
                errors.append(dict(side=side,p99_pixel_roundtrip=float(np.percentile(error,99)),samples=len(error)))
            rectification.append(dict(pair=source.name,checks=errors))
    geom=read(ROOT/'result.json');assert geom['script_sha256']==sha(Path(__file__).with_name('review_foundation_hand_geometry.py'))
    deps.update(geom['dependencies'])
    for r in geom['meshes']:assert sha(ROOT/r['pair']/(r['variant']+'.ply'))==r['mesh_sha256']
    q=read(TSDF/'request.json');done=read(TSDF/'complete.json');assert done['request_sha256']==sha(TSDF/'request.json')
    assert q['script_sha256']==sha(Path(__file__).with_name('study_forearm_tsdf_scale.py'))
    assert q['fuser_sha256']==sha(Path(__file__).with_name('fuse_depth_tsdf_mesh.py'));deps.update(q['source_hashes'])
    reference=read(TSDF/'fine/mesh.json')
    for r in done['variants']:
        folder=TSDF/r['variant'];meta=read(folder/'mesh.json')
        assert sha(folder/'mesh.ply')==r['mesh_sha256'] and sha(folder/'mesh.json')==r['metadata_sha256']
        for k in ['dataparser_transform','dataparser_scale']:np.testing.assert_allclose(meta[k],reference[k],atol=1e-7,rtol=0)
        assert meta['images']==reference['images'] and meta['train_image_count']==62
    model=Path('/mnt/data/dec5_foundation_model_mirror');receipt=read(model/'receipt.json')
    for r in receipt['files']:deps[r['path']]=r['sha256']
    assert len({r['sha256'] for r in receipt['mirrors']})==1
    for p,h in deps.items():assert sha(p)==h
    atomic_json(ROOT/'audit.json',dict(rectification=rectification,tsdf_train_depth_inputs_identical=True,
        six_learned_mesh_hashes_verified=True,model_mirror_hashes_agree=True,
        inference_rerun=False,learned_depth_is_not_ground_truth=True,metric_scale_from_fixed_calibration=True,
        camera_pose_optimized=False,production_updated=False,face_quality_metrics_not_recomputed=True))
    viewed=[TSDF/'geometry_review'/(n+'.png') for n in ['H004_A005_1210M6_coarse','moving_medium','moving_coarse','moving_coarsest']]
    for root in STEREO:
        for r in read(root/'request.json')['pairs']:
            source=Path(r['directory']);viewed += [source/'rectification_review.png',root/'inference'/source.name/'disparity_review.png']
    viewed += [ROOT/'review'/(n+'.png') for n in ['H004_A005_1210M6','E004_C005_1210YM','moving']]
    atomic_json(ROOT/'visual_review.json',dict(status='promising_depth_prior_not_accepted_surface',reviewer='main_agent',
        inspected={str(p):sha(p) for p in viewed},
        notes='Larger TSDF voxels do not restore the large wrist void. First two learned stereo pairs give smooth, separate front finger surfaces but leave occlusion/LR gaps and disagree in depth. Third G/A-H/A pair is markedly worse. No wholesale replacement or video rollout accepted.',
        scope='13 listed calibration/disparity/geometry panels; no new textured RGB prediction',production_updated=False))
    ps=subprocess.check_output(['ps','-eo','pid,etime,pcpu,rss,args'],text=True)
    jobs=[p for p in ps.splitlines() if any('python scripts/'+n in p for n in SCRIPTS[:-1]) and '/bin/bash' not in p]
    if jobs:raise ValueError('Canary jobs still live: '+str(jobs))
    atomic_json(ROOT/'terminal_check.json',dict(unix_time=time.time(),live_canary_workers=jobs,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        free_bytes=__import__('shutil').disk_usage('/mnt/data').free,known_jobs_terminal=True))
    roots=[ROOT,TSDF,STEREO[0].parent,STEREO[1].parent]
    files={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file() and p!=manifest}
    files.update(deps)
    external=[Path(__file__).with_name(n) for n in SCRIPTS]+[model/'receipt.json']
    external += [Path(__file__).parents[1]/'tests'/n for n in ['test_forearm_tsdf_scale.py','test_calibrated_stereo_rectification.py','test_foundation_depth_grid_mesh.py']]
    external += [Path(__file__).parents[1]/'experiments'/n for n in ['dec5_forearm_tsdf_scale.md','dec5_foundation_hand_stereo.md']]
    external += list(Path('/mnt/data').glob('dec5_foundation_*.log'))+list(Path('/mnt/data').glob('dec5_forearm_tsdf*.log'))
    # Bind the upstream code actually used, not just its repository URL.
    repo=Path('/home/brans/lookcloser_temp/FoundationStereo');external += list(repo.rglob('*.py'))+[repo/'LICENSE',repo/'readme.md']
    files.update({str(p):sha(p) for p in external})
    atomic_json(manifest,dict(files=files,status='geometry_canary_frozen_not_production',production_updated=False))
    print('Replayed native ray calibration; froze',len(files),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');run(p.parse_args().check)
