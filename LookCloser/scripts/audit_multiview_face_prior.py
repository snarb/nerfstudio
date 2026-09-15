"""Read-only source/residual replay for the bounded face correspondence experiment."""
from pathlib import Path
import argparse
import numpy as np
from study_multiview_face_prior import OUT,sha,read,save,portrait_to_native
from triangulate_face_prior import projection_matrices,project,quantiles

def audit(root):
    from joint_temporal_texture import cameras,HELD_CAMERAS,ROOT
    request=read(root/'request.json');inference=read(root/'inference.json');fitroot=root/'triangulation'
    fitrequest=read(fitroot/'request.json');result=read(fitroot/'result.json')
    assert sha(root/'request.json')==inference['request_sha256']==fitrequest['source_request_sha256']
    assert sha(root/'inference.json')==fitrequest['inference_sha256']
    assert sha(fitroot/'request.json')==result['request_sha256']
    assert sha(root/'face_landmarker.task')==request['model_sha256']
    assert sha(root/'topology.json')==inference['topology_sha256']
    assert sha(Path(__file__).with_name('study_multiview_face_prior.py'))==request['script_sha256']==fitrequest['helper_sha256']
    assert sha(Path(__file__).with_name('triangulate_face_prior.py'))==fitrequest['script_sha256']
    assert request['stage_sha256']==sha(root/'stage.json')
    records=[];outputs={};source_count=0;replayed=0
    for frame in request['frames']:
        spec=read(root/frame/'input.json');stage=next(r for r in read(root/'stage.json')['frames'] if r['frame']==frame)
        assert sha(root/frame/'input.json')==stage['input_sha256']
        current,mesh,meta=cameras(frame);rows={r['camera']['physical_camera']:r['camera'] for r in spec['inputs']}
        assert not (set(rows)&HELD_CAMERAS) and len(rows)==62
        for row in current:
            expected=rows[row['physical_camera']]
            for key in ['transform_matrix','fl_x','fl_y','cx','cy','w','h']:
                np.testing.assert_allclose(row[key],expected[key],rtol=0,atol=0)
        assert str(mesh)==spec['mesh'] and sha(mesh)==spec['mesh_sha256']
        assert str(meta)==spec['metadata'] and sha(meta)==spec['metadata_sha256']
        receipt=spec['display_receipt']
        for key,name in [('parameters_sha256','parameters.npz'),('profiles_sha256','camera_profiles.json'),('exposure_sha256','exposure.json')]:assert sha(ROOT/name)==receipt[key]
        for path,digest in receipt['source_rgb_hashes'].items():assert sha(path)==digest;source_count+=1
        for item in spec['inputs']:assert sha(item['path'])==item['sha256']
        detected={r['camera']:r for r in inference['records'] if r['frame']==frame and r['detected']==1}
        xy={n:portrait_to_native(r['portrait_xy'])[:468] for n,r in detected.items()}
        for arm in request['triangulation']['arms']:
            folder=fitroot/frame/arm;saved=read(folder/'result.json');points=np.load(folder/'points.npz')['points'];summary=saved['summary']
            assert sha(folder/'points.npz')==summary['points_sha256']
            recomputed={};valset=set(summary['validation_cameras'])
            for entry in saved['landmarks']:
                if entry['status']!='triangulated':continue
                k=entry['index'];names=entry['fit_cameras'];vn=entry['validation_cameras'];selected=entry['selected_fit_cameras']
                assert not (set(names)&valset) and set(vn)<=valset and not ((set(names)|set(vn))&HELD_CAMERAS)
                prediction,depth=project(points[k],projection_matrices([rows[n] for n in names]));errors=np.linalg.norm(prediction[0]-np.array([xy[n][k] for n in names]),axis=-1)
                np.testing.assert_allclose(errors,entry['fit_errors'],rtol=0,atol=1e-9)
                sel=np.array([n in selected for n in names]);np.testing.assert_allclose(errors[sel],entry['selected_fit_errors'],rtol=0,atol=1e-9)
                ve=[]
                if vn:
                    uv,z=project(points[k],projection_matrices([rows[n] for n in vn]));ve=np.linalg.norm(uv[0]-np.array([xy[n][k] for n in vn]),axis=-1)
                    np.testing.assert_allclose(ve,entry['validation_errors'],rtol=0,atol=1e-9);assert (z>0).all()
                passed=len(vn)>=2 and sel.sum()>=3 and np.median(errors[sel])<=2 and np.median(ve)<=2 and np.percentile(ve,90)<=4
                assert bool(passed)==entry['passed'];replayed+=1
            for group,indices in request['groups'].items():
                entries=[saved['landmarks'][k] for k in indices]
                fe=[x for e in entries for x in e.get('selected_fit_errors',[])];ve=[x for e in entries for x in e.get('validation_errors',[])]
                actual=dict(landmarks=len(indices),triangulated=sum(e['status']=='triangulated' for e in entries),passing=sum(e['passed'] for e in entries),
                    passing_fraction=sum(e['passed'] for e in entries)/len(indices),fit=quantiles(fe),validation=quantiles(ve))
                assert actual==summary['groups'][group];recomputed[group]=actual
            assert summary['lower_face_gate_passed']==all(recomputed[g]['passing_fraction']>=.8 for g in ['jaw','cheek'])
            records.append(dict(frame=frame,arm=arm,groups=recomputed,lower_face_gate_passed=summary['lower_face_gate_passed']))
            for path in folder.iterdir():
                if path.is_file():outputs[str(path)]=sha(path)
    assert len(records)==4 and not result['geometry_changed']
    assert result['geometry_completion_allowed']==all(r['lower_face_gate_passed'] for r in records if r['arm']=='train_consensus')
    save(root/'audit.json',dict(records=records,source_exr_hashes_verified=source_count,landmarks_reprojected=replayed,
        source_cameras_unchanged=True,original_meshes_unchanged=True,heldout_excluded=True,validation_excluded_from_fitting=True,
        model_hash_verified=True,inference_not_rerun=True,annotation_ground_truth_not_available=True,geometry_changed=False,
        outputs=outputs,script_sha256=sha(__file__),request_sha256=sha(root/'request.json')))
    print('Audit passed:',source_count,'source EXRs;',replayed,'triangulated landmark residuals; no geometry change',flush=True)

def finalize(root):
    """Record the specific native images actually inspected, not every saved view."""
    audit=read(root/'audit.json');result=read(root/'triangulation/result.json')
    assert not result['geometry_completion_allowed'] and not audit['geometry_changed']
    paths=[root/f/'train_head_preview.png' for f in ['001193','001195']]
    paths.extend(root/'001195/landmarks'/(n+'.png') for n in ['G004_B005_1210FG','M004_B005_12109O','F004_D005_1210KW'])
    for frame in ['001193','001195']:
        paths.extend(root/'triangulation'/frame/'all_fit_robust'/(n+'_reprojection.png') for n in ['G004_B005_1210FG','M004_B005_12109O'])
        paths.append(root/'triangulation'/frame/'train_consensus/M004_B005_12109O_reprojection.png')
    save(root/'visual_review.json',dict(reviewed_images={str(p):sha(p) for p in paths},
        verdict='Reject direct triangulated jaw/cheek completion on these two times',
        observations=['Predicted face topology looks plausible in single views',
            'Jaw-contour reprojections visibly shift relative to observed predictions',
            'Consensus cheek points are closer, but sparse and not dense surface validation'],
        parametric_head_fitting_not_tested=True,all_saved_images_reviewed=False,geometry_created=False))
    save(root/'complete.json',dict(audit_sha256=sha(root/'audit.json'),visual_review_sha256=sha(root/'visual_review.json'),
        inference_sha256=sha(root/'inference.json'),triangulation_sha256=sha(root/'triangulation/result.json'),
        report='experiments/dec5_multiview_face_prior.md',geometry_promoted=False,model_z_used_as_metric=False,
        psnr_ssim_lpips='N/A: failed correspondence gate before any surface/RGB prediction',status='completed_negative_pilot'))
    print('Finalized negative face correspondence pilot; 11 native images reviewed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:finalize(a.root)
    else:audit(a.root)
