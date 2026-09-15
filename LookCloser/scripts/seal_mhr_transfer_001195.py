"""Final frame-transfer scope, provenance, native-render and replay audit."""
from pathlib import Path
import ast,hashlib,importlib,inspect
from transfer_mhr_001195 import ROOT,HEAD,CONFORM,TEN,FINAL,FRAME,verify
from patch_mhr_transfer_001195 import OUT,CANDIDATES,ARM
from study_multiview_face_prior import read,save,sha


def main():
    verify();target=ROOT/'final_seal.json';assert not target.exists();checked={}
    def check(p,h):
        p=Path(p);assert sha(p)==h,str(p);checked[str(p)]=h
    def receipt(path,producer=None):
        r=read(path)
        if producer:check(Path(__file__).with_name(producer),r['script_sha256'])
        for p,h in r.get('input_hashes',{}).items():check(p,h)
        for x in r.get('files',[]):check(x['path'],x['sha256'])
        return r
    for p in [ROOT/'head_audit.json',ROOT/'conformance_audit.json',ROOT/'candidate_audit.json',FINAL/'audit.json',OUT/'audit.json']:
        assert read(p)['status']=='passed';checked[str(p)]=sha(p)
    prior=read(FINAL/'final_seal.json');assert prior['status']=='passed'
    for p,h in prior['inventory'].items():check(FINAL/p,h)
    for p,h in prior['checked_bindings'].items():check(p,h)
    for p,h in read(OUT/'audit.json')['inventory'].items():check(OUT/p,h)
    q=read(OUT/'request.json');old=read(Path('/mnt/data/dec5_mhr_silhouette_patch_admission/request.json'))
    change={'frame','candidate_request_sha256','candidate_result_sha256','inputs','transfer','final_prior_binding'}
    assert {k:v for k,v in q.items() if k not in change}=={k:v for k,v in old.items() if k not in change}
    assert read(FINAL/'protocol.json')['recipe']==read(Path('/mnt/data/dec5_mhr_silhouette_convergence/protocol.json'))['recipe']
    candidate=read(CANDIDATES/'request.json');oldc=read(Path('/mnt/data/dec5_mhr_silhouette_patch_candidates/request.json'))
    for key in ['neutral_y_range_cm','maximum_edge','maximum_original_surface_distance','maximum_boundary_vertex_distance','minimum_normal_dot','minimum_centroid_distance','actual_opposite_edge_must_be_open']:
        assert candidate[key]==oldc[key],key
    for p in ['source_mesh']:check(candidate[p],candidate[p+'_sha256'])
    for name,h in q['helpers'].items():check(Path(__file__).with_name(name),h)
    check(Path(__file__).with_name('admit_mhr_local_patch_depth.py'),q['script_sha256'])
    for p,h in candidate['input_hashes'].items():check(p,h)
    for p,h in read(CANDIDATES/ARM/'result.json')['hashes'].items():check(CANDIDATES/ARM/p,h)
    for directory,producer in [('native_clay_review','review_mhr_depth_admitted_patches.py'),('branch_difference','review_mhr_admission_branch_difference.py'),('facet_attribution','review_mhr_silhouette_patch.py'),('occlusion_review','localize_mhr_silhouette_patch_occlusion.py')]:
        receipt(OUT/directory/'result.json',producer)
    from joint_temporal_texture import ROOT as COLORS,cameras
    parent_path=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json');parent=read(parent_path)
    source_rows,_,_=cameras(FRAME)
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    render_seconds={}
    for view in ['old_moving','F004_E','M004_B','C004_E']:
        rr=receipt(OUT/'rgb_review'/(view+'.json'),'review_mhr_silhouette_patch_rgb.py');check(rr['panel_path'],rr['panel_sha256'])
        for branch in ['baseline','strict','interpolated']:
            root=OUT/'rgb'/view/branch;folder=root/'frames'/FRAME;r=read(root/'request.json');done=read(folder/'complete.json')
            check(root/'request.json',done['request_sha256'])
            for p,h in done['hashes'].items():check(folder/p,h)
            check(Path(__file__).with_name('render_mhr_silhouette_patch_cpu.py'),r['script_sha256'])
            check(parent_path,r['parent_request_sha256']);assert r['recipe']==parent['recipe']
            for p,h in r['helpers'].items():check(Path(__file__).with_name(p),h)
            for p,key in [(COLORS/'parameters.npz','source_profiles_sha256'),(COLORS/'exposure.json','exposure_sha256'),(OUT/'request.json','admission_request_sha256')]:check(p,r[key])
            record=r['inventory'][0];assert record['frame_id']==FRAME
            expected=next(x['camera'] for x in parent['inventory'] if x['frame_id']==FRAME) if view=='old_moving' else next(x for x in source_rows if x['physical_camera'].startswith(view))
            assert record['camera']==expected
            for key in ['mesh','metadata']:check(record[key],record[key+'_sha256'])
            source=r['source_rows'][0];dataset=Path(source['source_dataset']);assert dataset.name==FRAME
            check(dataset/'transforms.json',source['source_transforms_sha256'])
            result=read(folder/'result.json');names=set(result['source_cameras']);assert len(names)==62 and not names&forbidden
            images=[x for x in source['source_images'] if x['physical_camera'] in names];assert len(images)==62
            for x in images:assert 'frame_train_' in x['file_path'];check(dataset/x['file_path'],x['sha256'])
            spec=record['source_masks'];maskroot=Path(spec['root'])
            for name,key in [('complete.json','complete_sha256'),('cameras.json','cameras_sha256'),('masks.npz','masks_sha256')]:check(maskroot/name,spec[key])
            render_seconds[view+'/'+branch]=result['elapsed_seconds']
    for p in ROOT.glob('*adapter*.json'):
        r=read(p);check(r['frozen_path'],r['frozen_sha256']);check(Path(__file__).with_name('patch_mhr_transfer_001195.py'),r['wrapper_sha256'])
        function=ast.parse(r['original_source']).body[0].name
        module=importlib.import_module(Path(r['frozen_path']).stem)
        assert inspect.getsource(getattr(module,function))==r['original_source']
        generated=r['original_source']
        for change in r['replacements']:
            assert generated.count(change['before'])==change['expected_count']
            generated=generated.replace(change['before'],change['after'])
        assert generated==r['generated_source'] and hashlib.sha256(generated.encode()).hexdigest()==r['generated_sha256']
    for name in ['transfer_mhr_001195.py','transfer_mhr_001195_semantics.py','review_mhr_transfer_001195.py','audit_mhr_transfer_001195.py','patch_mhr_transfer_001195.py',Path(__file__).name]:
        p=Path(__file__).with_name(name);checked[str(p)]=sha(p)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_transfer_001195.md';test=Path('/mnt/data/dec5_mhr_transfer_001195_tests.log')
    checked[str(test)]=sha(test);checked[str(report)]=sha(report)
    for p in Path('/mnt/data').glob('dec5_mhr_transfer_001195_*.log'):
        if p.name!='dec5_mhr_transfer_001195_seal.log':checked[str(p)]=sha(p)
    save(target,dict(status='passed',frame=FRAME,checked_bindings=checked,
        inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()},
        render_seconds=render_seconds,report_path=str(report),report_sha256=sha(report),script_sha256=sha(__file__),
        no_previous_frame_fitted_pose_used=True,same_candidate_and_admission_hyperparameters=True,production_accepted=False))
    print('transfer seal passed',len(checked),'bindings',flush=True)


if __name__=='__main__':main()
