"""Seal replayed geometry/admission and matched train-textured visual evidence."""
from pathlib import Path
from admit_mhr_silhouette_patch import OUT,CANDIDATES,PRIOR,ARM,binding
from render_mhr_silhouette_patch_cpu import PARENT
from study_multiview_face_prior import read,save,sha


def main():
    target=OUT/'final_seal.json';assert not target.exists()
    checked={}
    def check(path,digest):
        path=Path(path);assert sha(path)==digest,str(path);checked[str(path)]=digest
    def receipt(path,producer=None):
        r=read(path)
        if producer:check(Path(__file__).with_name(producer),r['script_sha256'])
        for p,h in r.get('input_hashes',{}).items():check(p,h)
        for row in r.get('files',[]):check(row['path'],row['sha256'])
        return r
    audit=read(OUT/'audit.json');assert audit['status']=='passed'
    for p,h in audit['inventory'].items():check(OUT/p,h)
    a=receipt(OUT/'candidate_audit.json','audit_mhr_silhouette_patch_candidates.py');assert a['status']=='passed'
    q=read(OUT/'request.json');assert q['final_prior_binding']==binding()
    for p,h in read(CANDIDATES/'request.json')['input_hashes'].items():check(p,h)
    for p,h in read(CANDIDATES/ARM/'result.json')['hashes'].items():check(CANDIDATES/ARM/p,h)
    check(PRIOR/'final_seal.json',q['final_prior_binding']['actual_prior_seal_sha256'])
    wrapper=receipt(OUT/'review_wrapper.json','review_mhr_silhouette_patch.py')
    for p,h in wrapper['helpers'].items():check(p,h)
    receipt(OUT/'native_clay_review/result.json','review_mhr_depth_admitted_patches.py')
    receipt(OUT/'branch_difference/result.json','review_mhr_admission_branch_difference.py')
    receipt(OUT/'facet_attribution/result.json','review_mhr_silhouette_patch.py')
    receipt(OUT/'occlusion_review/result.json','localize_mhr_silhouette_patch_occlusion.py')
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    from joint_temporal_texture import ROOT as COLORS
    for view in ['old_moving','F004_E','M004_B','C004_E']:
        review=receipt(OUT/'rgb_review'/(view+'.json'),'review_mhr_silhouette_patch_rgb.py')
        check(review['panel_path'],review['panel_sha256'])
        for variant in ['baseline','strict','interpolated']:
            root=OUT/'rgb'/view/variant;folder=root/'frames/001193';r=read(root/'request.json');done=read(folder/'complete.json')
            check(root/'request.json',done['request_sha256'])
            for p,h in done['hashes'].items():check(folder/p,h)
            check(Path(__file__).with_name('render_mhr_silhouette_patch_cpu.py'),r['script_sha256'])
            for p,h in r['helpers'].items():check(Path(__file__).with_name(p),h)
            check(PARENT/'request.json',r['parent_request_sha256']);check(OUT/'request.json',r['admission_request_sha256'])
            check(COLORS/'parameters.npz',r['source_profiles_sha256']);check(COLORS/'exposure.json',r['exposure_sha256'])
            record=r['inventory'][0]
            for field in ['mesh','metadata']:check(record[field],record[field+'_sha256'])
            result=read(folder/'result.json');names=set(result['source_cameras'])
            assert len(names)==62 and not names&forbidden
            source=r['source_rows'][0];dataset=Path(source['source_dataset'])
            check(dataset/'transforms.json',source['source_transforms_sha256'])
            used=[row for row in source['source_images'] if row['physical_camera'] in names]
            assert len(used)==62 and all('frame_train_' in row['file_path'] for row in used)
            for row in used:check(dataset/row['file_path'],row['sha256'])
            masks=record['source_masks'];maskroot=Path(masks['root'])
            for key,filename in [('complete_sha256','complete.json'),('masks_sha256','masks.npz'),('cameras_sha256','cameras.json')]:
                check(maskroot/filename,masks[key])
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_silhouette_patch_admission.md'
    save(target,dict(status='passed',script_sha256=sha(__file__),report_path=str(report),report_sha256=sha(report),
        checked_bindings=checked,inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        actual_visual_review=['moving RGB','F/E RGB','M/B RGB','C/E RGB','moving occlusion2','F/E occlusion1,3','M/B occlusion1','C/E occlusion5'],
        main_visual_review=['requested clay','M/B clay','G/B clay','moving RGB'],
        verdict='local_single_frame_improvement_with_nonzero_occlusion_side_effects',production_accepted=False))
    print('seal passed',len(checked),'bindings')


if __name__=='__main__':main()
