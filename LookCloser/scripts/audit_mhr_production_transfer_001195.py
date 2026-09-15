"""Seal production-base transfer only after exact guards and native RGB review."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import run_mhr_production_transfer_001195 as run
from study_multiview_face_prior import read,save,sha


def main():
    import open3d as o3d
    from joint_temporal_texture import HELD_CAMERAS
    from review_mhr_production_patch_control import check_recipe
    root,out=run.ROOT,run.OUT;target=root/'final_seal.json';assert not target.exists()
    proof=run.binding();assert read(root/'request.json')==proof
    checked={}
    def check(p,h):
        p=Path(p);assert p.is_file() and sha(p)==h,str(p);checked[str(p.resolve())]=h
    check(__file__,sha(__file__));check(run.__file__,proof['wrapper_sha256'])
    check(proof['frame_adapter_path'],proof['frame_adapter_sha256'])
    check(proof['production_mesh'],proof['production_mesh_sha256']);check(proof['metadata'],proof['metadata_sha256'])
    check(run.PARENT/'request.json',proof['parent_request_sha256'])
    raw_seal=read(run.transfer.ROOT/'final_seal.json');assert raw_seal['status']=='passed'
    for p,h in raw_seal['inventory'].items():check(run.transfer.ROOT/p,h)
    for p,h in raw_seal['checked_bindings'].items():check(p,h)
    for path in [root/'candidate_adapter.json',root/'candidate_adapter_replay.json']:
        p=read(path);check(p['frozen_path'],p['frozen_sha256'])
        generated=p['original_source']
        for edit in p['replacements']:
            assert generated.count(edit['before'])==edit['expected_count'];generated=generated.replace(edit['before'],edit['after'])
        assert generated==p['generated_source'] and hashlib.sha256(generated.encode()).hexdigest()==p['generated_sha256']
        assert p['production_base_binding']==proof
    replay=read(root/'audit.json');assert replay['status']=='passed' and replay['candidate_arrays_replayed'] and replay['candidate_ply_byte_exact']
    check(out/'audit.json',replay['admission_audit_sha256'])
    depth=read(out/'audit.json');assert depth['status']=='passed' and depth['native_ray_checks_replayed']==248
    assert depth['source_geometric_depth_hashes']==62 and depth['original_prefix_exact']
    for name,h in depth['inventory'].items():
        if not name.startswith('rgb/'):check(out/name,h)
    request=read(out/'request.json');assert request['production_base_binding']==proof
    check(Path(__file__).with_name('admit_mhr_local_patch_depth.py'),request['script_sha256'])
    for name,h in request['helpers'].items():check(Path(__file__).with_name(name),h)
    old=o3d.io.read_triangle_mesh(proof['production_mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles);geometry={}
    for branch in ['strict','interpolated']:
        folder=out/run.ARM/branch;r=read(folder/'result.json')
        for p,h in r['hashes'].items():check(folder/p,h)
        m=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));v,t=np.asarray(m.vertices),np.asarray(m.triangles)
        np.testing.assert_array_equal(v[:len(ov)],ov);np.testing.assert_array_equal(t[:len(ot)],ot)
        assert np.isfinite(v).all() and t.min()>=0 and t.max()<len(v) and len(t)-len(ot)==r['added']
        checks=depth['details'][0]['branches'][branch]['native_checks'];assert len(checks)==len(set((r['camera'],r['offset']) for r in checks))==124
        assert all(r['trusted_free_pixels']==0 for r in checks)
        geometry[branch]=dict(vertices=len(v),triangles=len(t),added=r['added'])
    parent=read(run.PARENT/'request.json');check_recipe(parent)
    production=next(r for r in parent['inventory'] if r['frame_id']==run.FRAME)
    from joint_temporal_texture import cameras
    rows,_,_=cameras(run.FRAME);names=[r['physical_camera'] for r in rows]
    assert len(names)==len(set(names))==62 and not set(names)&HELD_CAMERAS
    moving_path=Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')
    moving=next(r['camera'] for r in read(moving_path)['inventory'] if r['frame_id']==run.FRAME)
    records=[]
    for view in ['old_moving','F004_E','M004_B','C004_E']:
        camera=moving if view=='old_moving' else next(r for r in rows if r['physical_camera'].startswith(view))
        for variant in ['baseline','strict','interpolated']:
            folder=out/'rgb'/view/variant;q=read(folder/'request.json');check_recipe(q)
            assert q['recipe']==parent['recipe'] and q['production_base_binding']==proof and q['inventory'][0]['camera']==camera
            assert q['inventory'][0]['source_masks']==production['source_masks']
            for key in ['profiles_sha256','exposure_sha256','source_quality_implementation_sha256']:assert q[key]==parent[key]
            check(Path(__file__).with_name('review_mhr_production_patch_control.py'),q['script_sha256'])
            for n,h in q['helpers'].items():check(Path(__file__).with_name(n),h)
            frame=folder/'frames'/run.FRAME;c=read(frame/'complete.json');check(folder/'request.json',c['request_sha256'])
            for n,h in c['hashes'].items():check(frame/n,h)
            result=read(frame/'result.json');assert result['source_cameras']==names and result['camera']==camera
            assert result['mesh_sha256']==q['inventory'][0]['mesh_sha256']
            assert Image.open(frame/'frame.png').size==(1080,1920)
            d=np.load(frame/'target_depth.npz')['depth'];assert d.shape==(1080,1920) and np.isfinite(d).all() and (d>=0).all()
            records.append(dict(view=view,branch=variant,complete_sha256=sha(frame/'complete.json')))
        review=read(out/'rgb_review'/(view+'.json'));check(review['panel_path'],review['panel_sha256'])
        for p,h in review['input_hashes'].items():check(p,h)
    for folder,producer in [('native_clay_review','review_mhr_depth_admitted_patches.py'),('branch_difference','review_mhr_admission_branch_difference.py'),
        ('occlusion_review','localize_mhr_silhouette_patch_occlusion.py'),('black_pixel_review','localize_mhr_production_patch_side_effects.py')]:
        r=read(out/folder/'result.json');check(Path(__file__).with_name(producer),r['script_sha256'])
        for p,h in r['input_hashes'].items():check(p,h)
        for item in r['files']:check(item['path'],item['sha256'])
    effects=read(root/'side_effects_wrapper.json')
    assert effects['production_base_binding']==proof and effects['wrapper_sha256']==proof['wrapper_sha256']
    check(effects['frozen_review_path'],effects['frozen_review_sha256'])
    # The frozen runner's common trailing receipt overwrote its richer adapter
    # receipt. Reconstruct these exact, hash-bound code transformations without
    # executing them; do not silently mutate that frozen producer or receipt.
    import localize_mhr_silhouette_patch_occlusion as occlusion
    import localize_mhr_production_patch_side_effects as side_effects
    source=Path(run.__file__).read_text()
    edits1=[("'frames/001193'","'frames'/FRAME",1)]
    edits2=[("    residual_mask_attribution()","    # No new target-defined residual search in this bounded transfer.",1)]
    assert "dict(OUT=OUT,FRAME=FRAME)" in source and "dict(ROOT=ROOT,OUT=OUT,FRAME=FRAME)" in source
    for edits in [edits1,edits2]:
        for before,after,count in edits:assert before in source and after in source
    _,a=run.adapters.adapted(occlusion,'main',edits1,dict(OUT=out,FRAME=run.FRAME))
    _,b=run.adapters.adapted(side_effects,'main',edits2,dict(ROOT=root,OUT=out,FRAME=run.FRAME))
    reconstructed=dict(occlusion_adapter=a,side_effects_adapter=b)
    for k in ['occlusion_adapter','side_effects_adapter']:
        p=reconstructed[k];check(p['frozen_path'],p['frozen_sha256']);generated=p['original_source']
        for edit in p['replacements']:
            assert generated.count(edit['before'])==edit['expected_count'];generated=generated.replace(edit['before'],edit['after'])
        assert generated==p['generated_source'] and hashlib.sha256(generated.encode()).hexdigest()==p['generated_sha256']
    verdict=read(root/'visual_review.json');assert verdict['reviewer']=='LLM' and verdict['production_promoted'] is False
    for p,h in verdict['viewed_images'].items():check(p,h)
    for p,h in verdict.get('main_reviewed_images',{}).items():check(p,h)
    repo=Path(__file__).resolve().parents[1];report=repo/'experiments/dec5_mhr_production_transfer_001195.md';tests=repo/'tests/test_mhr_production_transfer_001195.py'
    for p in [report,tests]:check(p,sha(p))
    save(target,dict(status='passed',checked_bindings=checked,inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        script_sha256=sha(__file__),geometry=geometry,rgb_records=records,report_path=str(report),report_sha256=sha(report),
        tests_path=str(tests),tests_sha256=sha(tests),production_modified=False,full_sequence_accepted=False,matched_cpu_not_cuda_equivalence=True,
        reconstructed_side_effect_adapters=reconstructed,adapter_receipt_caveat='Frozen runner common tail overwrote richer adapter receipt; exact source transformations reconstructed at seal, not represented as original execution receipt.'))
    print('seal passed',len(checked),'bindings',flush=True)


if __name__=='__main__':main()
