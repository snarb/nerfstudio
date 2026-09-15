"""Verify and seal the terminal two-arm geometry pilot and diagnostic trace."""
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from prune_measured_free_surface import ROOT
from ablate_free_surface_near_footprint import ROOT as CENTER
from diagnose_lipstick_fin_depth import ROOT as DIAG
from review_measured_free_surface import VIEWS,baseline
from review_jaw_repair_transfer import verified_image
from render_smooth_temporal_mesh_video import verify_request


def run():
    frame='000995';hashes={}
    dq=read(DIAG/'request.json');dr=read(DIAG/'result.json')
    assert dr['request_sha256']==sha(DIAG/'request.json')
    for name,digest in dr['hashes'].items():assert sha(DIAG/name)==digest
    for p,digest in dq['scripts'].items():assert sha(p)==digest;hashes[p]=digest
    for p,digest in dq['rgb_receipt']['source_rgb_hashes'].items():assert sha(p)==digest;hashes[p]=digest
    trace=read(DIAG/'rgb_trace/result.json')
    assert trace['diagnostic_request_sha256']==sha(DIAG/'request.json')
    assert trace['script_sha256']==sha(Path(__file__).with_name('trace_lipstick_fin_rgb_sources.py'))
    for case in trace['examples']:assert sha(DIAG/'rgb_trace'/case['image'])==case['sha256']
    assert read(DIAG/'visual_review.json')['status']=='false_surface_and_actual_cloth_source_confirmed'
    for base in (ROOT,CENTER):
        root=base/frame;q=read(root/'request.json');result=read(root/'result.json')
        audit=read(root/'independent_audit.json');visual=read(root/'visual_review.json')
        assert audit['request_sha256']==sha(root/'request.json')
        assert audit['result_sha256']==sha(root/'result.json')
        assert audit['script_sha256']==sha(Path(__file__).with_name('audit_measured_free_surface.py'))
        assert audit['all_removed_samples_replayed'] and audit['vertices_and_triangle_subset_verified']
        assert visual['production_promoted'] is False and set(visual['views'])==set(VIEWS)
        assert all(v['status']=='fail_to_fix' for v in visual['views'].values())
        for p,digest in q['scripts'].items():assert sha(p)==digest;hashes[p]=digest
        assert sha(q['mesh'])==q['mesh_sha256'];hashes[q['mesh']]=q['mesh_sha256']
        for name,digest in result['hashes'].items():assert sha(root/name)==digest
        review=read(root/'review/result.json')
        for r in review['records']:
            view=r['view'];out=root/'rgb'/view;old=baseline(frame,view)
            verify_request(out);verified_image(out,frame);verified_image(old,frame)
            assert sha(out/'request.json')==r['candidate_request_sha256']
            assert sha(out/'frames'/frame/'frame.png')==r['candidate_png_sha256']
            assert sha(old/'request.json')==r['baseline_request_sha256']
            assert sha(old/'frames'/frame/'frame.png')==r['baseline_png_sha256']
            for name,digest in r['hashes'].items():assert sha(root/'review'/view/name)==digest
            for name,digest in read(out/'request.json')['script_hashes'].items():
                p=Path(__file__).with_name(name);assert sha(p)==digest;hashes[str(p.resolve())]=digest
            for name in ('request.json',f'frames/{frame}/frame.png',f'frames/{frame}/target_depth.npz',f'frames/{frame}/complete.json'):
                p=old/name;hashes[str(p)]=sha(p)
        atomic_json(root/'review_completion.json',dict(status=visual['status'],
            visual_sha256=sha(root/'visual_review.json'),audit_sha256=sha(root/'independent_audit.json'),
            workers_terminal=True,production_promoted=False))
    # Bind the raw-depth receipt and original 62 maps, not just derived evidence.
    from review_full_block_transfer import ROOT as DEPTH_ROOT
    received=read(DEPTH_ROOT/frame/'received.json')
    for name,digest in received['depth_hashes'].items():
        p=DEPTH_ROOT/frame/name;assert sha(p)==digest;hashes[str(p)]=digest
    for root in (ROOT/frame,CENTER/frame,DIAG):
        for p in root.rglob('*'):
            if p.is_file() and p.name!='study_artifact_manifest.json':hashes[str(p)]=sha(p)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    atomic_json(DIAG/'study_artifact_manifest.json',dict(hashes=hashes,production_promoted=False))
    for p,digest in read(DIAG/'study_artifact_manifest.json')['hashes'].items():assert sha(p)==digest
    print('Sealed and rechecked',len(hashes),'SHA-256 bindings',flush=True)


if __name__=='__main__':run()
