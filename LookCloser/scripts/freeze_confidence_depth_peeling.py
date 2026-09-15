"""Seal terminal strict-source, deeper-intersection and conditional controls."""
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from study_measured_source_visibility import ROOT as STRICT,ARMS,old_root
from render_measured_depth_peeling import ROOT as PEELED,VIEWS
from render_confidence_gated_peeling import ROOT
from review_full_block_transfer import ROOT as DEPTH_ROOT
from render_smooth_temporal_mesh_video import verify_request
from review_jaw_repair_transfer import verified_image


def run():
    frame='000995';hashes={}
    assert read(STRICT/frame/'visual_review.json')['status']=='fail_global_strict_source_admission'
    visual=read(ROOT/frame/'visual_review.json');assert visual['production_promoted'] is False
    sr=read(STRICT/frame/'review_result.json');cr=read(ROOT/frame/'review_result.json')
    assert len(sr['records'])==6 and len(cr['records'])==3
    for r in sr['records']:
        out=STRICT/frame/r['arm']/r['view'];verify_request(out);verified_image(out,frame)
        assert sha(out/'request.json')==r['candidate_request_sha256']
        assert sha(out/'frames'/frame/'frame.png')==r['candidate_png_sha256']
        for name,digest in r['hashes'].items():assert sha(out/'review'/name)==digest
    for view in VIEWS:
        out=ROOT/frame/view;audit=read(out/'audit.json')
        assert audit['conditional_request_sha256']==sha(out/'request.json')
        assert audit['source_request_sha256']==sha(PEELED/frame/view/'request.json')
        assert audit['script_sha256']==sha(Path(__file__).with_name('audit_confidence_depth_peeling.py'))
        assert audit['all_conditional_changes_measured_free'] and audit['ray_and_mesh_membership_verified']
        assert audit['all_filled_rgb_reprojected_from_one_train_source']
        for result in [r for r in cr['records'] if r['view']==view]:
            for name,digest in result['hashes'].items():assert sha(out/'review'/name)==digest
        folders=[old_root(frame,arm,view) for arm in ARMS]+[STRICT/frame/arm/view for arm in ARMS]+[PEELED/frame/view,out]
        for folder in folders:
            q=verify_request(folder);verified_image(folder,frame)
            for name,digest in q['script_hashes'].items():
                p=Path(__file__).with_name(name);assert sha(p)==digest;hashes[str(p.resolve())]=digest
            for row in q['inventory']:
                if row['frame_id']!=frame:continue
                for key in ['mesh','metadata']:
                    assert sha(row[key])==row[key+'_sha256'];hashes[row[key]]=row[key+'_sha256']
            for source in q['source_rows']:
                if Path(source['source_dataset']).name!=frame:continue
                for im in source['source_images']:
                    p=Path(source['source_dataset'])/im['file_path'];assert sha(p)==im['sha256'];hashes[str(p)]=im['sha256']
            hashes[str(folder/'request.json')]=sha(folder/'request.json')
            for p in (folder/'frames'/frame).iterdir():
                if p.is_file():hashes[str(p)]=sha(p)
    received=read(DEPTH_ROOT/frame/'received.json')
    for name,digest in received['depth_hashes'].items():
        p=DEPTH_ROOT/frame/name;assert sha(p)==digest;hashes[str(p)]=digest
    for name in ['study_measured_source_visibility.py','review_measured_source_visibility.py',
                 'render_measured_depth_peeling.py','render_confidence_gated_peeling.py',
                 'review_confidence_depth_peeling.py','audit_confidence_depth_peeling.py',
                 'study_jaw_depth_footprint.py','diagnose_jaw_measured_depth.py',
                 'study_confidence_depth_prior.py','carve_patchmatch_mesh_free_space.py',
                 'native_texture_footprint.py','diffusion_mesh_repair.py','hard_surface_texture.py',
                 'view_consistent_source_quality.py','temporal_texture_view_prior.py',Path(__file__).name]:
        p=Path(__file__).with_name(name);hashes[str(p.resolve())]=sha(p)
    atomic_json(ROOT/frame/'completion.json',dict(status=visual['status'],
        visual_sha256=sha(ROOT/frame/'visual_review.json'),workers_terminal=True,
        production_promoted=False,mesh_improved=False))
    for base in [STRICT/frame,PEELED/frame,ROOT/frame]:
        for p in base.rglob('*'):
            if p.is_file() and p.name!='artifact_manifest.json':hashes[str(p)]=sha(p)
    atomic_json(ROOT/frame/'artifact_manifest.json',dict(hashes=hashes,production_promoted=False))
    for p,digest in read(ROOT/frame/'artifact_manifest.json')['hashes'].items():assert sha(p)==digest
    print('Sealed and rechecked',len(hashes),'SHA-256 bindings',flush=True)


if __name__=='__main__':run()
