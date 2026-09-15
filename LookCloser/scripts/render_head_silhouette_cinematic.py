"""Matched geometry transfer at the actual saved wide-spiral camera poses.

Two actual times x two shots, before the real-train ending. No camera/lens,
texture, color, actor time, presentation, or published movie is altered.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from extend_head_silhouette_guard import ROOT as GEOMETRY
from probe_head_silhouette_completion import FRAMES
from cinematic_wide_spiral_centered import BASE,VARIANTS
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_head_silhouette_cinematic')


def render(variant):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2)
    parent=engine.verify_request(BASE/variant);q=deepcopy(parent)
    q['inventory']=[r for r in q['inventory'] if r['frame_id'] in FRAMES];assert len(q['inventory'])==2
    geometry={}
    for entry in q['inventory']:
        frame=entry['frame_id'];assert entry['index']<118
        verified_image(BASE/variant,frame)
        root=GEOMETRY/frame/'guarded';g=read(root/'result.json');a=read(root/'audit.json')
        assert g['native_free_space_guard_passed'] and a['result_sha256']==sha(root/'result.json')
        assert len(a['checks'])==124 and all(x['trusted_free_pixels']==0 for x in a['checks'])
        mesh=root/'mesh.ply';assert sha(mesh)==a['mesh_sha256']
        entry.update(mesh=str(mesh),mesh_sha256=sha(mesh))
        geometry[frame]={str(root/n):sha(root/n) for n in ['request.json','result.json','audit.json','mesh.ply']}
    q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,
        geometry_bindings=geometry,matched_cinematic_parent=str(BASE/variant),
        matched_cinematic_parent_sha256=sha(BASE/variant/'request.json'),
        source_quality_implementation_sha256=implementation,production_promoted=False,
        artifact_free_approval=False,texture_source_masks_unchanged=True)
    q['script_hashes'][Path(__file__).name]=sha(__file__)
    out=ROOT/variant;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    if (out/'request.json').exists():assert read(out/'request.json')==q
    atomic_json(out/'request.json',q);engine.render(out,FRAMES)


def review():
    records=[]
    for variant in VARIANTS:
        qs=[read(root/'request.json') for root in [BASE/variant,ROOT/variant]]
        for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert qs[0][k]==qs[1][k]
        for frame in FRAMES:
            images=[];rs=[];depths=[]
            for root in [BASE/variant,ROOT/variant]:
                image,r=verified_image(root,frame);images.append(image);rs.append(r)
                depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
            for k in ['camera','source_cameras','fixed_exposure']:assert rs[0][k]==rs[1][k]
            entries=[next(e for e in q['inventory'] if e['frame_id']==frame) for q in qs]
            assert entries[0]['source_masks']==entries[1]['source_masks']
            panel(ROOT/'review'/variant/(frame+'_head.png'),images,['published raw 3D','silhouette completion'],(0,0,1080,1300))
            # Entire image is retained as well; the head box is a review convenience,
            # not a crop applied to either candidate or existing movie.
            before,after=images;old,new=[d>0 for d in depths]
            records.append(dict(variant=variant,frame=frame,camera_exact=True,texture_settings_exact=True,
                gained_depth=int((~old&new).sum()),lost_depth=int((old&~new).sum()),
                changed_rgb=int(np.any(before!=after,2).sum()),
                removed_black=int(((before.max(2)==0)&(after.max(2)>0)).sum()),
                introduced_black=int(((before.max(2)>0)&(after.max(2)==0)).sum()),counts_not_quality_metrics=True))
    atomic_json(ROOT/'review'/'result.json',dict(records=records,script_sha256=sha(__file__),
        visual_status='pending',production_promoted=False,quality_metrics=False));print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.action=='review':review()
    else:
        if not a.variant:p.error('--variant required')
        render(a.variant)
