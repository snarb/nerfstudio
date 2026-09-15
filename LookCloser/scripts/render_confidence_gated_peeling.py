"""Conditional measured-layer rendering; unchanged RGB outside proven free space.

Strict source visibility is unsuitable globally when real depths are missing.
Apply it only where the actually selected source has a farther measured layer,
all native-center near observations are absent, and >=6 stable/corroborated train
views put the current first hit in free space. No image ROI or target GT is used.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_measured_source_visibility import ROOT as STRICT,old_root
from render_measured_depth_peeling import ROOT as PEELED,VIEWS
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,project_integer,unproject,support
from diagnose_jaw_measured_depth import observed_at
from carve_patchmatch_mesh_free_space import free_space_evidence
from review_jaw_repair_transfer import verified_image

ROOT=Path('/mnt/data/dec5_confidence_gated_peeling')


def run(view):
    started=time.monotonic();frame='000995'
    base=old_root(frame,'production',view);peel=PEELED/frame/view
    _,original=verified_image(base,frame);_,layered=verified_image(peel,frame)
    for k in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert original[k]==layered[k]
    q=deepcopy(read(base/'request.json'));q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame]
    q['ordered_frame_ids']=[frame];q['source_rows']=[r for r in q['source_rows'] if Path(r['source_dataset']).name==frame]
    rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    assert receipt==read(peel/'request.json')['measured_source_visibility']['depth_receipt']
    source=base/'frames'/frame;after=peel/'frames'/frame
    before=np.asarray(Image.open(source/'prediction_native.png'));replacement=np.asarray(Image.open(after/'prediction_native.png'))
    sid=np.asarray(Image.open(source/'source_ids.png'));new_sid=np.asarray(Image.open(after/'source_ids.png'))
    depth=np.load(source/'target_depth.npz')['depth'];new_depth=np.load(after/'target_depth.npz')['depth']
    y,x=np.nonzero((depth>0)&(sid<62));chosen=sid[y,x]
    points=unproject(original['camera'],x,y,depth[y,x],offset=.5)
    prefilter=np.zeros(len(points),bool)
    for ci,(row,d) in enumerate(zip(rows,depths)):
        j=np.flatnonzero(chosen==ci)
        if not len(j):continue
        _,z,obs,ok=observed_at(points[j],row,d)
        prefilter[j]=ok&(obs>z+np.maximum(.005,.01*z))
    selected=np.flatnonzero(prefilter);p=points[selected]
    near=np.zeros(len(p),np.uint8);stable=np.zeros_like(near)
    for row,d in zip(rows,depths):
        _,z,obs,ok=observed_at(p,row,d);near+=ok&(np.abs(obs-z)<=.0015)
        uv,z=project_integer(row,p)
        free,_=free_space_evidence(d,uv[:,0],uv[:,1],z,minimum_gap=.005,near_gap=.0015)
        stable+=free
    candidates=np.flatnonzero((near==0)&(stable>=6));trusted=np.zeros(len(candidates),np.uint8)
    for row,d in zip(rows,depths):
        xy,z,obs,ok=observed_at(p[candidates],row,d);j=np.flatnonzero(ok&(obs>z+np.maximum(.005,.01*z)))
        if len(j):
            count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
            trusted[j]+=count>=3
    local=candidates[trusted>=6];take=selected[local]
    change=np.zeros(depth.shape,bool);change[y[take],x[take]]=True
    rgb=before.copy();rgb[change]=replacement[change]
    output_depth=depth.copy();output_depth[change]=new_depth[change]
    output_sid=sid.copy();output_sid[change]=new_sid[change]
    np.testing.assert_array_equal(rgb[~change],before[~change])
    np.testing.assert_array_equal(output_depth[~change],depth[~change])
    q.update(partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False,
        confidence_gated_peeling=dict(parent=str(base),parent_request_sha256=sha(base/'request.json'),
            layered_request_sha256=sha(peel/'request.json'),layered_complete_sha256=sha(after/'complete.json'),
            depth_receipt=receipt,selected_source_far_prefilter=True,zero_native_near_views=True,
            near_tolerance=.0015,minimum_stable_far_views=6,minimum_trusted_far_views=6,
            far_observation_other_views=3,minimum_far_gap=.005,minimum_relative_far_gap=.01,
            unchanged_outside_confident_free_points=True,uses_roi=False,uses_target_rgb=False,
            mesh_changed=False,geometry_repair_claim=False,no_rgb_averaging=True))
    q['script_hashes'][Path(__file__).name]=sha(__file__)
    out=ROOT/frame/view;out.mkdir(parents=True,exist_ok=False)
    target=out/'frames'/frame;target.mkdir(parents=True);atomic_json(out/'request.json',q)
    Image.fromarray(rgb).save(target/'prediction_native.png');Image.fromarray(np.rot90(rgb)).save(target/'frame.png')
    Image.fromarray(output_sid).save(target/'source_ids.png');np.savez_compressed(target/'target_depth.npz',depth=output_depth)
    np.savez_compressed(target/'confidence_evidence.npz',query_y=y,query_x=x,query_points=points,
        source_prefilter=prefilter,near_counts=near,stable_far_counts=stable,
        candidates=candidates,trusted_far_counts=trusted,changed_query_indices=take)
    r=deepcopy(original);r.update(render_sha256=sha(target/'frame.png'),rgb_coverage=float(rgb.any(2).mean()),
        visual_status='pending',elapsed_seconds=time.monotonic()-started,
        confidence_gated_peeling=dict(prefilter_pixels=len(selected),changed_confident_free_pixels=len(take),
            deeper_receiver_pixels=int((change&(new_depth>depth+1e-6)).sum()),
            changed_to_black=int((change&~rgb.any(2)&before.any(2)).sum()),
            unchanged_elsewhere=True,mesh_changed=False,counts_not_quality_metrics=True))
    atomic_json(target/'result.json',r)
    atomic_json(target/'complete.json',dict(request_sha256=sha(out/'request.json'),
        hashes={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='complete.json'}))
    print(view,r['confidence_gated_peeling'],'seconds',r['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',required=True,choices=VIEWS)
    run(p.parse_args().view)
