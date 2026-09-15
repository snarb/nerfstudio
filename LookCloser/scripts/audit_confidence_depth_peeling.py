"""Replay measured support, ray/mesh membership and hard-source fill colors."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,project,exr,display,ROOT as COLOR
from render_measured_depth_peeling import ROOT as PEELED,VIEWS,nearest_admitted
from render_confidence_gated_peeling import ROOT
from study_measured_source_visibility import ROOT as STRICT,old_root
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real,unproject,support,project_integer
from study_jaw_depth_footprint import train_reference_votes
from diagnose_jaw_measured_depth import observed_at
from carve_patchmatch_mesh_free_space import free_space_evidence
from review_jaw_repair_transfer import verified_image


def run(view):
    frame='000995';root=ROOT/frame/view;layer=PEELED/frame/view
    _,r=verified_image(layer,frame);_,cr=verified_image(root,frame)
    q=read(layer/'request.json');cq=read(root/'request.json')
    rows,depths,receipt=load_real(DEPTH_ROOT,frame)
    assert receipt==q['measured_source_visibility']['depth_receipt']==cq['confidence_gated_peeling']['depth_receipt']
    entry=q['inventory'][0];assert sha(entry['mesh'])==entry['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(entry['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    a=np.load(layer/'frames'/frame/'peeling_evidence.npz');chosen=a['selected_intersections']
    bary=a['barycentric'];weights=np.column_stack([1-bary.sum(1),bary])
    reconstructed=(v[t[a['faces']]]*weights[:,:,None]).sum(1)
    np.testing.assert_array_equal(reconstructed,a['points'])
    raypoints=a['rays'][a['ray_ids'],:3]+a['rays'][a['ray_ids'],3:]*a['distances'][:,None]
    # Float32 grazing intersections have small ray-vs-barycentric discrepancies.
    # Bound these by 1% of the actual TSDF voxel AND by 1/20 native pixel for
    # every displayed fill; neither bound changes an intersection or image.
    voxel=read(entry['metadata'])['parameters']['voxel_length']
    np.testing.assert_allclose(raypoints,reconstructed,rtol=0,atol=.01*voxel)
    target_uv,_=project_integer(entry['camera'],a['points'][chosen])
    py,px=np.unravel_index(a['query_pixels'][a['ray_ids'][chosen]],(1080,1920))
    reprojection=np.linalg.norm(target_uv-np.column_stack([px+.5,py+.5]),axis=1)
    assert reprojection.max(initial=0)<=.05
    counts,refs=train_reference_votes(a['points'],rows,depths)
    np.testing.assert_array_equal(counts,a['coherent_votes']);np.testing.assert_array_equal(refs,a['references'])
    expected=nearest_admitted(a['ray_ids'],a['distances'],(counts>=2)&(a['admitted_sources']>=0))
    np.testing.assert_array_equal(expected,chosen)
    assert (counts[chosen]>=2).all()
    selected_points=a['points'][chosen];sources=a['admitted_sources'][chosen].astype(int)
    uv,z=project(selected_points,rows);rounded=np.rint(uv);uv=np.where(np.abs(uv-rounded)<=.001,rounded,uv)
    actual=np.asarray(Image.open(layer/'frames'/frame/'prediction_native.png')).reshape(-1,3)[a['filled_pixels']]
    log_gain=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(log_gain-log_gain.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain'];predicted=np.zeros(actual.shape,np.uint8)
    source_checks=[]
    for ci in np.unique(sources):
        j=np.flatnonzero(sources==ci);xy=np.rint(uv[ci,j]+.5).astype(int);obs=depths[ci][xy[:,1],xy[:,0]]
        error=np.abs(obs-z[ci,j]);assert (obs>0).all() and (error<=.001502).all()
        f=uv[ci,j];x0=np.floor(f[:,0]).astype(int);y0=np.floor(f[:,1]).astype(int)
        fraction=f-np.floor(f);rgb=exr(rows[ci]['file_path']);sample=np.zeros((len(j),3),np.float32)
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            w=(fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1])
            sample+=rgb[y0+dy,x0+dx]*w[:,None]
            native=np.rint(np.column_stack([x0+dx,y0+dy])+.5).astype(int)
            tap=depths[ci][native[:,1],native[:,0]];active=w>0
            assert (tap[active]>0).all() and (np.abs(tap[active]-z[ci,j][active])<=.001502).all()
        predicted[j]=np.rint(display(np.maximum(sample*gain[ci],0),exposure)*255).clip(0,255).astype(np.uint8)
        source_checks.append(dict(camera=rows[ci]['physical_camera'],filled_pixels=len(j),
            maximum_measured_depth_error=float(error.max(initial=0)),source_rgb_sha256=sha(rows[ci]['file_path'])))
    rgb_error=np.abs(actual.astype(int)-predicted.astype(int));assert rgb_error.max(initial=0)<=1
    # Replay every conditional change: absent near layer AND positive evidence
    # of free space. This reuses established depth-projection/support helpers.
    c=np.load(root/'frames'/frame/'confidence_evidence.npz');ix=c['changed_query_indices'];p=c['query_points'][ix]
    near=np.zeros(len(p),int);far=np.zeros_like(near);trusted=np.zeros_like(near)
    for row,d in zip(rows,depths):
        xy,zp,obs,ok=observed_at(p,row,d);near+=ok&(np.abs(obs-zp)<=.0015)
        u,zp=project_integer(row,p);stable,_=free_space_evidence(d,u[:,0],u[:,1],zp,minimum_gap=.005,near_gap=.0015)
        far+=stable;j=np.flatnonzero(ok&(obs>zp+np.maximum(.005,.01*zp)))
        count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
        trusted[j]+=count>=3
    assert (near==0).all() and (far>=6).all() and (trusted>=6).all()
    old=old_root(frame,'production',view);before=np.asarray(Image.open(old/'frames'/frame/'prediction_native.png'))
    conditional=np.asarray(Image.open(root/'frames'/frame/'prediction_native.png'))
    change=np.zeros(before.shape[:2],bool);change[c['query_y'][ix],c['query_x'][ix]]=True
    np.testing.assert_array_equal(conditional[~change],before[~change])
    np.testing.assert_array_equal(conditional[change],np.asarray(Image.open(layer/'frames'/frame/'prediction_native.png'))[change])
    atomic_json(root/'audit.json',dict(view=view,ray_and_mesh_membership_verified=True,
        coherent_depth_support_replayed=True,reference_helpers_reused=True,
        selected_nearest_admitted_intersections_verified=True,filled_pixels=len(chosen),
        all_filled_rgb_reprojected_from_one_train_source=True,
        max_uint8_rgb_replay_difference=int(rgb_error.max(initial=0)),
        depth_numerical_replay_slack=2e-6,source_checks=source_checks,
        ray_world_component_error_max=float(np.abs(raypoints-reconstructed).max(initial=0)),
        ray_world_tolerance=.01*voxel,filled_target_reprojection_max_px=float(reprojection.max(initial=0)),
        filled_target_reprojection_tolerance_px=.05,
        conditional_changed_points=len(p),all_conditional_changes_measured_free=True,
        unchanged_outside_confident_points=True,source_request_sha256=sha(layer/'request.json'),
        conditional_request_sha256=sha(root/'request.json'),script_sha256=sha(__file__),
        production_promoted=False,quality_approval=False))
    print(view,'audit passed; filled',len(chosen),'conditional',len(p),'RGB max LSB',rgb_error.max(initial=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',required=True,choices=VIEWS)
    run(p.parse_args().view)
