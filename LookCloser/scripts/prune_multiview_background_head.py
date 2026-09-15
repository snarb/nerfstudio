"""Opt-in head-fringe proposal: multiview clear background plus weak depth.

Applies the same rule to the entire normalized head, never a drawn crown ROI.
Foreground masks are fallible evidence, not certified geometry. A separate
render gate must check whether deletion exposes holes before any promotion.
"""
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,project
from transfer_close_boundary_completion import SOURCE,FRAMES,MOVIE
from refine_measured_head_masks import ROOT as MASKS
from probe_inset_head_completion import ROOT as INSET
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes

ROOT=Path('/mnt/data/dec5_multiview_background_head')
SETTINGS=dict(min_head_x=-.03,mask_radius_pixels=4,minimum_clear_background_cameras=12,
    minimum_preserving_depth_votes=2,depth_tolerance=.001,
    samples='three vertices, three edge midpoints, centroid',component_cleanup=False)


def triangle_samples(vertices,triangles):
    p=np.asarray(vertices)[np.asarray(triangles)]
    return np.concatenate([p,(p+p[:,[1,2,0]])*.5,p.mean(1)[:,None]],axis=1)


def deletion_rule(clear_background,depth_votes,minimum_background=12,minimum_preserving_votes=2):
    clear=np.asarray(clear_background);votes=np.asarray(depth_votes)
    if votes.ndim!=2 or votes.shape[1]!=7 or clear.shape!=(len(votes),):raise ValueError('Invalid seven-point evidence')
    if minimum_background<1 or minimum_preserving_votes<1 or not np.isfinite(votes).all():
        raise ValueError('Invalid confidence threshold/evidence')
    return (clear>=minimum_background)&(votes<minimum_preserving_votes).all(1)


def clear_background_votes(points,rows,masks,names):
    """One vote only if all seven projected samples have clear 9x9 background."""
    count=len(points);result=np.zeros((len(rows),count),bool)
    if len(names)!=len(set(names)) or set(names)!={r['physical_camera'] for r in rows}:raise ValueError('Camera mask mismatch')
    radius=SETTINGS['mask_radius_pixels'];kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
    flat=points.reshape(-1,3)
    for i,row in enumerate(rows):
        mask=masks[names.index(row['physical_camera'])]
        occupied=cv2.dilate(mask.astype(np.uint8),kernel)>0
        uv,z=project(flat,[row]);xy=np.rint(np.where(np.isfinite(uv[0]),uv[0],0)).astype(np.int64);z=z[0]
        valid=(z>0)&np.isfinite(z)&np.isfinite(uv[0]).all(1)
        valid&=(xy[:,0]>=radius)&(xy[:,0]<mask.shape[1]-radius)&(xy[:,1]>=radius)&(xy[:,1]<mask.shape[0]-radius)
        clear=np.zeros(len(flat),bool);clear[valid]=~occupied[xy[valid,1],xy[valid,0]]
        result[i]=clear.reshape(count,7).all(1)
    return result


def run(frame):
    folder=ROOT/frame;folder.mkdir(parents=True,exist_ok=True)
    base=read(SOURCE/frame/'request.json');entry=next(r for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    assert sha(base['source_mesh'])==base['source_mesh_sha256']==entry['mesh_sha256']
    mr=read(MASKS/frame/'result.json')
    for p,h in mr['hashes'].items():assert sha(MASKS/frame/p)==h
    names=read(MASKS/frame/'cameras.json');masks=np.load(MASKS/frame/'masks.npz')['masks']
    rows,depths,receipt=load_real(Path(base['depth_root']),frame);assert receipt==base['depth_receipt']
    inset=INSET/frame/'guarded';ir=read(inset/'result.json');assert ir['native_free_space_guard_passed']
    assert sha(inset/'mesh.ply')==ir['hashes']['mesh.ply']
    paths=[Path(__file__).resolve().with_name(n) for n in [Path(__file__).name,
        'study_jaw_depth_footprint.py','study_confidence_depth_prior.py',
        'joint_temporal_texture.py','diagnose_jaw_measured_depth.py']]
    request=dict(frame=frame,settings=SETTINGS,source_mesh=base['source_mesh'],source_mesh_sha256=base['source_mesh_sha256'],
        source_request_sha256=sha(SOURCE/frame/'request.json'),depth_receipt=receipt,
        masks_sha256=sha(MASKS/frame/'masks.npz'),mask_camera_sha256=sha(MASKS/frame/'cameras.json'),
        mask_result_sha256=sha(MASKS/frame/'result.json'),inset_mesh_sha256=sha(inset/'mesh.ply'),
        inset_result_sha256=sha(inset/'result.json'),scripts={str(p):sha(p) for p in paths},
        roi_used=False,target_camera_used=False,heldout_used=False,direct_RGB_read=False,masks_from_train_RGB=True,
        original_vertex_positions_preserved=True,all_original_triangles_preserved=False,
        refined_masks_are_fallible=True,combined_newly_exposed_shell_guard_passed=False,production_promoted=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=request:raise ValueError('Geometry request mismatch')
    if (folder/'result.json').exists():
        result=read(folder/'result.json')
        assert result['request_sha256']==sha(folder/'request.json')
        for p,h in result['hashes'].items():assert sha(folder/p)==h
        print(frame,'already complete; all current inputs revalidated',flush=True);return
    atomic_json(folder/'request.json',request)
    mesh=o3d.io.read_triangle_mesh(base['source_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    ids=np.flatnonzero((v[t][:,:,0] > SETTINGS['min_head_x']).all(1))
    points=triangle_samples(v,t[ids]);atomic_json(folder/'progress.json',dict(stage='clear_background',head_triangles=len(ids),unix_time=time.time()))
    clear=clear_background_votes(points,rows,masks,names);counts=clear.sum(0)
    eligible=np.flatnonzero(counts>=SETTINGS['minimum_clear_background_cameras'])
    atomic_json(folder/'progress.json',dict(stage='independent_depth',candidates=len(eligible),unix_time=time.time()))
    votes,refs=train_reference_votes(points[eligible].reshape(-1,3),rows,depths,tolerance=SETTINGS['depth_tolerance'])
    votes=votes.reshape(-1,7);refs=refs.reshape(-1,7)
    take=deletion_rule(counts[eligible],votes,SETTINGS['minimum_clear_background_cameras'],
        SETTINGS['minimum_preserving_depth_votes']);removed=ids[eligible[take]]
    keep=np.ones(len(t),bool);keep[removed]=False
    im=o3d.io.read_triangle_mesh(str(inset/'mesh.ply'));iv=np.asarray(im.vertices);it=np.asarray(im.triangles)
    np.testing.assert_array_equal(iv[:len(v)],v);np.testing.assert_array_equal(it[:len(t)],t)
    for name,vv,tt in [('pruned',v,t[keep]),('pruned_inset',iv,np.concatenate([t[keep],it[len(t):]]))]:
        out=folder/name;out.mkdir(exist_ok=True)
        candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt))
        candidate.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(out/'mesh.ply'),candidate)
        reread=o3d.io.read_triangle_mesh(str(out/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(reread.vertices),vv);np.testing.assert_array_equal(np.asarray(reread.triangles),tt)
    np.savez_compressed(folder/'evidence.npz',head_triangle_ids=ids,background_camera_votes=clear,
        eligible_head_indices=eligible,query_points=points[eligible],sample_depth_votes=votes,
        sample_depth_references=refs,removed_triangle_ids=removed,retained_original_triangle_ids=np.flatnonzero(keep))
    atomic_json(folder/'result.json',dict(frame=frame,request_sha256=sha(folder/'request.json'),head_triangles=len(ids),
        background_candidates=len(eligible),removed_triangles=len(removed),original_triangles=len(t),
        protected_by_any_two_view_sample=int((~take).sum()),inset_triangles=len(it)-len(t),
        hashes={n:sha(folder/n) for n in ['evidence.npz','pruned/mesh.ply','pruned_inset/mesh.ply']},
        geometry_is_hypothesis=True,combined_newly_exposed_shell_guard_passed=False,visual_status='pending',production_promoted=False))
    atomic_json(folder/'progress.json',dict(stage='geometry_complete',unix_time=time.time()))
    print(frame,'head',len(ids),'background candidates',len(eligible),'removed',len(removed),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=FRAMES);a=p.parse_args();run(a.frame)
