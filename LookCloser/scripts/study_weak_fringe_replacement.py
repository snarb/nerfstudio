"""Opt-in conservative old-fringe removal, with/without an inferred inset shell.

Uniform all-head rule; no diagnostic triangle IDs, train ROI, target or heldout
camera selects geometry. Protect ANY vertex/centroid with >=2 depth witnesses.
"""
import argparse
from pathlib import Path
from copy import deepcopy
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import maximum_filter
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from probe_inset_head_completion import ROOT as INSET, MASKS, SOURCE, MOVIE, FRAMES
from study_confidence_depth_prior import load_real,REGIONS
from study_jaw_depth_footprint import train_reference_votes
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for

ROOT=Path('/mnt/data/dec5_weak_fringe_replacement')


def removable(votes,background_count):
    votes=np.asarray(votes)
    if votes.ndim!=2 or votes.shape[1]!=4:raise ValueError('Need three vertices plus centroid')
    return (votes<2).all(1)&(np.asarray(background_count)>=6)


def background_witnesses(points,rows,masks,names):
    answer=[]
    for row in rows:
        uv,z=project(points.reshape(-1,3),[row]);uv,z=uv[0],z[0];xy=np.rint(uv).astype(int)
        valid=(z>0)&(xy[:,0]>=4)&(xy[:,0]<1916)&(xy[:,1]>=4)&(xy[:,1]<1076)
        occupied=maximum_filter(masks[names.index(row['physical_camera'])].astype(bool),size=9,mode='constant',cval=1)
        good=np.zeros(len(uv),bool);ids=np.flatnonzero(valid)
        good[ids]=~occupied[xy[ids,1],xy[ids,0]]
        answer.append(good.reshape(-1,4).all(1))
    return np.stack(answer)


def prepare(frame):
    root=ROOT/frame;root.mkdir(parents=True,exist_ok=False)
    base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
    assert receipt==base['depth_receipt'] and sha(base['source_mesh'])==base['source_mesh_sha256']
    old=o3d.io.read_triangle_mesh(base['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles)
    mr=read(MASKS/frame/'result.json')
    for p,h in mr['hashes'].items():assert sha(MASKS/frame/p)==h
    masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
    shellroot=INSET/frame/'guarded';sr=read(shellroot/'result.json')
    assert sr['native_free_space_guard_passed'] and sha(shellroot/'mesh.ply')==sr['hashes']['mesh.ply']
    shell=o3d.io.read_triangle_mesh(str(shellroot/'mesh.ply'));sv=np.asarray(shell.vertices);st=np.asarray(shell.triangles)
    np.testing.assert_array_equal(sv[:len(v)],v);np.testing.assert_array_equal(st[:len(t)],t)
    request=dict(frame=frame,source_mesh=base['source_mesh'],source_mesh_sha256=sha(base['source_mesh']),
        shell_mesh_sha256=sha(shellroot/'mesh.ply'),shell_result_sha256=sha(shellroot/'result.json'),
        depth_receipt=receipt,refined_masks_sha256=sha(MASKS/frame/'masks.npz'),
        parameters=dict(min_head_x=-.03,clear_background_radius_pixels=4,minimum_background_cameras=6,
                        protect_if_any_sample_depth_votes_ge=2,query_samples='three vertices and centroid',max_pruning_rounds=8),
        heldout_used=False,geometry_uses_target=False,manual_roi_used=False,production_updated=False,
        original_geometry_may_be_removed=True,original_vertices_never_moved=True,
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
                 [Path(__file__).name,'guard_jaw_measured_depth.py','study_jaw_depth_footprint.py','study_confidence_depth_prior.py']})
    atomic_json(root/'request.json',request)
    head=np.flatnonzero((v[t,:,][...,0]>-.03).all(1))
    points=np.concatenate([v[t[head]],v[t[head]].mean(1)[:,None]],axis=1)
    bg=background_witnesses(points,rows,masks,names);counts=bg.sum(0);candidate=np.flatnonzero(counts>=6)
    atomic_json(root/'progress.json',dict(stage='depth_support',head_triangles=len(head),candidates=len(candidate),unix_time=time.time()))
    votes,refs=train_reference_votes(points[candidate].reshape(-1,3),rows,depths);votes=votes.reshape(-1,4)
    remove_ids=head[candidate[removable(votes,counts[candidate])]]
    keep=np.ones(len(t),bool);keep[remove_ids]=False;kept=t[keep]
    np.savez_compressed(root/'evidence.npz',head_triangles=head,background_by_camera=bg,background_counts=counts,
                        candidate_indices=candidate,depth_votes=votes,depth_references=refs.reshape(-1,4),removed_triangles=remove_ids)
    print(frame,'head',len(head),'background candidates',len(candidate),'remove',len(remove_ids),flush=True)
    for arm in ['remove_only','replace']:
        folder=root/arm;folder.mkdir()
        vv=v if arm=='remove_only' else sv
        tt=kept if arm=='remove_only' else np.concatenate([kept,st[len(t):]])
        rounds=[];retained=np.arange(len(tt)-len(kept))
        # Removing old faces reveals previously occluded shell surfaces, so the
        # replacement MUST rerun native safety instead of reusing the old audit.
        if arm=='replace':
            for iteration in range(8):
                scene=scene_for(vv,tt);bad=set();checks=[]
                for ci,(row,d) in enumerate(zip(rows,depths)):
                    for offset in [0,.5]:
                        ids,count,raw_count=measured_pixel_veto(scene,row,d,rows,depths,len(kept),len(tt),offset)
                        bad.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
                    if (ci+1)%16==0:atomic_json(root/'progress.json',dict(stage='revealed_shell_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
                rounds.append(dict(removed_triangles=len(bad),checks=checks));print(frame,'replace guard',iteration,len(bad),flush=True)
                if not bad:break
                take=np.ones(len(tt),bool);take[list(bad)]=False;assert take[:len(kept)].all()
                retained=retained[take[len(kept):]];tt=tt[take]
            assert not rounds[-1]['removed_triangles'],'Revealed shell safety failed'
        result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt))
        result.compute_vertex_normals();o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),result)
        np.savez_compressed(folder/'retained.npz',original_triangle_ids=np.flatnonzero(keep),shell_triangle_ids=retained)
        atomic_json(folder/'result.json',dict(request_sha256=sha(root/'request.json'),arm=arm,
            removed_original_triangles=len(remove_ids),kept_original_triangles=len(kept),added_shell_triangles=len(retained),
            rounds=rounds,revealed_shell_guard_passed=True,vertices_moved=False,production_updated=False,
            hashes={n:sha(folder/n) for n in ['mesh.ply','retained.npz']},visual_status='pending'))
    atomic_json(root/'complete.json',dict(request_sha256=sha(root/'request.json'),evidence_sha256=sha(root/'evidence.npz'),
        results={arm:sha(root/arm/'result.json') for arm in ['remove_only','replace']},production_updated=False))


def render(frame):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    complete=read(ROOT/frame/'complete.json');assert sha(ROOT/frame/'request.json')==complete['request_sha256']
    rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
    for arm in ['remove_only','replace']:
        geometry=ROOT/frame/arm;g=read(geometry/'result.json')
        assert g['revealed_shell_guard_passed'] and sha(geometry/'mesh.ply')==g['hashes']['mesh.ply']
        for view in ['moving','native_unmasked']:
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
            if view=='native_unmasked':
                target=deepcopy(next(r for r in rows if r['physical_camera']==name))
                target['physical_camera']='diagnostic_unmasked_target_'+name;target['reference_physical_camera']=name;entry['camera']=target
            entry.update(mesh=str(geometry/'mesh.ply'),mesh_sha256=sha(geometry/'mesh.ply'))
            q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,weak_fringe_arm=arm,
                source_quality_implementation_sha256=implementation,fringe_request_sha256=sha(ROOT/frame/'request.json'),
                fringe_result_sha256=sha(geometry/'result.json'),native_target_mask_disabled=view=='native_unmasked',
                texture_source_masks_unchanged=True,inferred_not_measured_geometry=arm=='replace')
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            dest=ROOT/'rgb'/frame/view/arm;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
            if (dest/'request.json').exists() and read(dest/'request.json')!=q:raise ValueError('Changed fringe RGB request')
            atomic_json(dest/'request.json',q);engine.render(dest,[frame])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render'])
    p.add_argument('--frame',required=True,choices=FRAMES);a=p.parse_args()
    prepare(a.frame) if a.action=='prepare' else render(a.frame)
