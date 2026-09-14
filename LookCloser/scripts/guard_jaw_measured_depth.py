"""Experimental observed-depth gate for bounded camera-independent notch caps.

No old-mesh occlusion veto: independently corroborated QUERY-camera depth is
required to declare free space. Original geometry is preserved byte-for-byte
as arrays. This pilot is opt-in and never replaces production artifacts.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from study_confidence_depth_prior import load_real, support, unproject
from study_jaw_boundary_notches import PARENT, PHASE
from diagnose_jaw_measured_depth import RAW, ATTRIBUTION
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def initial_admission(votes, trusted_free, mask_support, mask_outside):
    return ((votes[:,:3]>=2).sum(1)>=2) & (np.median(votes,axis=1)>=2) & \
        ~trusted_free.any(axis=(0,2)) & (mask_support>=2) & (mask_outside==0)


def measured_pixel_veto(scene, camera, observed, rows, depths, original_count, total_count, offset):
    actual=dict(camera)
    if offset==0: actual['cx']+=.5;actual['cy']+=.5
    d,ids,_=camera_depth(scene,actual)
    y,x=np.nonzero(np.isfinite(d)&(ids>=original_count)&(ids<total_count))
    # Align each ray with its nearest native integer COLMAP pixel.
    qx=np.rint(x+offset).astype(int);qy=np.rint(y+offset).astype(int)
    valid=(qx<1920)&(qy<1080);x,y,qx,qy=x[valid],y[valid],qx[valid],qy[valid]
    obs=observed[qy,qx]
    far=np.isfinite(obs)&(obs>0)&(obs>d[y,x]+.003)
    x,y,qx,qy,obs=x[far],y[far],qx[far],qy[far],obs[far]
    votes,_=support(unproject(camera,qx,qy,obs),camera,rows,depths)
    bad=votes>=3
    return np.unique(ids[y[bad],x[bad]]).astype(int),int(bad.sum()),len(x)


def run(root,frame):
    out=root/'guarded'/frame;out.mkdir(parents=True,exist_ok=True)
    diagnosis=root/'analysis'/frame; analysis=read(diagnosis/'result.json')
    if analysis['evidence_sha256']!=sha(diagnosis/'evidence.npz'):raise ValueError('Changed sample evidence')
    arrays=np.load(diagnosis/'evidence.npz'); semantic=np.load(ATTRIBUTION/frame/'admission.npz')
    source=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    mesh=o3d.io.read_triangle_mesh(source['mesh']);raw=o3d.io.read_triangle_mesh(str(RAW/frame/'candidate.ply'))
    v,t,rt=np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.asarray(raw.triangles)
    raw_record=read(RAW/frame/'result.json')
    if sha(source['mesh'])!=raw_record['source_mesh_sha256'] or sha(RAW/frame/'candidate.ply')!=raw_record['candidate_sha256']:raise ValueError('Changed raw proposal')
    if not np.array_equal(v,np.asarray(raw.vertices)) or not np.array_equal(t,rt[:len(t)]):raise ValueError('Changed prefix')
    request=dict(frame=frame,script_sha256=sha(__file__),diagnosis_sha256=sha(diagnosis/'result.json'),
        semantic_evidence_sha256=sha(ATTRIBUTION/frame/'admission.npz'),raw_mesh_sha256=sha(RAW/frame/'candidate.ply'),
        parent_request_sha256=sha(PARENT/'request.json'),phase_request_sha256=sha(PHASE/'request.json'),
        rule=dict(min_vertices_with_ge2_votes=2,median_sample_votes=2,min_mask_support=2,no_available_mask_veto=True,
                  max_trusted_free_samples=0,free_depth_separation=.003,other_observed_support=3,
                  ray_offsets=[0,.5],max_pruning_passes=8),production_changed=False,geometry_is_inferred=True)
    if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Frozen gate mismatch')
    atomic_json(out/'request.json',request)
    if (out/'result.json').exists():
        result=read(out/'result.json')
        if result['request_sha256']!=sha(out/'request.json') or result['mesh_sha256']!=sha(out/'mesh.ply'):raise ValueError('Changed completed gate')
        print(frame,'verified completed gate',flush=True);return
    keep=initial_admission(arrays['votes'],arrays['trusted_free'],semantic['support'],semantic['outside'])
    proposal_ids=np.flatnonzero(keep);triangles=np.concatenate((t,rt[len(t):][keep]));initial=len(proposal_ids)
    rows,depths,receipt=load_real(root/'analysis',frame);rounds=[]
    for iteration in range(8):
        scene=scene_for(v,triangles);remove=set();checks=[]
        for camera,depth in zip(rows,depths):
            for offset in [0,.5]:
                implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,len(t),len(triangles),offset)
                remove.update(implicated.tolist())
                checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
        rounds.append(dict(iteration=iteration,removed_triangles=len(remove),checks=checks))
        print(frame,'pruning_pass',iteration,'removed',len(remove),flush=True)
        if not remove:break
        take=np.ones(len(triangles),bool);take[list(remove)]=False
        if not take[:len(t)].all():raise ValueError('Guard attempted original deletion')
        proposal_ids=proposal_ids[take[len(t):]];triangles=triangles[take]
    passed=not rounds[-1]['removed_triangles']
    if not np.array_equal(triangles[:len(t)],t):raise ValueError('Original triangles changed')
    final=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(triangles));final.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'mesh.ply'),final)
    reread=o3d.io.read_triangle_mesh(str(out/'mesh.ply'))
    if not np.array_equal(np.asarray(reread.vertices),v) or not np.array_equal(np.asarray(reread.triangles)[:len(t)],t):raise ValueError('Saved prefix mismatch')
    scene=scene_for(v,triangles);oldscene=scene_for(v,t)
    spot=next(r for r in read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components'] if r['frame_id']==frame)
    x0,y0,x1,y1=spot['bbox_inclusive'];phase=next(r for r in read(PHASE/'request.json')['inventory'] if r['frame_id']==frame)
    moving=[]
    for label,camera in [('old_moving',source['camera']),('phase_moving',phase['camera'])]:
        d,ids,_=camera_depth(scene,camera);od,_,_=camera_depth(oldscene,camera);valid=np.isfinite(d)
        lighting=np.abs(np.asarray(final.triangle_normals)@np.array([.3,.4,.866]));rgb=np.zeros((*ids.shape,3),np.uint8)
        rgb[valid]=(60+170*lighting[ids[valid],None]).astype(np.uint8);rgb[valid&(ids>=len(t))]=[240,60,50]
        portrait=Image.fromarray(np.rot90(rgb));portrait.save(out/f'{label}_added.png')
        rec=dict(camera=label,added_surface_pixels=int((valid&(ids>=len(t))).sum()),newly_visible=int((valid&~np.isfinite(od)).sum()))
        if label=='old_moving':
            rec['selected_spot_misses']=int((~np.rot90(valid)[y0:y1+1,x0:x1+1]).sum())
            portrait.crop((x0-65,y0-65,x1+66,y1+66)).save(out/'spot_added_native.png')
        moving.append(rec)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),mesh_sha256=sha(out/'mesh.ply'),
        initial_admitted=initial,final_added=len(proposal_ids),retained_proposal_ids=proposal_ids.tolist(),
        rounds=rounds,observed_free_space_guard_passed=passed,original_vertex_triangle_prefix_preserved=True,
        moving=moving,visual_status='pending',production_accepted=False,depth_receipt=receipt))
    print(frame,'done',len(proposal_ids),passed,moving,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_measured_depth'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);args=p.parse_args();run(args.root,args.frame)
