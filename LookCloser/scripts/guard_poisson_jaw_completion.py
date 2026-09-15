"""Direct-depth versus bounded observed-anchor admission for a Poisson prior."""
from pathlib import Path
import argparse,time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from study_poisson_jaw_completion import OUT,SOURCE,FRAME
from study_jaw_repair_transfer import mask_votes,render
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from diagnose_jaw_measured_depth import barycentric_samples
from guard_jaw_measured_depth import initial_admission,measured_pixel_veto
from diffusion_mesh_repair import scene_for


def anchored_admission(strict, anchor_votes, free, support, outside):
    """Nearby observed anchors support inference, not direct depth at new points."""
    a=np.asarray(anchor_votes)
    if a.shape!=(len(strict),3):raise ValueError('Expected three original-surface anchors per triangle')
    inferred=(a>=3).all(1)&~np.asarray(free).any(axis=(0,2))&(np.asarray(support)>=2)&(np.asarray(outside)==0)
    return np.asarray(strict)|inferred


def prepare(root):
    result=read(root/'result.json');request=read(root/'request.json')
    if result['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed Poisson request')
    for p,h in result['hashes'].items():assert sha(root/p)==h
    base=read(SOURCE/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),FRAME)
    if receipt!=base['depth_receipt']:raise ValueError('Changed measured depths')
    mesh=o3d.io.read_triangle_mesh(str(root/'local_raw.ply'));v=np.asarray(mesh.vertices);tt=np.asarray(mesh.triangles)
    nt=result['original_triangles'];nv=result['original_vertices'];raw=np.load(root/'proposal_evidence.npz');proposals=raw['proposals']
    parent=read('/mnt/data/dec5_phase30_early_texture_dynamic_150/request.json');entry=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    maskroot=Path(entry['source_masks']['root']);assert sha(maskroot/'masks.npz')==base['source_mask_sha256']
    masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
    override=Path(base['mask_override']['root'])/FRAME;mr=read(override/'result.json')
    assert sha(override/'result.json')==base['mask_override']['result_sha256']
    for p,h in mr['hashes'].items():assert sha(override/p)==h
    masks=masks.copy();masks[names.index(mr['camera'])]=np.load(override/'mask.npy')
    evidence=root/'admission';evidence.mkdir(exist_ok=False)
    atomic_json(evidence/'request.json',dict(proposal_result_sha256=sha(root/'result.json'),
        source_request_sha256=sha(SOURCE/'request.json'),depth_receipt=receipt,script_sha256=sha(__file__),
        strict_rule='same two-view direct sample support',anchor_rule='all three nearest original-surface points have >=3 train-depth views',
        maximum_prior_distance=request['maximum_original_surface_distance'],ray_offsets=[0,.5],max_rounds=8,
        anchored_points_are_not_direct_depth_measurements=True,heldout_used=False,
        scripts={n:sha(Path(__file__).with_name(n)) for n in ['study_jaw_repair_transfer.py','study_jaw_depth_footprint.py','study_jaw_train_confidence.py','guard_jaw_measured_depth.py','study_confidence_depth_prior.py']}))
    ms,mo=mask_votes(v,proposals,rows,masks,names);semantic=np.flatnonzero((ms>=2)&(mo==0));pp=proposals[semantic]
    print('semantic',len(semantic),'of',len(proposals),flush=True)
    points=barycentric_samples(v[pp]);votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths)
    free=footprint_veto(points.reshape(-1,3),rows,depths).reshape(62,-1,10)
    closest_ids=pp-nv
    unique,inverse=np.unique(closest_ids,return_inverse=True)
    av,ar=train_reference_votes(raw['closest_points'][unique],rows,depths)
    anchors=av[inverse].reshape(-1,3)
    strict=initial_admission(votes.reshape(-1,10),free,ms[semantic],mo[semantic])
    anchored=anchored_admission(strict,anchors,free,ms[semantic],mo[semantic])
    np.savez_compressed(evidence/'samples.npz',semantic_ids=semantic,points=points,votes=votes.reshape(-1,10),references=refs.reshape(-1,10),
        free=free,mask_support=ms,mask_outside=mo,anchor_votes=anchors,strict=strict,anchored=anchored)
    atomic_json(evidence/'result.json',dict(request_sha256=sha(evidence/'request.json'),arrays_sha256=sha(evidence/'samples.npz'),
        semantic=len(semantic),strict=int(strict.sum()),anchored=int(anchored.sum())))
    for arm,keep in [('strict',strict),('anchored',anchored)]:
        folder=root/arm/FRAME;folder.mkdir(parents=True,exist_ok=False)
        atomic_json(folder/'request.json',dict(arm=arm,frame=FRAME,admission_result_sha256=sha(evidence/'result.json'),
            raw_result_sha256=sha(root/'result.json'),script_sha256=sha(__file__),source_mesh=base['source_mesh'],source_mesh_sha256=base['source_mesh_sha256']))
        ids=semantic[keep];t=np.concatenate([tt[:nt],proposals[ids]]);rounds=[]
        for iteration in range(8):
            scene=scene_for(v,t);remove=set();checks=[]
            for ci,(camera,depth) in enumerate(zip(rows,depths)):
                for offset in [0,.5]:
                    implicated,count,raw_count=measured_pixel_veto(scene,camera,depth,rows,depths,nt,len(t),offset)
                    remove.update(implicated.tolist());checks.append(dict(camera=camera['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw_count))
                if (ci+1)%10==0:atomic_json(folder/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
            rounds.append(dict(removed_triangles=len(remove),checks=checks));print(arm,'guard',iteration,'remove',len(remove),flush=True)
            if not remove:break
            take=np.ones(len(t),bool);take[list(remove)]=False
            if not take[:nt].all():raise ValueError('Original deletion')
            ids=ids[take[nt:]];t=t[take]
        if rounds[-1]['removed_triangles']:raise ValueError('Guard did not converge')
        saved=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));saved.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),saved)
        reread=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(reread.vertices),v);np.testing.assert_array_equal(np.asarray(reread.triangles)[:nt],tt[:nt])
        np.savez_compressed(folder/'evidence.npz',retained_proposal_ids=ids)
        atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),initially_admitted=int(keep.sum()),added=len(ids),rounds=rounds,
            original_vertices=nv,original_triangles=nt,observed_guard_passed=True,original_prefix_exact=True,
            hashes={p:sha(folder/p) for p in ['mesh.ply','evidence.npz']},visual_status='pending',production_accepted=False,
            new_surface_is_prior=True,nonmanifold_edges=len(saved.get_non_manifold_edges(allow_boundary_edges=True))))
        print(arm,'done',len(ids),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--arm',choices=['strict','anchored'],default='strict');a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    else:render(a.output/a.arm,FRAME)
