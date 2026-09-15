"""Continue both eight-round rejected screens with identical depth rules.

Only the convergence budget changes to 16 total rounds. Original attempts stay
immutable. The overlay contains hash-verified symlinks to frozen proposal data;
new guards, audits and RGB live separately. No extra triangles are admitted.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
import study_head_silhouette_completion as study
from study_confidence_depth_prior import load_real
from guard_jaw_measured_depth import measured_pixel_veto
from diffusion_mesh_repair import scene_for

SOURCE=study.ROOT
ROOT=Path('/mnt/data/dec5_head_silhouette_depth16')


def extend(frame):
    src,q,r=study.verify(frame);old=src/'guarded';g=read(old/'result.json');gq=read(old/'request.json')
    assert not g['native_free_space_guard_passed'] and len(g['rounds'])==8
    assert g['request_sha256']==sha(old/'request.json')
    for p,h in g['hashes'].items():assert sha(old/p)==h
    for p,h in gq['scripts'].items():assert sha(p)==h
    out=ROOT/frame;out.mkdir(parents=True,exist_ok=False)
    links=[src/'request.json',src/'result.json',src/'proposal_audit.json']+[src/p for p in r['hashes']]
    for p in links:(out/p.name).symlink_to(p.resolve(strict=True))
    dest=out/'guarded';dest.mkdir()
    b=read(study.SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(b['depth_root']),frame);assert receipt==gq['depth_receipt']
    original=o3d.io.read_triangle_mesh(q['source_mesh']);nt=len(original.triangles)
    mesh=o3d.io.read_triangle_mesh(str(old/'mesh.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    retained=np.load(old/'evidence.npz')['retained_candidate_triangle_ids'];rounds=list(g['rounds'])
    request=dict(frame=frame,prior_guard_result=str(old/'result.json'),prior_guard_result_sha256=sha(old/'result.json'),
        source_result_sha256=sha(src/'result.json'),depth_receipt=receipt,
        rule=dict(offsets=[0.,.5],free_separation=.003,minimum_other_views=3,maximum_rounds=16),
        unchanged_depth_rules=True,additional_admission=False,
        scripts={str(Path(__file__).resolve().with_name(n)):sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name,'study_head_silhouette_completion.py','guard_jaw_measured_depth.py','study_confidence_depth_prior.py']},
        frozen_proposal_links={str(p):sha(p) for p in links},production_updated=False)
    atomic_json(dest/'request.json',request)
    for iteration in range(8,16):
        scene=scene_for(v,t);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0.,.5]:
                ids,count,raw=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(t),offset)
                remove.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=count,raw_far_pixels=raw))
            if (ci+1)%10==0:atomic_json(dest/'progress.json',dict(stage='extended_native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed=len(remove),checks=checks));print(frame,'guard',iteration,'removed',len(remove),flush=True)
        if not remove:break
        keep=np.ones(len(t),bool);keep[list(remove)]=False;assert keep[:nt].all()
        retained=retained[keep[nt:]];t=t[keep]
    passed=not rounds[-1]['removed']
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(dest/'mesh.ply'),mesh)
    np.savez_compressed(dest/'evidence.npz',retained_candidate_triangle_ids=retained)
    atomic_json(dest/'result.json',dict(request_sha256=sha(dest/'request.json'),added=len(retained),rounds=rounds,
        original_prefix_exact=True,native_free_space_guard_passed=passed,production_updated=False,visual_status='pending',
        hashes={p:sha(dest/p) for p in ['mesh.ply','evidence.npz']}))
    assert passed,'Still not converged; no quality acceptance'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['extend','audit_guard','render','review'])
    p.add_argument('--frame',choices=study.FRAMES);a=p.parse_args()
    if a.action=='extend':
        if not a.frame:p.error('--frame required')
        extend(a.frame)
    else:
        study.ROOT=ROOT
        if a.action=='review':study.review()
        else:
            if not a.frame:p.error('--frame required')
            getattr(study,a.action)(a.frame)
