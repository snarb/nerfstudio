"""A weaker observed-neighborhood prior; keep all measured free-space vetoes."""
import argparse
from pathlib import Path
from copy import deepcopy
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from joint_temporal_texture import read,sha,atomic_json
import study_body_neighborhood_completion as base
from body_surface_neighborhood import select_seeds
from local_surface_certificate import certify
from diffusion_mesh_repair import scene_for
from guard_jaw_measured_depth import measured_pixel_veto

ROOT=Path('/mnt/data/dec5_body_single_depth_seed')


def prepare(frame):
    src=base.ROOT/frame;out=ROOT/frame;out.mkdir(parents=True,exist_ok=False)
    req=read(src/'request.json');result=read(src/'result.json')
    for n,h in result['hashes'].items():assert sha(src/n)==h
    rows,depths,hashes,_,_=base.load_inputs(frame);assert hashes==req['source_depth_sha256']
    request=deepcopy(req);request['settings']['minimum_seed_depth_views']=1
    request.update(parent_result_sha256=sha(src/'result.json'),parent_admission_sha256=sha(src/'admission.npz'),
        weaker_seed_evidence_not_multiview_certainty=True,script_sha256=sha(__file__))
    atomic_json(out/'request.json',request)
    old=o3d.io.read_triangle_mesh(req['source_mesh']);old.compute_vertex_normals();v=np.asarray(old.vertices);t=np.asarray(old.triangles);nt=len(t)
    p=np.load(src/'proposal.npz');mv=p['vertices'];proposals=p['proposals'];e=np.load(src/'admission.npz')
    q=mv[e['query_ids']];normals=e['query_normals'];seed_ids=e['seed_ids']
    valid=(e['seed_votes']>=1)&(e['seed_poisson_distance']<=.0005)
    seeds=v[seed_ids[valid]];sn=np.asarray(old.vertex_normals)[seed_ids[valid]];certificate=np.zeros(len(q),bool);notes=[];neighbors=[]
    for i,ids in enumerate(cKDTree(seeds).query_ball_point(q,.006)):
        ids=np.asarray(ids,int);chosen=ids[select_seeds(q[i],normals[i],seeds[ids],sn[ids])]
        certificate[i],note=certify(q[i],normals[i],seeds[chosen]);notes.append(note);neighbors.append(chosen.tolist())
    lookup=np.zeros(len(mv),bool);lookup[e['query_ids']]=certificate
    prior=lookup[proposals[e['semantic_ids']]].all(1)&~e['free'].any(axis=(0,2))
    selected=e['semantic_ids'][e['strict']|prior];triangles=np.concatenate([t,proposals[selected]]);initial=len(selected)
    np.savez_compressed(out/'certificate_evidence.npz',valid_seed_mask=valid,certificate=certificate,prior=prior)
    atomic_json(out/'certificates.json',dict(notes=notes,seed_neighbors=neighbors))
    print(frame,'one-depth seeds',len(seeds),'certified vertices',int(certificate.sum()),'initial triangles',initial,flush=True)
    rounds=[]
    for iteration in range(8):
        scene=scene_for(mv,triangles);remove=set();checks=[]
        for ci,(row,depth) in enumerate(zip(rows,depths)):
            for offset in [0,.5]:
                ids,n,raw_n=measured_pixel_veto(scene,row,depth,rows,depths,nt,len(triangles),offset)
                remove.update(ids.tolist());checks.append(dict(camera=row['physical_camera'],offset=offset,trusted_free_pixels=n,raw_far_pixels=raw_n))
            if (ci+1)%10==0:atomic_json(out/'progress.json',dict(stage='native_guard',iteration=iteration,cameras=ci+1,unix_time=time.time()))
        rounds.append(dict(removed=len(remove),checks=checks));print(frame,'guard',iteration,len(remove),flush=True)
        if not remove:break
        take=np.ones(len(triangles),bool);take[list(remove)]=False;assert take[:nt].all()
        selected=selected[take[nt:]];triangles=triangles[take]
    if rounds[-1]['removed']:raise ValueError('Guard did not converge')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mv),o3d.utility.Vector3iVector(triangles));mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'mesh.ply'),mesh);np.savez_compressed(out/'retained.npz',proposal_ids=selected)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),added=len(selected),initial=initial,
        seeds=len(seeds),certified_vertices=int(certificate.sum()),rounds=rounds,observed_guard_passed=True,
        original_vertices=len(v),original_triangles=nt,production_accepted=False,visual_status='pending',
        hashes={n:sha(out/n) for n in ['mesh.ply','retained.npz','certificate_evidence.npz','certificates.json']}))
    print(frame,'completed',len(selected),flush=True)


def render(frame):
    base.ROOT=ROOT;base.SCRIPTS=base.SCRIPTS+[Path(__file__).name];base.render(frame)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render'])
    p.add_argument('--frame',choices=['001029','001033','001037'],required=True);a=p.parse_args()
    {'prepare':prepare,'render':render}[a.action](a.frame)
