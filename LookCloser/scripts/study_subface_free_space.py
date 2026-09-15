"""Isolated two-round subdivision control with frozen native-depth deletion gates.

Only depth-conflicted parent faces select refinement, across the whole mesh.
No RGB/ROI or learned prior chooses vertices or deletions. Production is untouched.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from ablate_free_surface_near_footprint import ROOT as PARENT
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real, project_integer, support, unproject
from diagnose_jaw_measured_depth import observed_at
from prune_measured_free_surface import near_tap_evidence, removable
from carve_patchmatch_mesh_free_space import free_space_evidence
from subdivide_conflicted_surface import subdivide, verify_coverage

ROOT = Path('/mnt/data/dec5_subface_free_space')
FRAME = '000995'


def write_mesh(path, vertices, triangles):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    actual = o3d.io.read_triangle_mesh(str(path))
    np.testing.assert_array_equal(np.asarray(actual.vertices), vertices)
    np.testing.assert_array_equal(np.asarray(actual.triangles), triangles)
    _, components, _ = actual.cluster_connected_triangles()
    return sorted(map(int, components), reverse=True)


def geometry():
    parent = PARENT/FRAME
    pq, pr = read(parent/'request.json'), read(parent/'result.json')
    assert pr['request_sha256'] == sha(parent/'request.json')
    assert sha(pq['mesh']) == pq['mesh_sha256']
    for path, digest in pq['scripts'].items(): assert sha(path) == digest
    for name, digest in pr['hashes'].items(): assert sha(parent/name) == digest
    previous = np.load(parent/'evidence.npz')
    mesh = o3d.io.read_triangle_mesh(pq['mesh'])
    ov, ot = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    np.testing.assert_array_equal(previous['points'][:len(ov)], ov)
    np.testing.assert_array_equal(previous['sample_indices'][:, :3], ot)
    old_near = previous['near_counts'][previous['sample_indices']]
    old_far = previous['stable_far_counts'][previous['sample_indices']]
    selected = (old_near > 0).any(1) & (old_far >= 6).any(1)
    root = ROOT/FRAME; root.mkdir(parents=True, exist_ok=False)
    q = deepcopy(pq)
    q.update(parent_request_sha256=sha(parent/'request.json'), parent_evidence_sha256=sha(parent/'evidence.npz'),
             subdivision_rounds=2, subdivision_selection='any sample near > 0 AND any sample stable_far >= 6',
             selected_parent_count=int(selected.sum()), geometry_uses_roi=False, geometry_uses_rgb=False,
             selected_parameters_unchanged=True, production_changed=False)
    q['scripts'].update({str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in
                        [Path(__file__).name, 'subdivide_conflicted_surface.py']})
    atomic_json(root/'request.json', q)
    v,t,p = ov.copy(),ot.copy(),np.arange(len(ot))
    rounds=[]
    for iteration in range(2):
        v,t,p = subdivide(v,t,selected[p],p)
        check = verify_coverage(ov,ot,v,t,p)
        rounds.append(dict(round=iteration+1, vertices=len(v), triangles=len(t), checks=check))
    np.savez_compressed(root/'subdivision.npz', vertices=v, triangles=t, parents=p, selected_parents=selected)
    rows,depths,receipt = load_real(DEPTH_ROOT, FRAME)
    assert receipt == pq['depth_receipt']
    points=np.concatenate([v, v[t].mean(1)])
    samples=np.column_stack([t,np.arange(len(t))+len(v)])
    near=np.zeros(len(points),np.uint8); far=np.zeros_like(near)
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        uv,z=project_integer(row,points)
        near+=near_tap_evidence(depth,uv,z,radius=0)
        stable,_=free_space_evidence(depth,uv[:,0],uv[:,1],z,minimum_gap=.005,near_gap=.0015)
        far+=stable
        atomic_json(root/'progress.json',dict(stage='all_subface_footprints',camera=ci+1,time=time.time()))
    candidates=np.flatnonzero((near[samples]==0).all(1)&(far[samples]>=6).all(1))
    queries=points[samples[candidates]].reshape(-1,3)
    trusted=np.zeros((len(rows),len(candidates),4),bool)
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        xy,z,obs,ok=observed_at(queries,row,depth)
        j=np.flatnonzero(ok&(obs>z+np.maximum(.005,.01*z)))
        if len(j):
            count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
            trusted[ci].reshape(-1)[j]=count>=3
        atomic_json(root/'progress.json',dict(stage='far_corroboration',camera=ci+1,candidates=len(candidates),time=time.time()))
    remove=candidates[removable(near[samples[candidates]],far[samples[candidates]],trusted.sum(0))]
    keep=np.ones(len(t),bool); keep[remove]=False
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,near_counts=near,
                        stable_far_counts=far,candidates=candidates,trusted_far_by_camera=trusted,removed_triangle_ids=remove)
    records=[]
    for control,triangles in [('refined',t),('pruned',t[keep])]:
        dest=ROOT/control/FRAME;dest.mkdir(parents=True,exist_ok=False)
        request=deepcopy(q);request.update(control=control,study_request_sha256=sha(root/'request.json'))
        atomic_json(dest/'request.json',request)
        components=write_mesh(dest/'mesh.ply',v,triangles)
        result=dict(request_sha256=sha(dest/'request.json'),control=control,
                    hashes={'mesh.ply':sha(dest/'mesh.ply')},vertices=len(v),triangles=len(triangles),
                    components=components,production_changed=False,visual_status='pending',
                    no_vertex_displacement=True,no_component_cleanup=True)
        atomic_json(dest/'result.json',result);records.append(result)
    area=np.linalg.norm(np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]),axis=1)/2
    removed_area=np.bincount(p[remove],weights=area[remove],minlength=len(ot))
    total_area=np.bincount(p,weights=area,minlength=len(ot))
    np.savez_compressed(root/'parent_removal.npz', removed_area=removed_area,total_area=total_area)
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),rounds=rounds,
        original_triangles=len(ot),refined_triangles=len(t),selected_parents=int(selected.sum()),
        candidate_subfaces=len(candidates),removed_subfaces=len(remove),
        affected_parents=int((removed_area>0).sum()),partially_removed_parents=int(((removed_area>0)&(removed_area<total_area-1e-15)).sum()),
        removed_area=float(removed_area.sum()),controls=records,
        hashes={n:sha(root/n) for n in ['subdivision.npz','evidence.npz','parent_removal.npz']},
        visual_status='pending',production_changed=False))
    print('subface study',len(ot),'->',len(t),'removed',len(remove),'selected parents',selected.sum(),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['geometry','prepare','render'])
    parser.add_argument('--control',choices=['refined','pruned'])
    parser.add_argument('--view')
    args=parser.parse_args()
    if args.action=='geometry': geometry()
    else:
        if not args.control: parser.error('Control required')
        import review_measured_free_surface as workflow
        workflow.ROOT=ROOT/args.control
        if args.action=='prepare': workflow.prepare(FRAME)
        else:
            if args.view not in workflow.VIEWS:parser.error('Known view required')
            workflow.render(FRAME,args.view)
