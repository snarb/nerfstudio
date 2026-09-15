"""Opt-in local-normal consensus screen for inferred head completion.

Only rescue nearest-facet normal failures; keep distance, edge and all-camera
silhouette conditions unchanged. This is an unchecked geometry proposal, not a
measured-depth certificate or production mesh. Diagnostic regions do not select
triangles. Original and already guarded triangles are preserved exactly.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, cameras
from probe_inset_head_completion import ROOT as INSET, RAW, FRAMES, MOVIE, inset_vertices
from refine_measured_head_masks import ROOT as MASKS
from study_jaw_repair_transfer import mask_votes
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from study_confidence_depth_prior import REGIONS, region_masks
from review_jaw_repair_transfer import panel

ROOT = Path('/mnt/data/dec5_consensus_head_normals')
SETTINGS = dict(neighbors=32, radius=.003, minimum_neighbors=8,
                minimum_coherence=.5, minimum_weighted_agreement=.7,
                minimum_normal_dot=.25, weight_sigma=.0015)


def consensus(points, directions, centers, normals, areas, settings=SETTINGS):
    """Bounded-area, distance-weighted oriented consensus; no sign flipping."""
    if len(centers) < settings['minimum_neighbors']:
        raise ValueError('Insufficient reference facets')
    tree = cKDTree(centers)
    arrays = [[] for _ in range(5)]
    for start in range(0, len(points), 8192):
        stop = min(start + 8192, len(points))
        distance, idx = tree.query(points[start:stop], k=min(settings['neighbors'], len(centers)))
        valid = distance <= settings['radius']
        a = areas[idx]
        # A single oversized facet cannot dominate the neighborhood.
        cap = np.median(a, axis=1)
        weights = np.minimum(a, cap[:, None]) * np.exp(-.5*(distance/settings['weight_sigma'])**2) * valid
        total = weights.sum(1)
        vector = np.einsum('nk,nkj->nj', weights, normals[idx])
        length = np.linalg.norm(vector, axis=1)
        coherence = length / np.maximum(total, 1e-30)
        dot = np.sum(vector * directions[start:stop], axis=1) / np.maximum(length, 1e-30)
        votes = np.einsum('nkj,nj->nk', normals[idx], directions[start:stop]) >= settings['minimum_normal_dot']
        agreement = (weights * votes).sum(1) / np.maximum(total, 1e-30)
        count = valid.sum(1)
        good = ((count >= settings['minimum_neighbors']) & (coherence >= settings['minimum_coherence']) &
                (dot >= settings['minimum_normal_dot']) & (agreement >= settings['minimum_weighted_agreement']))
        for target, value in zip(arrays, [good, coherence, dot, agreement, count]): target.append(value)
    return [np.concatenate(x) for x in arrays]


def run(frame):
    out = ROOT/frame; out.mkdir(parents=True, exist_ok=False)
    config = read(INSET/frame/'request.json')
    assert sha(config['source_mesh']) == config['source_mesh_sha256']
    assert sha(RAW/frame/'poisson_raw.ply') == config['raw_mesh_sha256']
    guard = INSET/frame/'guarded'; gr = read(guard/'result.json')
    assert gr['native_free_space_guard_passed']
    for p, h in gr['hashes'].items(): assert sha(guard/p) == h
    mr = read(MASKS/frame/'result.json')
    for p, h in mr['hashes'].items(): assert sha(MASKS/frame/p) == h
    atomic_json(out/'request.json', dict(frame=frame, settings=SETTINGS, source_config_sha256=sha(INSET/frame/'request.json'),
        baseline_guard_result_sha256=sha(guard/'result.json'), masks_result_sha256=sha(MASKS/frame/'result.json'),
        script_sha256=sha(__file__), helpers={n:sha(Path(__file__).with_name(n)) for n in
        ['probe_inset_head_completion.py','study_jaw_repair_transfer.py','bake_joint_temporal_mesh.py']},
        inferred_not_measured=True, geometry_uses_target=False, production_updated=False))
    original = o3d.io.read_triangle_mesh(config['source_mesh']); original.compute_triangle_normals()
    ov, ot = np.asarray(original.vertices), np.asarray(original.triangles)
    raw = o3d.io.read_triangle_mesh(str(RAW/frame/'poisson_raw.ply')); raw.compute_vertex_normals()
    rv = inset_vertices(np.asarray(raw.vertices), np.asarray(config['center']), .001)
    rt = np.asarray(raw.triangles); rn = np.asarray(raw.vertex_normals)
    closest = scene_for(ov,ot).compute_closest_points(o3d.core.Tensor(rv.astype(np.float32)))
    dist = np.linalg.norm(rv-closest['points'].numpy(),axis=1)
    edges,n = np.unique(np.sort(ot[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    bd = cKDTree(ov[np.unique(edges[n==1])]).query(rv)[0]
    dot = np.sum(rn*np.asarray(original.triangle_normals)[closest['primitive_ids'].numpy()],axis=1)
    geometric = (dist<=.006)&(bd<=.006)&(rv[:,0]>-.03)
    length = np.linalg.norm(rv[rt]-rv[rt[:,[1,2,0]]],axis=2).max(1)
    old_eligible = geometric & (dot>=.25)
    existing = np.load(INSET/frame/'inset_001000'/'evidence.npz')['proposal_ids']
    np.testing.assert_array_equal(np.flatnonzero(old_eligible[rt].all(1)&(length<=.0015)),existing)
    eligible = np.flatnonzero(geometric[rt].all(1)&(length<=.0015)&~(dot[rt]>=.25).all(1))
    vertices = np.unique(rt[eligible]); head = (ov[ot][:,:,0]>-.03).all(1)
    centers = ov[ot[head]].mean(1); normals = np.asarray(original.triangle_normals)[head]
    areas = .5*np.linalg.norm(np.cross(ov[ot[head,1]]-ov[ot[head,0]],ov[ot[head,2]]-ov[ot[head,0]]),axis=1)
    good, coherence, normal_dot, agreement, count = consensus(rv[vertices],rn[vertices],centers,normals,areas)
    accepted = np.zeros(len(rv),bool); accepted[vertices] = good
    # The rescue uses consensus at every vertex, not just the one that failed.
    rescued = eligible[accepted[rt[eligible]].all(1)]
    rows,_,_ = cameras(frame)
    masks = np.load(MASKS/frame/'masks.npz')['masks']; names = read(MASKS/frame/'cameras.json')
    support,outside = mask_votes(rv,rt[rescued],rows,masks,names)
    retained = rescued[(support>=2)&(outside==0)]
    baseline = o3d.io.read_triangle_mesh(str(guard/'mesh.ply'))
    v,t = np.asarray(baseline.vertices),np.asarray(baseline.triangles)
    np.testing.assert_array_equal(v[len(ov):],rv)
    tt = np.concatenate([t,rt[retained]+len(ov)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(tt)); mesh.compute_triangle_normals()
    assert o3d.io.write_triangle_mesh(str(out/'unchecked_mesh.ply'),mesh)
    native = next(r for r in rows if r['physical_camera']==REGIONS[frame]['camera'])
    moving = next(r['camera'] for r in read(MOVIE/'request.json')['inventory'] if r['frame_id']==frame)
    scenes = [scene_for(v,t),scene_for(v,tt)]; records=[]
    for name,cam in [('native_train',native),('moving',moving)]:
        depths = [camera_depth(s,cam) for s in scenes]; old = np.isfinite(depths[0][0]); now = np.isfinite(depths[1][0])
        gain = ~old&now; rgb = np.zeros((*old.shape,3),np.uint8)
        shade = np.abs(np.asarray(mesh.triangle_normals)@np.array([.3,.4,.866]))
        rgb[now]=(60+170*shade[depths[1][1][now],None]).astype(np.uint8); rgb[gain]=[255,70,70]
        Image.fromarray(np.rot90(rgb).copy()).save(out/(name+'.png'))
        np.savez_compressed(out/(name+'_depth.npz'), baseline=depths[0][0], candidate=depths[1][0], triangle_ids=depths[1][1])
        record=dict(view=name,gained_depth=int(gain.sum()),lost_depth=int((old&~now).sum()),visible_rescue=int((now&(depths[1][1]>=len(t))).sum()))
        if name=='native_train': record['coarse_hair_gained_depth']=int((gain&region_masks(frame)['hair']).sum())
        records.append(record)
    np.savez_compressed(out/'evidence.npz',eligible=eligible,vertex_ids=vertices,consensus_pass=good,coherence=coherence,
        normal_dot=normal_dot,weighted_agreement=agreement,neighbor_count=count,rescued=rescued,
        mask_support=support,mask_outside=outside,retained_raw_triangle_ids=retained)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),nearest_normal_failed=len(eligible),
        consensus_rescued=len(rescued),silhouette_admitted=len(retained),views=records,
        hashes={p.name:sha(p) for p in out.iterdir() if p.name not in ['request.json','result.json']},
        measured_guard_passed=False,production_updated=False,visual_status='pending',counts_not_quality_metrics=True))
    print(frame,len(eligible),len(rescued),len(retained),records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=FRAMES);run(p.parse_args().frame)
