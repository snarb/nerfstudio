"""Private mask-qualified near-evidence ablation; measured far gates unchanged.

Five reviewed train hand+lipstick instance unions can reject a close-depth vote
only when BOTH the queried point and the sampled depth pixel lie clearly outside
the union. A five-crop intersection and two negative views bound this exception.
This is a disclosed semantic prior, not mask-independent measured free space.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_fill_holes

from joint_temporal_texture import read, sha, atomic_json, project
from study_confidence_depth_prior import load_real, project_integer, support, unproject
from diagnose_jaw_measured_depth import observed_at
from prune_measured_free_surface import near_tap_evidence, removable
from ablate_free_surface_near_footprint import ROOT as PARENT
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_lipstick_instance_masks import ROOT as OBJECT, portrait_xy
from audit_lipstick_instance_witnesses import signed_mask_distance

ROOT = Path('/mnt/data/dec5_instance_qualified_free_surface')
HAND = Path('/mnt/data/dec5_lipstick_hand_mask_000995')
FRAME = '000995'


def outside_mask(sdf, xy, margin=2):
    xy = np.asarray(xy, np.float32)
    valid = np.isfinite(xy).all(1) & ((xy >= 0) & (xy <= [sdf.shape[1]-1, sdf.shape[0]-1])).all(1)
    safe = np.where(np.isfinite(xy), xy, -10000)
    values = np.empty(len(safe),np.float32)
    for start in range(0,len(safe),16384):
        values[start:start+16384] = cv2.remap(sdf, safe[start:start+16384].reshape(1,-1,2),
                                             None, cv2.INTER_LINEAR)[0]
    return valid & (values <= -margin), valid


def load_masks():
    q = read(OBJECT/'request.json'); records = []; unions = {}
    for root, child in [(OBJECT, 'sam_v2'), (HAND, 'sam_v1')]:
        review = read(root/'mask_review.json'); result = read(root/child/'result.json')
        assert review['status'] == 'usable_for_bounded_semantic_diagnostic'
        assert review['result_sha256'] == sha(root/child/'result.json')
        assert result['request_sha256'] == sha(OBJECT/'request.json')
        for path, digest in review['reviewed_bindings'].items(): assert sha(path) == digest
        for row in q['views']:
            name = row['camera']; source = next(r for r in result['views'] if r['camera'] == name)
            index = review['selected'][name]; assert index == int(np.argmax(source['scores']))
            path = root/child/name/f'mask_{index}.png'
            assert sha(path) == source['outputs'][str(path.relative_to(root/child))]
            with Image.open(path) as im: mask = np.asarray(im) > 0
            unions[name] = mask | unions.get(name, np.zeros_like(mask))
            records.append(dict(path=str(path), sha256=sha(path), review_sha256=sha(root/'mask_review.json')))
    return q, {k:binary_fill_holes(v) for k,v in unions.items()}, records


def geometry():
    root = ROOT/FRAME; assert not root.exists()
    parent = PARENT/FRAME; pq = read(parent/'request.json'); pr = read(parent/'result.json')
    assert pr['request_sha256'] == sha(parent/'request.json')
    for path,digest in pq['scripts'].items(): assert sha(path) == digest
    for name,digest in pr['hashes'].items(): assert sha(parent/name) == digest
    assert sha(pq['mesh']) == pq['mesh_sha256']
    ev = np.load(parent/'evidence.npz'); points, samples = ev['points'], ev['sample_indices']
    rows, depths, receipt = load_real(DEPTH_ROOT, FRAME); assert receipt == pq['depth_receipt']
    mq, masks, mask_records = load_masks()
    negatives = np.zeros((62,len(points)), bool); domain = np.ones(len(points), bool)
    for source in mq['views']:
        name = source['camera']; ci = next(i for i,r in enumerate(rows) if r['physical_camera'] == name)
        row = rows[ci]
        for key in ['transform_matrix','fl_x','fl_y','cx','cy','w','h']:
            np.testing.assert_allclose(row[key], source['camera_parameters'][key], atol=0, rtol=0)
        sdf = signed_mask_distance(masks[name]); offset = np.array(source['crop'][:2])
        uv,z = project(points,[row]); pixel,zz = project_integer(row,points)
        qxy = portrait_xy(uv[0],row['w'])-offset
        # Check the actual native nearest depth-array pixel as well as query UV.
        sxy = portrait_xy(np.rint(pixel),row['w'])-offset
        nq,vq = outside_mask(sdf,qxy); ns,vs = outside_mask(sdf,sxy)
        valid = vq & vs & (z[0]>0) & (zz>0)
        domain &= valid; negatives[ci] = nq & ns & valid
    eligible = domain & (negatives.sum(0)>=2)
    root.mkdir(parents=True)
    q = deepcopy(pq); q.update(parent_request_sha256=sha(parent/'request.json'),
        parent_evidence_sha256=sha(parent/'evidence.npz'), mask_request_sha256=sha(OBJECT/'request.json'),
        mask_inputs=mask_records, geometry_uses_masks=True, geometry_uses_rgb=True, geometry_uses_roi=True,
        roi='intersection of five recorded train-image crops; outside is unknown',
        semantic_prior=True, masks_are_not_independent_measured_geometry=True,
        ablation='qualify native near votes with reviewed hand+lipstick union masks; all far gates unchanged')
    q['parameters'].update(protect_any_near_tap=False, mask_margin_pixels=2,
        mask_minimum_negative_views=2, both_query_and_sample_outside=True,
        fill_enclosed_union_holes=True)
    for name in [Path(__file__).name,'study_lipstick_instance_masks.py','audit_lipstick_instance_witnesses.py']:
        p=Path(__file__).with_name(name); q['scripts'][str(p.resolve())]=sha(p)
    atomic_json(root/'request.json',q)
    raw = np.zeros(len(points),np.int16); adjusted=raw.copy(); suppressed=[]
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        uv,z=project_integer(row,points); near=near_tap_evidence(depth,uv,z,radius=0)
        veto=near & eligible & negatives[ci]
        raw+=near; adjusted+=near & ~veto
        suppressed.append(dict(camera=row['physical_camera'],near=int(near.sum()),rejected=int(veto.sum())))
    np.testing.assert_array_equal(raw,ev['near_counts']); assert (adjusted>=0).all()
    stable=ev['stable_far_counts']
    candidates=np.flatnonzero((adjusted[samples]==0).all(1)&(stable[samples]>=6).all(1))
    query=points[samples[candidates]].reshape(-1,3); trusted=np.zeros((62,len(candidates),4),bool)
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        xy,z,obs,ok=observed_at(query,row,depth);j=np.flatnonzero(ok&(obs>z+np.maximum(.005,.01*z)))
        if len(j):
            count,_=support(unproject(row,xy[j,0],xy[j,1],obs[j]),row,rows,depths)
            trusted[ci].reshape(-1)[j]=count>=3
        atomic_json(root/'progress.json',dict(stage='far_corroboration',camera=ci+1,candidates=len(candidates),utc_seconds=time.time()))
    remove=candidates[removable(adjusted[samples[candidates]],stable[samples[candidates]],trusted.sum(0))]
    mesh=o3d.io.read_triangle_mesh(pq['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(points[:len(v)],v);np.testing.assert_array_equal(samples[:,:3],t)
    keep=np.ones(len(t),bool);keep[remove]=False;assert keep.any()
    out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]));out.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),out)
    saved=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices),v);np.testing.assert_array_equal(np.asarray(saved.triangles),t[keep])
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,near_counts=adjusted,
        unqualified_near_counts=raw,stable_far_counts=stable,negative_mask=negatives,mask_domain=domain,
        eligible=eligible,candidates=candidates,trusted_far_by_camera=trusted,removed_triangle_ids=remove)
    old_remove=ev['removed_triangle_ids'];assert np.isin(old_remove,remove).all()
    _,components,_=out.cluster_connected_triangles()
    atomic_json(root/'result.json',dict(request_sha256=sha(root/'request.json'),before_triangles=len(t),
        after_triangles=int(keep.sum()),removed_triangles=len(remove),candidate_triangles=len(candidates),
        additional_removed_triangles=len(np.setdiff1d(remove,old_remove)),mask_domain_points=int(domain.sum()),
        mask_eligible_points=int(eligible.sum()),near_suppression=suppressed,
        components=sorted(map(int,components),reverse=True),vertices_unchanged=True,triangle_subset_exact=True,
        no_component_cleanup=True,production_changed=False,visual_status='pending',
        hashes={name:sha(root/name) for name in ['mesh.ply','evidence.npz']}))
    print('Semantic near qualification:',len(remove),'removed;',len(np.setdiff1d(remove,old_remove)),'additional',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['geometry','prepare','render'])
    p.add_argument('--view');a=p.parse_args();cv2.setNumThreads(2)
    if a.stage=='geometry': geometry()
    else:
        import review_measured_free_surface as workflow
        workflow.ROOT=ROOT
        if a.stage=='prepare':workflow.prepare(FRAME)
        else:
            assert a.view in workflow.VIEWS
            workflow.render(FRAME,a.view)
