"""Isolate geometric sampling from texture effects of local gap subdivision.

Two conforming midpoint rounds preserve the surface exactly. The same frozen
four-view negative/positive masks, depth slabs and confidence rules then select
subfaces. Publish both subdivision-only and subdivision+carving controls.
"""
import argparse
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image
from study_multiview_face_prior import read,save,sha
from study_train_gap_positive_veto import ROOT as COARSE,FRAME,positive_samples
from study_train_gap_carving import negative_samples,removable
from study_confidence_depth_prior import project_integer
from study_lipstick_instance_masks import portrait_xy
from subdivide_conflicted_surface import subdivide,verify_coverage

ROOT=Path('/mnt/data/dec5_train_gap_subfaces')


def selected_parents(negative,samples):
    """Refine faces touching the >=3-view negative domain, not a rendered ROI."""
    return (np.asarray(negative).sum(0)[samples]>=3).any(1)


def classify(q,points,positive_masks):
    negatives=[];positives=[]
    for s in q['views']:
        assert sha(s['negative_mask'])==s['negative_sha256']
        mask=np.array(Image.open(s['negative_mask']))>0
        uv,z=project_integer(s['camera_parameters'],points)
        xy=portrait_xy(uv,1920)-s['crop'][:2]
        negatives.append(negative_samples(mask,xy,z,s['depth_slab']))
        positives.append(positive_samples(positive_masks[s['camera']],xy,z,s['depth_slab']))
    return np.stack(negatives),np.stack(positives)


def geometry():
    import open3d as o3d
    base=COARSE/FRAME;q=read(base/'request.json');r=read(base/'result.json')
    assert r['request_sha256']==sha(base/'request.json')
    for p,h in r['hashes'].items():assert sha(base/p)==h
    for p,h in q['source_masks'].items():assert sha(p)==h
    assert sha(q['mesh'])==q['mesh_sha256']
    assert not ROOT.exists();root=ROOT/FRAME;root.mkdir(parents=True)
    e=np.load(base/'evidence.npz');positive_masks=np.load(base/'positive_masks.npz')
    mesh=o3d.io.read_triangle_mesh(q['mesh']);ov=np.asarray(mesh.vertices);ot=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(ov,e['points'][:len(ov)]);np.testing.assert_array_equal(ot,e['sample_indices'][:,:3])
    negative,positive=classify(q,e['points'],positive_masks)
    np.testing.assert_array_equal(negative,e['negative_by_view']);np.testing.assert_array_equal(positive,e['positive_by_view'])
    selected=selected_parents(negative,e['sample_indices'])
    v,t,p=ov.copy(),ot.copy(),np.arange(len(ot));rounds=[]
    for step in range(2):
        v,t,p=subdivide(v,t,selected[p],p)
        check=verify_coverage(ov,ot,v,t,p)
        rounds.append(dict(round=step+1,vertices=len(v),triangles=len(t),coverage=check))
    points=np.concatenate([v,v[t].mean(1)]);samples=np.c_[t,np.arange(len(t))+len(v)]
    negative,positive=classify(q,points,positive_masks)
    remove=removable(negative,samples)&~positive[:,samples].any((0,2))
    request=dict(q,coarse_request_sha256=sha(base/'request.json'),coarse_result_sha256=sha(base/'result.json'),
        coarse_evidence_sha256=sha(base/'evidence.npz'),positive_masks_sha256=sha(base/'positive_masks.npz'),
        subdivision_rounds=2,subdivision_selection='any face sample has >=3 negative views',
        rules_and_masks_unchanged=True,production_changed=False,subface_script_sha256=sha(__file__),
        helper_hashes={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in
          ['study_train_gap_carving.py','study_train_gap_positive_veto.py','subdivide_conflicted_surface.py',
           'study_confidence_depth_prior.py','study_lipstick_instance_masks.py']})
    save(root/'request.json',request)
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,parents=p,
        selected_parents=selected,negative_by_view=negative,positive_by_view=positive,
        removed_triangle_ids=np.flatnonzero(remove),vertices=v,triangles=t)
    controls=[]
    for arm,triangles in [('refined',t),('carved',t[~remove])]:
        out=ROOT/arm/FRAME;out.mkdir(parents=True)
        save(out/'request.json',dict(request,control=arm,study_request_sha256=sha(root/'request.json')))
        m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(triangles))
        m.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(out/'mesh.ply'),m)
        actual=o3d.io.read_triangle_mesh(str(out/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(actual.vertices),v);np.testing.assert_array_equal(np.asarray(actual.triangles),triangles)
        result=dict(control=arm,request_sha256=sha(out/'request.json'),hashes={'mesh.ply':sha(out/'mesh.ply')},
            vertices=len(v),triangles=len(triangles),production_changed=False,visual_status='pending')
        save(out/'result.json',result);controls.append(result)
    area=np.linalg.norm(np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]),axis=1)/2
    removed_area=np.bincount(p[remove],weights=area[remove],minlength=len(ot))
    old_removed=e['removed_triangle_ids'];parent_area=np.linalg.norm(np.cross(ov[ot[:,1]]-ov[ot[:,0]],ov[ot[:,2]]-ov[ot[:,0]]),axis=1)/2
    save(root/'result.json',dict(request_sha256=sha(root/'request.json'),rounds=rounds,
        selected_parents=int(selected.sum()),removed_subfaces=int(remove.sum()),
        affected_parents=int((removed_area>0).sum()),removed_area=float(removed_area.sum()),
        coarse_removed_area=float(parent_area[old_removed].sum()),
        previously_removed_area_restored=float(np.maximum(parent_area[old_removed]-removed_area[old_removed],0).sum()),
        controls=controls,hashes={'evidence.npz':sha(root/'evidence.npz')},production_changed=False))
    print('selected parents',selected.sum(),'removed subfaces',remove.sum(),'affected parents',(removed_area>0).sum(),flush=True)


def render_view(task):
    arm,view=task
    import review_measured_free_surface as renderer
    renderer.ROOT=ROOT/arm;renderer.render(FRAME,view)


def render():
    import review_measured_free_surface as renderer
    tasks=[]
    for arm in ['refined','carved']:
        renderer.ROOT=ROOT/arm;renderer.prepare(FRAME)
        tasks.extend((arm,v) for v in renderer.VIEWS)
    with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as pool:pool.map(render_view,tasks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['geometry','render'])
    globals()[p.parse_args().stage]()
