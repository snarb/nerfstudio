"""Paired gap-carving control: foreground evidence vetoes negative-view majority.

Uses the same four reviewed train masks, depth slabs and parent mesh. A point
inside/within two pixels of hand or tube in any one slab-valid view protects
the whole triangle. All four adjacent sampling pixels are checked for protection.
"""
import argparse
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes,distance_transform_edt
from study_multiview_face_prior import read,save,sha
from study_train_gap_carving import ROOT as NEGATIVE,FRAME,MASKS,HAND,removable
from study_confidence_depth_prior import project_integer
from study_lipstick_instance_masks import portrait_xy

ROOT=Path('/mnt/data/dec5_train_gap_positive_veto')


def positive_samples(mask,xy,z,bounds):
    xy,z=np.asarray(xy),np.asarray(z)
    finite=np.isfinite(xy).all(1)&np.isfinite(z)
    xy=np.floor(np.where(np.isfinite(xy),xy,-99999)).astype(int)
    positive=np.zeros(len(z),bool)
    for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
        x,y=xy[:,0]+dx,xy[:,1]+dy
        valid=finite&(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])&(z>=bounds[0])&(z<=bounds[1])
        ids=np.flatnonzero(valid);positive[ids]|=mask[y[ids],x[ids]]
    return positive


def geometry():
    import open3d as o3d
    parent=NEGATIVE/FRAME;root=ROOT/FRAME;assert not ROOT.exists()
    q=read(parent/'request.json');r=read(parent/'result.json')
    assert r['request_sha256']==sha(parent/'request.json')
    for p,h in r['hashes'].items():assert sha(parent/p)==h
    for p,h in q['source_masks'].items():assert sha(p)==h
    assert sha(q['mesh'])==q['mesh_sha256']
    e=np.load(parent/'evidence.npz');points=e['points'];samples=e['sample_indices'];neg=e['negative_by_view']
    positives=[];positive_masks={}
    for s in q['views']:
        name=s['camera'];masks=[]
        for folder,run in [(MASKS,'sam_v2'),(HAND,'sam_v1')]:
            rv=read(folder/'mask_review.json');p=folder/run/name/f'mask_{rv["selected"][name]}.png'
            assert sha(p)==q['source_masks'][str(p)]
            masks.append(np.array(Image.open(p))>0)
        union=binary_fill_holes(masks[0]|masks[1]);positive_mask=distance_transform_edt(~union)<=2
        uv,z=project_integer(s['camera_parameters'],points);xy=portrait_xy(uv,1920)-s['crop'][:2]
        positives.append(positive_samples(positive_mask,xy,z,s['depth_slab']))
        positive_masks[name]=positive_mask
    positives=np.stack(positives)
    original=removable(neg,samples)
    np.testing.assert_array_equal(np.flatnonzero(original),e['removed_triangle_ids'])
    veto=positives[:,samples].any((0,2));remove=original&~veto
    mesh=o3d.io.read_triangle_mesh(q['mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
    np.testing.assert_array_equal(points[:len(v)],v);np.testing.assert_array_equal(samples[:,:3],t)
    root.mkdir(parents=True)
    request=dict(q,parent_gap_request_sha256=sha(parent/'request.json'),
        parent_gap_result_sha256=sha(parent/'result.json'),parent_gap_evidence_sha256=sha(parent/'evidence.npz'),
        positive_veto=True,positive_margin=2,positive_sampling='any bilinear neighbour',
        veto_rule='any vertex or centroid positive in any slab-valid reviewed camera',
        positive_script_sha256=sha(__file__),parent_mask_review_sha256=sha(parent/'mask_review.json'))
    save(root/'request.json',request)
    out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[~remove]))
    out.compute_vertex_normals();assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),out)
    check=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(check.vertices),v)
    np.testing.assert_array_equal(np.asarray(check.triangles),t[~remove])
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,negative_by_view=neg,
        positive_by_view=positives,removed_triangle_ids=np.flatnonzero(remove),vetoed_triangle_ids=np.flatnonzero(original&veto))
    np.savez_compressed(root/'positive_masks.npz',**positive_masks)
    save(root/'result.json',dict(request_sha256=sha(root/'request.json'),removed_triangles=int(remove.sum()),
        vetoed_triangles=int((original&veto).sum()),vertices_unchanged=True,triangle_subset_exact=True,
        production_changed=False,visual_status='pending',
        hashes={n:sha(root/n) for n in ['mesh.ply','evidence.npz','positive_masks.npz']}))
    print('removed',remove.sum(),'vetoed',int((original&veto).sum()),flush=True)


def render_view(view):
    import review_measured_free_surface as workflow
    workflow.ROOT=ROOT;workflow.render(FRAME,view)


def render():
    import review_measured_free_surface as workflow
    workflow.ROOT=ROOT;workflow.prepare(FRAME)
    with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as p:p.map(render_view,workflow.VIEWS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['geometry','render'])
    globals()[p.parse_args().stage]()
