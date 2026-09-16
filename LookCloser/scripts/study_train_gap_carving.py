"""Train-only visible-gap silhouette constraint, bounded to the object depth layer.

This intentionally permits reviewed semantic evidence to override correlated
PatchMatch foreground leakage. It is an opt-in experiment, not measured-depth
ground truth, and never edits production. Prepare and visually review masks
before geometry. No rendered diagnostic polygon is used to construct them.
"""
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from study_multiview_face_prior import read, save, sha
from study_query_support_quorum import ROOT as PARENT, FRAME
from study_lipstick_instance_masks import ROOT as MASKS, portrait_xy
from study_confidence_depth_prior import project_integer

ROOT = Path('/mnt/data/dec5_train_gap_carving')
HAND = Path('/mnt/data/dec5_lipstick_hand_mask_000995')
# Native train crop coordinates, drawn from the actual RGB crops, not a render.
# These broad context polygons are intersected with the eroded complement of
# reviewed hand/tube masks and an observed foreground depth slab.
CONTEXT = {
    'H004_C005_1210SZ': [(47,105),(116,105),(140,130),(144,160),(125,181),(89,200),(45,193)],
    'K004_B005_1210DS': [(25,104),(131,104),(132,155),(103,177),(69,196),(27,193)],
    'I004_C005_1210BA': [(32,96),(137,96),(145,155),(113,189),(78,209),(30,201)],
    'J004_C005_1210I4': [(25,94),(132,94),(151,151),(118,190),(68,208),(25,190)],
}


def negative_samples(mask, xy, z, bounds):
    xy, z = np.asarray(xy), np.asarray(z)
    if mask.ndim != 2 or xy.shape != (len(z),2) or len(bounds) != 2 or bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError('Invalid mask, projections or positive depth slab')
    # A complete 2x2 footprint must agree; do not round outside into a mask.
    finite = np.isfinite(xy).all(1) & np.isfinite(z)
    pixel = np.floor(np.where(np.isfinite(xy), xy, -99999)).astype(int)
    valid = finite & (pixel >= 0).all(1) & (pixel < [mask.shape[1]-1, mask.shape[0]-1]).all(1)
    valid &= (z >= bounds[0]) & (z <= bounds[1])
    ids = np.flatnonzero(valid); x, y = pixel[ids].T
    answer = np.zeros(len(z), bool)
    answer[ids] = mask[y,x] & mask[y+1,x] & mask[y,x+1] & mask[y+1,x+1]
    return answer


def removable(negative_by_view, samples):
    negative_by_view, samples = np.asarray(negative_by_view), np.asarray(samples)
    if negative_by_view.ndim != 2 or samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError('Expected view x point evidence and four samples per face')
    # Require the same three views to reject the entire sampled triangle.
    return negative_by_view[:,samples].all(2).sum(0) >= 3


def prepare():
    from study_confidence_depth_prior import load_real
    from review_full_block_transfer import ROOT as DEPTH_ROOT
    root = ROOT / FRAME
    assert not ROOT.exists()
    source = read(MASKS / 'request.json')
    rows, depths, receipt = load_real(DEPTH_ROOT, FRAME)
    inputs = {str(MASKS / 'request.json'): sha(MASKS / 'request.json')}
    masks = {}
    for folder, run in [(MASKS,'sam_v2'), (HAND,'sam_v1')]:
        review = read(folder / 'mask_review.json'); record = read(folder / run / 'result.json')
        assert review['status'] == 'usable_for_bounded_semantic_diagnostic'
        assert sha(folder / run / 'result.json') == review['result_sha256']
        assert record['request_sha256'] == sha(MASKS / 'request.json')
        inputs[str(folder / 'mask_review.json')] = sha(folder / 'mask_review.json')
        inputs[str(folder / run / 'result.json')] = sha(folder / run / 'result.json')
        for name in CONTEXT:
            path = folder / run / name / f'mask_{review["selected"][name]}.png'
            r = next(r for r in record['views'] if r['camera'] == name)
            assert sha(path) == r['outputs'][str(path.relative_to(folder / run))]
            inputs[str(path)] = sha(path)
            m = np.array(Image.open(path)) > 0
            masks[name] = masks.get(name, np.zeros_like(m)) | m
    (root / 'masks').mkdir(parents=True)
    records = []
    for name, polygon in CONTEXT.items():
        s = next(r for r in source['views'] if r['camera'] == name)
        i = next(i for i,r in enumerate(rows) if r['physical_camera'] == name)
        for k in ['transform_matrix','fl_x','fl_y','cx','cy','w','h']:
            np.testing.assert_array_equal(rows[i][k],s['camera_parameters'][k])
        path = Path(s['image']); assert sha(path) == s['image_sha256']; inputs[str(path)] = sha(path)
        im = Image.open(path).convert('RGB'); w,h = im.size
        poly = Image.new('1',(w,h)); ImageDraw.Draw(poly).polygon(polygon,fill=1)
        context = np.array(poly,bool); union = binary_fill_holes(masks[name])
        negative = context & (distance_transform_edt(~union) > 2)
        x0,y0,x1,y1 = s['crop']; d = np.rot90(depths[i])[y0:y1,x0:x1]
        anchor = context & (distance_transform_edt(union) > 3) & np.isfinite(d) & (d>0)
        assert anchor.sum() >= 100
        bounds = np.quantile(d[anchor],[.05,.95]) + [-.003,.003]
        maskpath = root / 'masks' / (name + '.png')
        Image.fromarray(negative.astype(np.uint8)*255).save(maskpath)
        marked = np.array(im); marked[negative] = (marked[negative]*.45+[0,140,140]).clip(0,255).astype(np.uint8)
        sheet = Image.new('RGB',(w*2,h+24)); sheet.paste(im,(0,24)); sheet.paste(Image.fromarray(marked),(w,24))
        ImageDraw.Draw(sheet).text((2,2), name + ' / cyan: gap constraint',fill='white')
        sheet.save(root / 'masks' / (name + '_review.png'))
        records.append(dict(camera=name,camera_parameters=rows[i],crop=s['crop'],
            context_polygon=polygon,negative_mask=str(maskpath),negative_sha256=sha(maskpath),
            depth_slab=bounds.tolist(),anchor_pixels=int(anchor.sum()),negative_pixels=int(negative.sum())))
    parent_result = read(PARENT / FRAME / 'result.json')
    assert sha(PARENT / FRAME / 'mesh.ply') == parent_result['hashes']['mesh.ply']
    save(root / 'request.json',dict(frame=FRAME,views=records,source_masks=inputs,
        depth_receipt=receipt,mesh=str(PARENT / FRAME / 'mesh.ply'),mesh_sha256=sha(PARENT / FRAME / 'mesh.ply'),
        parent_result_sha256=sha(PARENT / FRAME / 'result.json'),script_sha256=sha(__file__),
        mask_margin=2,mask_footprint='all four bilinear neighbours',minimum_same_negative_views=3,
        samples='three vertices plus centroid',depth_slab_rule='5..95 percentile interior hand/tube depth plus/minus .003',
        semantic_overrides_near_depth=True,training_rgb_only=True,uses_rendered_core=False,
        semantic_prior_not_measured_free_space=True,production_changed=False,
        review_images={str(p):sha(p) for p in (root/'masks').glob('*_review.png')}))
    print([(r['camera'],r['negative_pixels'],r['depth_slab']) for r in records],flush=True)


def geometry():
    import open3d as o3d
    root=ROOT/FRAME; q=read(root/'request.json'); review=read(root/'mask_review.json')
    assert review['status']=='accepted_for_bounded_experiment'
    assert review['request_sha256']==sha(root/'request.json')
    assert q['script_sha256']==sha(__file__) and sha(q['mesh'])==q['mesh_sha256']
    assert not (root/'mesh.ply').exists()
    for p,h in q['source_masks'].items(): assert sha(p)==h
    mesh=o3d.io.read_triangle_mesh(q['mesh']); v=np.asarray(mesh.vertices); t=np.asarray(mesh.triangles)
    points=np.concatenate([v,v[t].mean(1)]); samples=np.column_stack([t,np.arange(len(t))+len(v)])
    negative=[]
    for s in q['views']:
        assert sha(s['negative_mask'])==s['negative_sha256']
        mask=np.array(Image.open(s['negative_mask']))>0
        uv,z=project_integer(s['camera_parameters'],points)
        xy=portrait_xy(uv,1920)-s['crop'][:2]
        negative.append(negative_samples(mask,xy,z,s['depth_slab']))
    negative=np.stack(negative); remove=removable(negative,samples); keep=~remove
    assert keep.any()
    out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
    out.compute_vertex_normals(); assert o3d.io.write_triangle_mesh(str(root/'mesh.ply'),out)
    saved=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(saved.vertices),v)
    np.testing.assert_array_equal(np.asarray(saved.triangles),t[keep])
    np.savez_compressed(root/'evidence.npz',points=points,sample_indices=samples,
        negative_by_view=negative,removed_triangle_ids=np.flatnonzero(remove))
    save(root/'result.json',dict(request_sha256=sha(root/'request.json'),
        removed_triangles=int(remove.sum()),before_triangles=len(t),after_triangles=int(keep.sum()),
        vertices_unchanged=True,triangle_subset_exact=True,production_changed=False,
        mask_review_sha256=sha(root/'mask_review.json'),visual_status='pending',
        hashes={n:sha(root/n) for n in ['mesh.ply','evidence.npz']}))
    print('removed',remove.sum(),'triangles',flush=True)


def render_view(view):
    import review_measured_free_surface as workflow
    workflow.ROOT=ROOT; workflow.render(FRAME,view)


def render():
    import review_measured_free_surface as workflow
    workflow.ROOT=ROOT; workflow.prepare(FRAME)
    with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as p:
        p.map(render_view,workflow.VIEWS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','geometry','render'])
    globals()[p.parse_args().stage]()
