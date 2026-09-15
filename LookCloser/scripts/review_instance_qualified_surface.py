"""Matched semantic-only controls and exact deletion replay, not quality metrics."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from joint_temporal_texture import read, sha, atomic_json
from prune_measured_free_surface import removable
from review_jaw_repair_transfer import verified_image, panel
from review_measured_free_surface import VIEWS, DEPTH_ROOT

PAIRS = {
    'coarse': ('/mnt/data/dec5_instance_qualified_free_surface',
               '/mnt/data/dec5_measured_free_center_pruning'),
    'subface': ('/mnt/data/dec5_subface_instance_qualified_free_surface',
                '/mnt/data/dec5_subface_free_space/pruned'),
}
BOXES = {'moving': (380,1400,680,1870), 'H004_C005_1210SZ': (230,1080,530,1550),
         'K004_B005_1210DS': (40,1120,340,1590)}
HEADS = {'moving': (150,900,850,1490), 'H004_C005_1210SZ': (100,500,1000,1200),
         'K004_B005_1210DS': (100,450,950,1210)}


def geometry_audit(root):
    q,r = read(root/'request.json'),read(root/'result.json')
    assert sha(root/'request.json') == r['request_sha256']
    bindings = {str(root/'request.json'):sha(root/'request.json'),
                str(root/'result.json'):sha(root/'result.json')}
    for p,h in {**q['scripts'],q['mesh']:q['mesh_sha256'],
                **{str(root/p):h for p,h in r['hashes'].items()},
                **{v['path']:v['sha256'] for v in q['mask_inputs']}}.items():
        assert sha(p)==h,p
        bindings[p]=h
    e=np.load(root/'evidence.npz');s=e['sample_indices'];c=e['candidates']
    expected=c[removable(e['near_counts'][s[c]],e['stable_far_counts'][s[c]],
                        e['trusted_far_by_camera'].sum(0))]
    np.testing.assert_array_equal(expected,e['removed_triangle_ids'])
    assert np.all(e['near_counts']<=e['unqualified_near_counts'])
    assert np.all(e['near_counts']>=0)
    np.testing.assert_array_equal(e['near_counts'][~e['eligible']],
                                  e['unqualified_near_counts'][~e['eligible']])
    np.testing.assert_array_equal(e['eligible'],e['mask_domain'] & (e['negative_mask'].sum(0)>=2))
    before=o3d.io.read_triangle_mesh(q['mesh']);after=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    vertices=np.asarray(before.vertices);triangles=np.asarray(before.triangles)
    np.testing.assert_array_equal(vertices,np.asarray(after.vertices))
    np.testing.assert_array_equal(e['points'][:len(vertices)],vertices)
    np.testing.assert_array_equal(s[:,:3],triangles)
    keep=np.ones(len(triangles),bool);keep[expected]=False
    np.testing.assert_array_equal(triangles[keep],np.asarray(after.triangles))
    assert len(expected)==r['removed_triangles']
    return bindings


def sheets(variant):
    root=Path(PAIRS[variant][0])/'000995/semantic_review'
    audit=read(root/'audit.json');paths=[]
    for row in audit['records']:
        paths.extend(Path(c['path']) for c in row['new_black_components'])
    output=[]
    for start in range(0,len(paths),6):
        group=paths[start:start+6]
        images=[Image.open(p).convert('RGB') for p in group]
        canvas=Image.new('RGB',(max(im.width for im in images),sum(im.height+22 for im in images)))
        draw=ImageDraw.Draw(canvas);y=0
        for p,im in zip(group,images):
            draw.text((0,y),f'{p.parent.name}/{p.name}',fill='white');canvas.paste(im,(0,y+22));y+=im.height+22
        dest=root/f'black_sheet_{start//6:02}.png';assert not dest.exists();canvas.save(dest)
        output.append(dict(path=str(dest),sha256=sha(dest),sources=[str(p) for p in group]))
    atomic_json(root/'black_sheets.json',dict(native_pixels_no_resize=True,sheets=output))


def review(variant):
    root,control=[Path(p)/'000995' for p in PAIRS[variant]]
    out=root/'semantic_review';assert not out.exists();out.mkdir()
    bindings=geometry_audit(root);records=[]
    for view in VIEWS:
        old,new=control/'rgb'/view,root/'rgb'/view
        a,ar=verified_image(old,'000995');b,br=verified_image(new,'000995')
        for k in ['camera','source_cameras','fixed_exposure']:assert ar[k]==br[k],k
        aq,bq=read(old/'request.json'),read(new/'request.json')
        for k in ['profiles_sha256','exposure_sha256','calibration_sha256',
                  'source_quality_implementation_sha256']:assert aq[k]==bq[k],k
        for folder in [old,new]:
            for p in [folder/'request.json',*(folder/'frames/000995').iterdir()]:
                if p.is_file():bindings[str(p)]=sha(p)
        ad=np.rot90(np.load(old/'frames/000995/target_depth.npz')['depth'])
        bd=np.rot90(np.load(new/'frames/000995/target_depth.npz')['depth'])
        assert ad.shape==a.shape[:2]==bd.shape
        assert np.isfinite(ad).all() and np.isfinite(bd).all()
        assert not ((ad==0)&(bd>0)).any()
        assert not ((ad>0)&(bd>0)&(bd<ad-1e-6)).any()
        images=[a,b];names=['same topology, depth-only','+ semantic near qualification']
        if view!='moving':
            p=DEPTH_ROOT/'000995/review'/view/'train_gt.png'
            images.insert(0,np.array(Image.open(p)));names.insert(0,'real train GT');bindings[str(p)]=sha(p)
        for kind,box in [('lipstick',BOXES[view]),('head',HEADS[view])]:
            panel(out/view/(kind+'.png'),images,names,box)
        black=np.any(a>0,2)&np.all(b==0,2)
        components,n=label(black);localized=[]
        for idx,slices in enumerate(find_objects(components),1):
            if slices is None:continue
            ys,xs=slices;box=(max(0,xs.start-18),max(0,ys.start-18),
                             min(a.shape[1],xs.stop+18),min(a.shape[0],ys.stop+18))
            p=out/view/f'new_black_{idx:03}.png';panel(p,images,names,box)
            localized.append(dict(path=str(p),pixels=int((components==idx).sum()),box=box))
        records.append(dict(view=view,changed_rgb_pixels=int(np.any(a!=b,2).sum()),
            lost_depth=int(((ad>0)&(bd==0)).sum()),farther_depth=int(((ad>0)&(bd>ad+1e-6)).sum()),
            new_black_pixels=int(black.sum()),new_black_components=localized))
    for p in out.rglob('*.png'):bindings[str(p)]=sha(p)
    atomic_json(out/'audit.json',dict(variant=variant,records=records,bindings=bindings,
        geometry_subset_replayed=True,matching_cameras_color_and_renderer_checked=True,
        diagnostic_counts_not_quality_metrics=True,full_frame_quality_metrics=False,
        script_sha256=sha(__file__),visual_status='pending',production_promoted=False))
    print(variant,[(r['view'],r['changed_rgb_pixels'],r['new_black_pixels']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('variant',choices=PAIRS)
    p.add_argument('--sheets',action='store_true');a=p.parse_args()
    (sheets if a.sheets else review)(a.variant)
