"""Replay gap carving independently and compare matched hard-source renders."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from study_multiview_face_prior import read,save,sha
from study_train_gap_carving import ROOT,PARENT,FRAME
from review_measured_free_surface import VIEWS
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_jaw_repair_transfer import verified_image,panel
from review_subface_free_space import BOXES,HEADS


def audit():
    root=ROOT/FRAME; q=read(root/'request.json'); r=read(root/'result.json')
    assert r['request_sha256']==sha(root/'request.json')
    assert q['script_sha256']==sha(Path(__file__).with_name('study_train_gap_carving.py'))
    assert sha(q['mesh'])==q['mesh_sha256']
    for p,h in q['source_masks'].items():assert sha(p)==h
    for p,h in q['review_images'].items():assert sha(p)==h
    assert r['mask_review_sha256']==sha(root/'mask_review.json')
    for p,h in r['hashes'].items():assert sha(root/p)==h
    before=o3d.io.read_triangle_mesh(q['mesh']);after=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    v=np.asarray(before.vertices);t=np.asarray(before.triangles)
    points=np.concatenate([v,v[t].mean(1)]);samples=np.c_[t,np.arange(len(t))+len(v)]
    cached=np.load(root/'evidence.npz');np.testing.assert_array_equal(points,cached['points'])
    np.testing.assert_array_equal(samples,cached['sample_indices'])
    reproduced=[]
    for s in q['views']:
        assert sha(s['negative_mask'])==s['negative_sha256']
        m=np.array(Image.open(s['negative_mask']))>0
        c=s['camera_parameters'];pose=np.array(c['transform_matrix'])
        cam=(points-pose[:3,3])@pose[:3,:3];z=-cam[:,2]
        uv=np.c_[c['fl_x']*cam[:,0]/z+c['cx'],-c['fl_y']*cam[:,1]/z+c['cy']].astype(np.float32)
        xy=np.c_[uv[:,1],1919-uv[:,0]]-s['crop'][:2]
        p=np.floor(xy).astype(int);x,y=p.T;lo,hi=s['depth_slab']
        ok=(x>=0)&(y>=0)&(x<m.shape[1]-1)&(y<m.shape[0]-1)&np.isfinite(xy).all(1)&np.isfinite(z)&(z>=lo)&(z<=hi)
        ii=np.flatnonzero(ok);b=np.zeros(len(points),bool)
        b[ii]=np.logical_and.reduce([m[y[ii]+dy,x[ii]+dx] for dy,dx in [(0,0),(1,0),(0,1),(1,1)]])
        reproduced.append(b)
    reproduced=np.array(reproduced);np.testing.assert_array_equal(reproduced,cached['negative_by_view'])
    hits=np.zeros(len(t),int)
    for b in reproduced:
        hits += b[t[:,0]] & b[t[:,1]] & b[t[:,2]] & b[len(v):]
    removed=np.flatnonzero(hits>=3);np.testing.assert_array_equal(removed,cached['removed_triangle_ids'])
    keep=hits<3
    np.testing.assert_array_equal(np.asarray(after.vertices),v)
    np.testing.assert_array_equal(np.asarray(after.triangles),t[keep])
    save(root/'independent_audit.json',dict(request_sha256=sha(root/'request.json'),
        result_sha256=sha(root/'result.json'),independent_projection_and_footprint_replay=True,
        removed_faces=int((~keep).sum()),whole_mesh_samples_checked=len(points),
        exact_vertices_and_subset=True,semantic_truth_not_certified=True,script_sha256=sha(__file__)))


def review():
    root=ROOT/FRAME;dest=root/'review';assert not dest.exists()
    audit(); records=[];bindings={}
    for view in VIEWS:
        before=PARENT/FRAME/'rgb'/view;after=root/'rgb'/view
        a,ar=verified_image(before,FRAME);b,br=verified_image(after,FRAME)
        for k in ['camera','source_cameras','fixed_exposure']:assert ar[k]==br[k]
        aq,bq=read(before/'request.json'),read(after/'request.json')
        for k in ['recipe','profiles_sha256','exposure_sha256','calibration_sha256','source_quality_implementation_sha256']:
            assert aq[k]==bq[k]
        ds=[np.rot90(np.load(p/'frames'/FRAME/'target_depth.npz')['depth']) for p in [before,after]]
        d0,d1=ds;assert np.isfinite(d0).all() and np.isfinite(d1).all()
        assert not ((d0<=0)&(d1>0)).any()
        assert not ((d0>0)&(d1>0)&(d1<d0-1e-6)).any()
        images=[a,b];names=['quorum mesh','+ train-gap constraint']
        if view!='moving':
            gt=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png'
            images.insert(0,np.array(Image.open(gt)));names.insert(0,'actual train GT');bindings[str(gt)]=sha(gt)
        folder=dest/view
        panel(folder/'lipstick_native.png',images,names,BOXES[view]);panel(folder/'head_native.png',images,names,HEADS[view])
        newly_black=(a.max(2)>0)&(b.max(2)==0)
        lab,n=label(newly_black);patches=[];components=[]
        for i,sl in enumerate(find_objects(lab),1):
            yy,xx=sl;box=(max(0,xx.start-12),max(0,yy.start-12),min(1080,xx.stop+12),min(1920,yy.stop+12))
            size=int((lab==i).sum());p=folder/f'new_black_{i:03d}.png'
            panel(p,[a,b],[f'{i}: before / {size}px','after'],box)
            patches.append(Image.open(p).copy());components.append(dict(pixels=size,box=box,path=str(p)))
        # Pack native-size component patches into rows, no thumbnail resizing.
        width=max(800,max((p.width for p in patches),default=1));x=y=0;row_h=0;positions=[]
        for p in patches:
            if x+p.width>width:y+=row_h+8;x=0;row_h=0
            positions.append((x,y));x+=p.width+8;row_h=max(row_h,p.height)
        sheet=Image.new('RGB',(width,max(1,y+row_h)),(35,35,35))
        for p,xy in zip(patches,positions):sheet.paste(p,xy)
        sheet.save(folder/'new_black_native_sheet.png')
        for p in [before,after]:
            for name in ['request.json',f'frames/{FRAME}/complete.json']:bindings[str(p/name)]=sha(p/name)
            comp=read(p/'frames'/FRAME/'complete.json')
            for name,h in comp['hashes'].items():bindings[str(p/'frames'/FRAME/name)]=h
        records.append(dict(view=view,changed_rgb=int(np.any(a!=b,2).sum()),new_black=int(newly_black.sum()),
            lost_depth=int(((d0>0)&(d1<=0)).sum()),farther_common_depth=int(((d0>0)&(d1>d0+1e-6)).sum()),
            components=components))
    save(dest/'result.json',dict(records=records,input_hashes=bindings,
        images={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.png')},
        independent_audit_sha256=sha(root/'independent_audit.json'),visual_status='pending',
        production_changed=False,diagnostic_counts_not_quality_metrics=True))
    print([{k:v for k,v in r.items() if k!='components'} for r in records],flush=True)


if __name__=='__main__':review()
