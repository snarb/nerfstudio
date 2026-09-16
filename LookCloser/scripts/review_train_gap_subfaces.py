"""Separate subdivision texture changes from actual gap-carving changes."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from study_multiview_face_prior import read,save,sha
from study_train_gap_subfaces import ROOT,FRAME,COARSE
from study_train_gap_carving import PARENT
from review_measured_free_surface import VIEWS
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_jaw_repair_transfer import verified_image,panel
from review_subface_free_space import BOXES,HEADS
from subdivide_conflicted_surface import verify_coverage


def audit():
    root=ROOT/FRAME;q=read(root/'request.json');r=read(root/'result.json')
    assert r['request_sha256']==sha(root/'request.json')
    assert q['subface_script_sha256']==sha(Path(__file__).with_name('study_train_gap_subfaces.py'))
    for p,h in q['helper_hashes'].items():assert sha(p)==h
    for p,h in q['source_masks'].items():assert sha(p)==h
    for p,h in r['hashes'].items():assert sha(root/p)==h
    assert sha(q['mesh'])==q['mesh_sha256']
    assert sha(COARSE/FRAME/'positive_masks.npz')==q['positive_masks_sha256']
    e=np.load(root/'evidence.npz');v,t,p=e['vertices'],e['triangles'],e['parents']
    old=o3d.io.read_triangle_mesh(q['mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles)
    coverage=verify_coverage(ov,ot,v,t,p)
    points=np.concatenate([v,v[t].mean(1)]);samples=np.c_[t,np.arange(len(t))+len(v)]
    np.testing.assert_array_equal(points,e['points']);np.testing.assert_array_equal(samples,e['sample_indices'])
    pm=np.load(COARSE/FRAME/'positive_masks.npz');negatives=[];positives=[]
    for s in q['views']:
        assert sha(s['negative_mask'])==s['negative_sha256']
        nm=np.array(Image.open(s['negative_mask']))>0;pos=pm[s['camera']]
        c=s['camera_parameters'];pose=np.array(c['transform_matrix']);cam=(points-pose[:3,3])@pose[:3,:3];z=-cam[:,2]
        uv=np.c_[c['fl_x']*cam[:,0]/z+c['cx'],-c['fl_y']*cam[:,1]/z+c['cy']].astype(np.float32)
        xy=np.c_[uv[:,1],1919-uv[:,0]]-s['crop'][:2];pixel=np.floor(xy).astype(int)
        slab=(z>=s['depth_slab'][0])&(z<=s['depth_slab'][1])&np.isfinite(xy).all(1)&np.isfinite(z)
        neg=np.ones(len(points),bool);positive=np.zeros(len(points),bool)
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            x,y=pixel[:,0]+dx,pixel[:,1]+dy
            valid=slab&(x>=0)&(y>=0)&(x<nm.shape[1])&(y<nm.shape[0])
            idx=np.flatnonzero(valid);value=np.zeros(len(points),bool);value[idx]=nm[y[idx],x[idx]]
            neg &= value;positive[idx] |= pos[y[idx],x[idx]]
        negatives.append(neg);positives.append(positive)
    negatives=np.stack(negatives);positives=np.stack(positives)
    np.testing.assert_array_equal(negatives,e['negative_by_view']);np.testing.assert_array_equal(positives,e['positive_by_view'])
    count=np.zeros(len(t),int);veto=np.zeros(len(t),bool)
    for neg,pos in zip(negatives,positives):
        count+=np.logical_and.reduce([neg[column] for column in samples.T])
        veto|=np.logical_or.reduce([pos[column] for column in samples.T])
    remove=(count>=3)&~veto
    np.testing.assert_array_equal(np.flatnonzero(remove),e['removed_triangle_ids'])
    for arm,expected in [('refined',t),('carved',t[~remove])]:
        path=ROOT/arm/FRAME/'mesh.ply';result=read(ROOT/arm/FRAME/'result.json')
        assert sha(path)==result['hashes']['mesh.ply']
        actual=o3d.io.read_triangle_mesh(str(path))
        np.testing.assert_array_equal(np.asarray(actual.vertices),v)
        np.testing.assert_array_equal(np.asarray(actual.triangles),expected)
    save(root/'independent_audit.json',dict(request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),
        coverage=coverage,independent_mask_projection_replay=True,all_samples_checked=len(points),
        removed_subfaces=int(remove.sum()),physical_mask_truth_not_certified=True,script_sha256=sha(__file__)))


def main():
    root=ROOT/FRAME;dest=root/'review';assert not dest.exists();audit()
    records=[];bindings={}
    for view in VIEWS:
        paths=[PARENT/FRAME/'rgb'/view,COARSE/FRAME/'rgb'/view,ROOT/'refined'/FRAME/'rgb'/view,ROOT/'carved'/FRAME/'rgb'/view]
        images=[];meta=[];depths=[];requests=[]
        for path in paths:
            image,r=verified_image(path,FRAME);images.append(image);meta.append(r);requests.append(read(path/'request.json'))
            depths.append(np.rot90(np.load(path/'frames'/FRAME/'target_depth.npz')['depth']))
            for f in [path/'request.json',path/'frames'/FRAME/'complete.json']:bindings[str(f)]=sha(f)
            for n,h in read(path/'frames'/FRAME/'complete.json')['hashes'].items():bindings[str(path/'frames'/FRAME/n)]=h
        for r in meta[1:]:
            for k in ['camera','source_cameras','fixed_exposure']:assert r[k]==meta[0][k]
        for q in requests[1:]:
            for k in ['recipe','profiles_sha256','exposure_sha256','calibration_sha256','source_quality_implementation_sha256']:assert q[k]==requests[0][k]
        ds0,dc,dr,dp=depths
        # Pure retessellation can differ at a float32 ray/triangle edge, but
        # substantial depth movement must not be explained away as roundoff.
        common=(ds0>0)&(dr>0)
        max_difference=float(np.abs(ds0[common]-dr[common]).max())
        assert max_difference<1e-5
        assert not ((dr<=0)&(dp>0)).any() and not ((dr>0)&(dp>0)&(dp<dr-1e-6)).any()
        names=['quorum parent','coarse gap+veto','subdivision only','subdivision gap+veto']
        display_images=images.copy()
        if view!='moving':
            gt=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png';display_images.insert(0,np.array(Image.open(gt)));names.insert(0,'actual train GT');bindings[str(gt)]=sha(gt)
        folder=dest/view
        panel(folder/'lipstick_native.png',display_images,names,BOXES[view])
        panel(folder/'head_native.png',display_images,names,HEADS[view])
        pairs={}
        for label,old,new,d0,d1 in [('subdivision_only',images[0],images[2],ds0,dr),
                                  ('carving_on_refined',images[2],images[3],dr,dp),
                                  ('final_vs_coarse',images[1],images[3],dc,dp)]:
            black=(old.max(2)>0)&(new.max(2)==0)
            pairs[label]=dict(changed_rgb=int(np.any(old!=new,2).sum()),new_black=int(black.sum()),
                lost_depth=int(((d0>0)&(d1<=0)).sum()),new_depth=int(((d0<=0)&(d1>0)).sum()))
            overlay=new.copy();overlay[black]=[255,0,255]
            panel(folder/f'{label}_new_black.png',[old,new,overlay],['before','after','magenta:new black'],BOXES[view])
        records.append(dict(view=view,subdivision_max_depth_difference=max_difference,pairs=pairs))
    save(dest/'result.json',dict(records=records,input_hashes=bindings,
        images={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.png')},
        independent_audit_sha256=sha(root/'independent_audit.json'),visual_status='pending',
        counts_not_quality_metrics=True,production_promoted=False))
    print(records,flush=True)


if __name__=='__main__':main()
