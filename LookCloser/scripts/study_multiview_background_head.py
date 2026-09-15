"""Matched RGB and independent evidence replay for head-background pruning.

No production promotion. The pruned+inset arm has NOT rerun the native
free-space guard on its newly exposed shell; its RGB is a diagnostic only.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from prune_multiview_background_head import ROOT,INSET,MOVIE,SOURCE,MASKS,FRAMES
from study_confidence_depth_prior import REGIONS,load_real
from study_jaw_depth_footprint import train_reference_votes
from review_jaw_repair_transfer import verified_image,panel


def audit(frame):
    root=ROOT/frame;q=read(root/'request.json');result=read(root/'result.json')
    assert result['request_sha256']==sha(root/'request.json')
    for p,h in q['scripts'].items():assert sha(p)==h
    for p,h in result['hashes'].items():assert sha(root/p)==h
    assert sha(q['source_mesh'])==q['source_mesh_sha256']
    assert sha(SOURCE/frame/'request.json')==q['source_request_sha256']
    b=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(b['depth_root']),frame)
    assert receipt==q['depth_receipt'];a=np.load(root/'evidence.npz')
    old=o3d.io.read_triangle_mesh(q['source_mesh']);v=np.asarray(old.vertices);t=np.asarray(old.triangles)
    ids=np.flatnonzero((v[t][:,:,0]>q['settings']['min_head_x']).all(1))
    np.testing.assert_array_equal(ids,a['head_triangle_ids'])
    p=v[t[ids]];samples=np.stack([p[:,0],p[:,1],p[:,2],(p[:,0]+p[:,1])/2,
        (p[:,1]+p[:,2])/2,(p[:,2]+p[:,0])/2,p.mean(1)],axis=1)
    masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
    assert sha(MASKS/frame/'masks.npz')==q['masks_sha256']
    assert sha(MASKS/frame/'cameras.json')==q['mask_camera_sha256']
    assert sha(MASKS/frame/'result.json')==q['mask_result_sha256']
    # Independent summed-area window test instead of producer's dilation.
    radius=q['settings']['mask_radius_pixels'];counts=[];flat=samples.reshape(-1,3)
    for row in rows:
        mask=masks[names.index(row['physical_camera'])];h,w=mask.shape
        uv,z=project(flat,[row]);finite=np.isfinite(uv[0]).all(1)&np.isfinite(z[0])
        xy=np.rint(np.where(np.isfinite(uv[0]),uv[0],0)).astype(int);x,y=xy.T
        valid=finite&(z[0]>0)&(x>=radius)&(x<w-radius)&(y>=radius)&(y<h-radius)
        integral=cv2.integral(mask.astype(np.uint8));x=x[valid];y=y[valid]
        area=integral[y+radius+1,x+radius+1]-integral[y-radius,x+radius+1]-integral[y+radius+1,x-radius]+integral[y-radius,x-radius]
        clear=np.zeros(len(flat),bool);clear[valid]=area==0;counts.append(clear.reshape(-1,7).all(1))
    counts=np.asarray(counts);np.testing.assert_array_equal(counts,a['background_camera_votes'])
    eligible=np.flatnonzero(counts.sum(0)>=q['settings']['minimum_clear_background_cameras'])
    np.testing.assert_array_equal(eligible,a['eligible_head_indices'])
    np.testing.assert_array_equal(samples[eligible],a['query_points'])
    votes,refs=train_reference_votes(samples[eligible].reshape(-1,3),rows,depths,tolerance=q['settings']['depth_tolerance'])
    np.testing.assert_array_equal(votes.reshape(-1,7),a['sample_depth_votes'])
    np.testing.assert_array_equal(refs.reshape(-1,7),a['sample_depth_references'])
    remove=ids[eligible[(votes.reshape(-1,7)<q['settings']['minimum_preserving_depth_votes']).all(1)]]
    np.testing.assert_array_equal(remove,a['removed_triangle_ids']);keep=np.ones(len(t),bool);keep[remove]=False
    inset=o3d.io.read_triangle_mesh(str(INSET/frame/'guarded/mesh.ply'))
    assert sha(INSET/frame/'guarded/mesh.ply')==q['inset_mesh_sha256']
    for arm,vertices,triangles in [('pruned',v,t[keep]),('pruned_inset',np.asarray(inset.vertices),
        np.concatenate([t[keep],np.asarray(inset.triangles)[len(t):]]))]:
        mesh=o3d.io.read_triangle_mesh(str(root/arm/'mesh.ply'))
        np.testing.assert_array_equal(vertices,np.asarray(mesh.vertices));np.testing.assert_array_equal(triangles,np.asarray(mesh.triangles))
    atomic_json(root/'audit.json',dict(status='mask_window_depth_and_triangle_assembly_replayed',
        request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),removed_triangles=len(remove),
        independent_mask_window_algorithm=True,depth_evidence_recomputed=True,unchanged_retained_coordinates=True,
        combined_newly_exposed_shell_guard_passed=False,production_promoted=False,script_sha256=sha(__file__)))
    print(frame,'evidence audit passed; removed',len(remove),flush=True)


def render(frame):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    g=read(ROOT/frame/'result.json');audit=read(ROOT/frame/'audit.json')
    assert audit['result_sha256']==sha(ROOT/frame/'result.json')
    rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
    for view in ['moving','native_unmasked']:
        for arm in ['pruned','pruned_inset']:
            mesh=ROOT/frame/arm/'mesh.ply';assert sha(mesh)==g['hashes'][arm+'/mesh.ply']
            q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
            if view=='native_unmasked':
                target=deepcopy(next(r for r in rows if r['physical_camera']==name))
                target['physical_camera']='diagnostic_unmasked_target_'+name;target['reference_physical_camera']=name
                entry['camera']=target
            entry.update(mesh=str(mesh),mesh_sha256=sha(mesh))
            q.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_changed=True,
                background_pruning_arm=arm,geometry_request_sha256=sha(ROOT/frame/'request.json'),
                geometry_result_sha256=sha(ROOT/frame/'result.json'),geometry_audit_sha256=sha(ROOT/frame/'audit.json'),
                source_quality_implementation_sha256=implementation,texture_source_masks_unchanged=True,
                combined_newly_exposed_shell_guard_passed=False,production_promoted=False)
            q['script_hashes'][Path(__file__).name]=sha(__file__)
            out=ROOT/'rgb'/frame/view/arm;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
            if (out/'request.json').exists():assert read(out/'request.json')==q
            atomic_json(out/'request.json',q);engine.render(out,[frame])


def review():
    records=[]
    for frame in FRAMES:
        for view in ['moving','native_unmasked']:
            roots=[MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline',
                INSET/'rgb'/frame/view/'completion',ROOT/'rgb'/frame/view/'pruned',ROOT/'rgb'/frame/view/'pruned_inset']
            images=[];receipts=[];depths=[]
            for root in roots:
                image,r=verified_image(root,frame);images.append(image);receipts.append(r)
                depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
            for r in receipts[1:]:
                for key in ['camera','source_cameras','fixed_exposure']:assert receipts[0][key]==r[key]
            for i,name in [(2,'pruned'),(3,'pruned_inset')]:
                old=depths[i-2]>0;new=depths[i]>0;before=images[i-2];after=images[i]
                records.append(dict(frame=frame,view=view,arm=name,matched_baseline='original' if i==2 else 'inset_only',
                    lost_depth_pixels=int((old&~new).sum()),gained_depth_pixels=int((new&~old).sum()),
                    depth_changed_pixels=int((np.abs(depths[i]-depths[i-2])>1e-6).sum()),
                    introduced_black=int(((before.max(2)>0)&(after.max(2)==0)).sum()),
                    removed_black=int(((before.max(2)==0)&(after.max(2)>0)).sum()),
                    changed_RGB_pixels=int(np.any(before!=after,2).sum()),counts_not_quality_metrics=True))
            labels=['production','inset only','pruned','pruned + inset'];panels=list(images)
            if view=='native_unmasked':
                from PIL import Image
                gt=INSET/'review'/frame/'native_unmasked_gt.png';panels.insert(0,np.asarray(Image.open(gt)));labels.insert(0,'train GT')
            for part,box in [('crown',(170,450,970,850)),('jaw',(400,850,900,1250))]:
                panel(ROOT/'review'/frame/(view+'_'+part+'.png'),panels,labels,box)
    atomic_json(ROOT/'review/result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        production_promoted=False,full_frame_quality_metrics=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','render','review']);p.add_argument('--frame',choices=FRAMES);a=p.parse_args()
    if a.action=='review':review()
    else:
        if not a.frame:p.error('--frame required')
        globals()[a.action](a.frame)
