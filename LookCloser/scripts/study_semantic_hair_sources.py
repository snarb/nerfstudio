"""Train-consensus hair-only hard-source repair; preserve baseline elsewhere.

No semantic target mask or RGB averaging. Source images, poses, mesh, depth,
and graph labels are frozen. Only uncertain hair source pixels may switch to
a visible interior observation classified as hair by both train-image models.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from joint_temporal_texture import read,sha,atomic_json
from diagnose_cinematic_hair_rim import signed_distance,BASE
from build_train_hair_semantics import ROOT as SEMANTICS,FRAMES,CROP

ROOT=Path('/mnt/data/dec5_semantic_hair_sources')
SETTINGS=dict(probability_threshold=.75,minimum_hair_views=3,hair_fraction=.5,
    protected_veto_views=3,interior_margin_pixels=16.,keep_clean_original_source=True)


def select_hair_sources(quality,visible,semantics,known,margin,baseline):
    """All arrays are camera-by-point, except Cx3xN semantics and N baseline."""
    c,n=quality.shape
    if visible.shape!=(c,n) or semantics.shape!=(c,3,n) or known.shape!=(c,n) or margin.shape!=(c,n) or baseline.shape!=(n,):
        raise ValueError('Mismatched source evidence')
    if not torch.isfinite(quality).all() or not torch.isfinite(semantics).all() or not torch.isfinite(margin).all():
        raise ValueError('Non-finite source evidence')
    if ((quality<0).any() or (semantics<0).any() or (semantics>1.00001).any()
        or ((baseline!=255)&((baseline<0)|(baseline>=c))).any()):raise ValueError('Invalid source evidence')
    available=visible&known
    hair=(semantics[:,0]>=.75)&(semantics[:,1]>=.75)&available
    protected=(semantics[:,2]>=.75)&available
    hair_count=hair.sum(0);protected_count=protected.sum(0)
    gate=(hair_count>=3)&(2*hair_count>=available.sum(0))&(protected_count<3)
    interior=hair&(margin>=16)&(quality>0)
    filtered=quality*interior
    best=filtered.argmax(0);has_source=filtered.max(0).values>0
    index=baseline.clamp(0,c-1).long();clean=interior.gather(0,index[None])[0]
    replace=gate&has_source&~clean&(baseline!=255)
    selected=torch.where(replace,best,baseline)
    return selected,gate,hair_count,protected_count


def install():
    import render_smooth_temporal_mesh_video as engine
    import study_native_texture_footprint as footprint
    from view_consistent_source_quality import quality
    from study_unwarped_head_texture import zero_registration
    from wide_dynamic_camera_flight import install_source_masks
    source=footprint.transform(inspect.getsource(engine.render_one))
    first=source.index("    status('surface_labels')")
    last=source.index("    status('target_rgb')")
    source=source[:first]+"""    labels=np.load(_semantic_parent/'frames'/frame/'face_source_labels.npy')
    graph=read(_semantic_parent/'frames'/frame/'result.json')['graph']
    np.save(target/'face_source_labels.npy',labels)
    baseline_sources=np.asarray(Image.open(_semantic_parent/'frames'/frame/'source_ids.png')).ravel()
"""+source[last:]
    edits={
        'np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2':"_view_quality((direction*normal[f]).sum(-1),length,'incidence2')*angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0][:,None]",
        "chosen,source,fallback=gather_hard_rgb(colors,torch.tensor(quality,device='cuda')*valid,torch.tensor(labels[f],device='cuda'))": "chosen,source,fallback=_semantic_gather(colors,torch.tensor(quality,device='cuda')*valid,valid,q,torch.tensor(baseline_sources[pixels[s:end]],device='cuda'),pixels[s:end])",
    }
    for old,new in edits.items():
        if source.count(old)!=1:raise ValueError('Changed source statement: '+old)
        source=source.replace(old,new)
    engine.__dict__.update(_semantic_parent=BASE,_view_quality=quality,angle_weights=footprint.angle_weights,
        _snap_centers=footprint.snap_centers,_relevant_tap=footprint.relevant_tap,
        sample=footprint.sample_native,bounded_warp=zero_registration)
    exec(compile(source,__file__+':semantic_source_control','exec'),engine.__dict__)
    install_source_masks(engine);original=engine.render_one
    def wrapped(output,record,source_manifest):
        frame=record['frame_id'];rows,_,_=engine.cameras(frame)
        parent=BASE/'frames'/frame;receipt=read(parent/'complete.json')
        assert receipt['request_sha256']==sha(BASE/'request.json')
        for p,h in receipt['hashes'].items():assert sha(parent/p)==h
        semantics=SEMANTICS/frame;sr=read(semantics/'complete.json')
        assert sr['request_sha256']==sha(SEMANTICS/'request.json')
        by_name={r['camera']:r for r in sr['records']};assert len(by_name)==62
        arrays=[]
        for row in rows:
            name=row['physical_camera'];path=semantics/'predictions'/(name+'.npz')
            assert sha(path)==by_name[name]['output_sha256']
            arrays.append(np.load(path)['confidence'])
        maps=torch.from_numpy(np.stack(arrays).astype(np.float32)/255).cuda();del arrays
        maskroot=Path(record['source_masks']['root']);assert sha(maskroot/'masks.npz')==record['source_masks']['masks_sha256']
        masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
        distances=torch.from_numpy(np.stack([signed_distance(masks[names.index(r['physical_camera'])]) for r in rows])[:,None]).cuda()
        evidence=np.zeros((4,1080*1920),np.uint8)
        def gather(colors,quality,visible,uv,baseline,pixels):
            crop_uv=torch.stack((uv[...,1]-CROP[0],1919-uv[...,0]-CROP[1]),-1)
            known=((crop_uv[:,0,:,0]>=0)&(crop_uv[:,0,:,0]<=1079)&(crop_uv[:,0,:,1]>=0)&(crop_uv[:,0,:,1]<=1249))
            grid=(crop_uv+.5)*crop_uv.new_tensor([2/1080,2/1250])-1
            predictions=F.grid_sample(maps,grid,align_corners=False,padding_mode='zeros')[:,:,0]
            margin=footprint.sample_native(distances,uv)[:,0,0]
            selected,gate,hair,protected=select_hair_sources(quality,visible,predictions,known,margin,baseline.long())
            valid=selected!=255;index=selected.clamp(0,len(rows)-1)
            chosen=colors.permute(2,0,1)[torch.arange(len(index),device='cuda'),index].T
            chosen=torch.where(valid[None],chosen,0)
            changed=selected!=baseline
            evidence[:,pixels]=torch.stack((gate,changed,hair,protected)).to(torch.uint8).cpu().numpy()
            return chosen,selected,changed
        engine._semantic_gather=gather
        try:
            result=original(output,record,source_manifest)
            folder=output/'frames'/frame
            np.savez_compressed(folder/'semantic_decisions.npz',gate=evidence[0].reshape(1080,1920),
                changed=evidence[1].reshape(1080,1920),hair_votes=evidence[2].reshape(1080,1920),protected_votes=evidence[3].reshape(1080,1920))
            complete=read(folder/'complete.json');complete['hashes']['semantic_decisions.npz']=sha(folder/'semantic_decisions.npz')
            atomic_json(folder/'complete.json',complete)
            return result
        finally:del engine._semantic_gather,maps,distances
    engine.render_one=wrapped
    return hashlib.sha256(source.encode()).hexdigest()


def render():
    import render_smooth_temporal_mesh_video as engine
    engine.torch.set_num_threads(2);q=deepcopy(engine.verify_request(BASE));implementation=install()
    q['inventory']=[r for r in q['inventory'] if r['frame_id'] in FRAMES]
    q.update(semantic_source_control=SETTINGS,semantic_request_sha256=sha(SEMANTICS/'request.json'),
        semantic_frame_bindings={f:sha(SEMANTICS/f/'complete.json') for f in FRAMES},
        semantic_implementation_sha256=implementation,matched_parent_sha256=sha(BASE/'request.json'),
        geometry_changed=False,full_video_candidate=False,production_promoted=False,
        graph_labels_reused=True,preserve_original_non_hair_sources=True,partial_diagnostic_only=True)
    for name in [Path(__file__).name,'build_train_hair_semantics.py','diagnose_cinematic_hair_rim.py']:
        q['script_hashes'][name]=sha(Path(__file__).with_name(name))
    ROOT.mkdir(exist_ok=True);(ROOT/'frames').mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():assert read(ROOT/'request.json')==q
    else:atomic_json(ROOT/'request.json',q)
    engine.render(ROOT,FRAMES)


def review():
    from PIL import Image
    from review_jaw_repair_transfer import verified_image,panel
    records=[]
    for frame in FRAMES:
        before,old=verified_image(BASE,frame);after,new=verified_image(ROOT,frame)
        a=BASE/'frames'/frame;b=ROOT/'frames'/frame
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert old[key]==new[key]
        np.testing.assert_array_equal(np.load(a/'target_depth.npz')['depth'],np.load(b/'target_depth.npz')['depth'])
        np.testing.assert_array_equal(np.load(a/'face_source_labels.npy'),np.load(b/'face_source_labels.npy'))
        ids=[np.asarray(Image.open(p/'source_ids.png')) for p in [a,b]]
        decisions=np.load(b/'semantic_decisions.npz');changed=decisions['changed'].astype(bool)
        assert np.array_equal(ids[0]==255,ids[1]==255) and np.array_equal(changed,ids[0]!=ids[1])
        assert not (changed&~decisions['gate'].astype(bool)).any()
        old_rgb=np.asarray(Image.open(a/'prediction_native.png'));new_rgb=np.asarray(Image.open(b/'prediction_native.png'))
        np.testing.assert_array_equal(old_rgb[~changed],new_rgb[~changed])
        for name,box in [('crown',(120,0,1080,520)),('face',(190,500,1080,1320)),('body',(0,1150,1080,1920))]:
            panel(ROOT/'review'/f'{frame}_{name}.png',[before,after],['published source choice','train-consensus hair sources'],box)
        records.append(dict(frame=frame,changed_source_pixels=int(changed.sum()),
            changed_rgb_pixels=int(np.any(old_rgb!=new_rgb,2).sum()),depth_equal=True,graph_labels_equal=True,
            unchanged_source_pixels_rgb_exact=True,source_missing_mask_equal=True,
            protected_pixels_changed=int((changed&(decisions['protected_votes']>=3)).sum()),
            black_introduced=int(((old_rgb.max(2)>0)&(new_rgb.max(2)==0)).sum())))
    atomic_json(ROOT/'review'/'result.json',dict(records=records,visual_status='pending',production_promoted=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);a=p.parse_args()
    render() if a.action=='render' else review()
