"""Opt-in hard-source control: prefer visible silhouette-interior observations.

This is a testable proxy for avoiding background-bearing sparse hair pixels,
not opacity estimation or a geometry fix. Preserve all visibility gates and
use the old admissible set when no sufficiently interior source is available.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import inspect
import numpy as np
import torch
from joint_temporal_texture import read, sha, atomic_json
from diagnose_cinematic_hair_rim import signed_distance, BASE

ROOT = Path('/mnt/data/dec5_interior_texture_sources')
MARGIN = 16.
FRAMES = ['001083', '001123']


def prefer_interior(quality, margin, threshold=MARGIN):
    quality = np.asarray(quality)
    margin = np.asarray(margin)
    if quality.ndim != 2 or quality.shape != margin.shape:
        raise ValueError('Expected matching camera-by-point arrays')
    if not np.isfinite(quality).all() or not np.isfinite(margin).all() or (quality < 0).any() or threshold <= 0:
        raise ValueError('Invalid source evidence')
    good = (margin >= threshold) & (quality > 0)
    return np.where(good | ~good.any(0, keepdims=True), quality, 0)


def install():
    import render_smooth_temporal_mesh_video as engine
    import study_native_texture_footprint as footprint
    from view_consistent_source_quality import quality
    from study_unwarped_head_texture import zero_registration
    from wide_dynamic_camera_flight import install_source_masks
    source = footprint.transform(inspect.getsource(engine.render_one))
    edits = {
        'np.abs((directions*normal).sum(-1))**8/length.clip(.01)**2': "_view_quality((directions*normal).sum(-1),length,'incidence2')",
        'np.abs((direction*normal[f]).sum(-1))**8/length.clip(.01)**2': "_view_quality((direction*normal[f]).sum(-1),length,'incidence2')*angle_weights(rows,record['camera'],read(output/'request.json')['recipe']['target_angle_sigma_degrees'])[0][:,None]",
        'quality=_early_quality(quality,': 'quality=_early_quality(_interior_filter(quality,q),',
        "chosen,source,fallback=gather_hard_rgb(colors,torch.tensor(quality,device='cuda')*valid,": "chosen,source,fallback=gather_hard_rgb(colors,torch.tensor(_interior_filter(quality*valid.cpu().numpy(),q),device='cuda'),",
    }
    for old, new in edits.items():
        if source.count(old) != 1:raise ValueError('Changed renderer substitution: '+old)
        source = source.replace(old,new)
    engine.__dict__.update(_view_quality=quality,_early_quality=footprint.early_quality,
        angle_weights=footprint.angle_weights,_snap_centers=footprint.snap_centers,
        _relevant_tap=footprint.relevant_tap,sample=footprint.sample_native,bounded_warp=zero_registration)
    exec(compile(source,__file__+':interior_source_control','exec'),engine.__dict__)
    install_source_masks(engine)
    original = engine.render_one
    def wrapped(output,record,manifest):
        maskroot=Path(record['source_masks']['root'])
        assert sha(maskroot/'masks.npz')==record['source_masks']['masks_sha256']
        masks=np.load(maskroot/'masks.npz')['masks'];names=read(maskroot/'cameras.json')
        rows,_,_=engine.cameras(record['frame_id'])
        distances=torch.from_numpy(np.stack([signed_distance(masks[names.index(r['physical_camera'])]) for r in rows])[:,None]).cuda()
        def filtered(q,uv):
            margins=footprint.sample_native(distances,uv)[:,0,0].cpu().numpy()
            return prefer_interior(q,margins)
        engine._interior_filter=filtered
        try:return original(output,record,manifest)
        finally:del engine._interior_filter,distances
    engine.render_one=wrapped
    return hashlib.sha256(source.encode()).hexdigest()


def run():
    import render_smooth_temporal_mesh_video as engine
    engine.torch.set_num_threads(2)
    q=deepcopy(engine.verify_request(BASE));implementation=install()
    q['inventory']=[r for r in q['inventory'] if r['frame_id'] in FRAMES]
    assert len(q['inventory'])==2
    q.update(interior_source_control=dict(native_mask_margin=MARGIN,implementation_sha256=implementation,
        fallback='retain original admissible sources if no interior one is visible',
        applies_to='all mesh surface centroids and all target surface samples',opacity_estimated=False),
        matched_parent=str(BASE),matched_parent_sha256=sha(BASE/'request.json'),
        partial_diagnostic_only=True,full_video_candidate=False,production_promoted=False,
        geometry_changed=False,target_camera_changed=False,averages_rgb=False)
    for name in [Path(__file__).name,'diagnose_cinematic_hair_rim.py']:
        q['script_hashes'][name]=sha(Path(__file__).with_name(name))
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'frames').mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():assert read(ROOT/'request.json')==q
    atomic_json(ROOT/'request.json',q);engine.render(ROOT,FRAMES)


def review():
    from review_jaw_repair_transfer import verified_image,panel
    result=[]
    for frame in FRAMES:
        old,ro=verified_image(BASE,frame);new,rn=verified_image(ROOT,frame)
        for key in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert ro[key]==rn[key]
        np.testing.assert_array_equal(np.load(BASE/'frames'/frame/'target_depth.npz')['depth'],np.load(ROOT/'frames'/frame/'target_depth.npz')['depth'])
        for name,box in [('crown',(120,0,1080,520)),('face',(190,500,1080,1320)),('body',(0,1150,1080,1920))]:
            panel(ROOT/'review'/f'{frame}_{name}.png',[old,new],['published source choice','visible interior preference'],box)
        from PIL import Image
        before=np.asarray(Image.open(BASE/'frames'/frame/'source_ids.png'))
        after=np.asarray(Image.open(ROOT/'frames'/frame/'source_ids.png'))
        # Both policies can select some visible source or neither; the preference
        # must never turn an admitted pixel into a source-less pixel.
        assert np.array_equal(before==255,after==255)
        result.append(dict(frame=frame,depth_equal=True,source_missing_mask_equal=True,
            changed_source_pixels=int((before!=after).sum()),changed_rgb_pixels=int(np.any(old!=new,2).sum()),
            black_introduced=int(((old.max(2)>0)&(new.max(2)==0)).sum())))
    atomic_json(ROOT/'review'/'result.json',dict(records=result,script_sha256=sha(__file__),
        visual_status='pending',production_promoted=False,no_image_quality_metrics=True))
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['render','review']);args=parser.parse_args()
    run() if args.action=='render' else review()
