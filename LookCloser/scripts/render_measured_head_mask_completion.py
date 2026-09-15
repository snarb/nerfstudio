"""Current-recipe controls; geometry masks refined, RGB source masks unchanged.

Native train diagnostic targets use a synthetic physical-camera label at the
EXACT real pose/intrinsics. This prevents the source-mask wrapper from masking
the target raycast itself; both native baseline and candidate follow this rule.
The actual source cameras and their masks retain their real physical names.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT as CAL
from transfer_close_boundary_completion import MOVIE,FRAMES
from apply_measured_head_mask_completion import ROOT
from study_confidence_depth_prior import REGIONS
from review_jaw_repair_transfer import verified_image,panel


def render():
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2);parent=engine.verify_request(MOVIE)
    for frame in FRAMES:
        geometry=ROOT/frame/'interpolated'/frame;g=read(geometry/'result.json');ga=read(geometry/'audit.json')
        assert g['observed_guard_passed'] and ga['mesh_sha256']==g['hashes']['mesh.ply']==sha(geometry/'mesh.ply')
        rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
        for view in ['moving','native_unmasked']:
            for variant in ['baseline','completion']:
                if view=='moving' and variant=='baseline':
                    continue
                q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
                if view=='native_unmasked':
                    target=deepcopy(next(r for r in rows if r['physical_camera']==name))
                    target['physical_camera']='diagnostic_unmasked_target_'+name
                    target['reference_physical_camera']=name;entry['camera']=target
                if variant=='completion':
                    entry.update(mesh=str(geometry/'mesh.ply'),mesh_sha256=sha(geometry/'mesh.ply'))
                q.update(partial_diagnostic_only=True,full_video_candidate=False,
                    geometry_changed=variant=='completion',measured_semantic_completion_variant=variant,
                    source_quality_implementation_sha256=implementation,
                    semantic_controller_request_sha256=sha(ROOT/frame/'controller_request.json'),
                    native_target_mask_disabled=view=='native_unmasked',texture_source_masks_unchanged=True,
                    actual_native_physical_camera=name if view=='native_unmasked' else None,
                    geometry_audit_sha256=sha(geometry/'audit.json'))
                q['script_hashes'][Path(__file__).name]=sha(__file__)
                dest=ROOT/'rgb'/frame/view/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
                if (dest/'request.json').exists() and read(dest/'request.json')!=q:
                    raise ValueError('Changed measured-mask RGB request')
                atomic_json(dest/'request.json',q);engine.render(dest,[frame])


def review():
    gains=read(CAL/'camera_profiles.json');gainmap=dict(zip(gains['physical_cameras'],gains['rgb_gain']))
    exposure=read(CAL/'exposure.json')['fixed_exposure_gain'];records=[]
    for frame in FRAMES:
        rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
        for view in ['moving','native_unmasked']:
            roots=[MOVIE if view=='moving' else ROOT/'rgb'/frame/view/'baseline',ROOT/'rgb'/frame/view/'completion']
            images=[];receipts=[];depths=[]
            for root in roots:
                im,r=verified_image(root,frame);images.append(im);receipts.append(r)
                depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
            for key in ['camera','source_cameras','fixed_exposure']:
                assert receipts[0][key]==receipts[1][key]
            old,new=depths[0]>0,depths[1]>0
            record=dict(frame=frame,view=view,new_depth_pixels=int((new&~old).sum()),lost_depth_pixels=int((old&~new).sum()),
                new_black_pixels=int(((images[0].max(2)>0)&(images[1].max(2)==0)).sum()),
                removed_black_pixels=int(((images[0].max(2)==0)&(images[1].max(2)>0)).sum()),
                changed_rgb_pixels=int(np.any(images[0]!=images[1],2).sum()),
                counts_not_anatomical_metrics=True,inputs={str(root/'request.json'):sha(root/'request.json') for root in roots})
            overlay=images[1].copy();overlay[new&~old]=[255,50,50]
            overlay[np.any(images[0]!=images[1],2)&old&new]=[50,255,100]
            labels=['production mesh','measured-mask geometry'];display_images=list(images)
            if view=='native_unmasked':
                row=next(r for r in rows if r['physical_camera']==name)
                gt=np.rot90(np.rint(display(exr(row['file_path'])*gainmap[name],exposure)*255).clip(0,255).astype(np.uint8)).copy()
                gtpath=ROOT/'review'/frame/(view+'_gt.png');gtpath.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(gt).save(gtpath)
                record['gt_sha256']=sha(gtpath);record['source_rgb_sha256']=sha(row['file_path'])
                display_images.insert(0,gt);labels.insert(0,'real train GT')
            for part,box in [('head',(170,450,970,1350)),('crown',(170,450,970,850)),('jaw',(400,850,900,1250))]:
                panel(ROOT/'review'/frame/(view+'_'+part+'.png'),display_images,labels,box)
            panel(ROOT/'review'/frame/(view+'_changes.png'),[images[0],overlay],
                ['production mesh','red new depth / green RGB change'],(170,450,970,1350))
            records.append(record)
    atomic_json(ROOT/'review/result.json',dict(records=records,visual_status='pending',production_updated=False,
        source_masks_unchanged=True,full_frame_quality_metrics=False,script_sha256=sha(__file__)))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);a=p.parse_args()
    render() if a.action=='render' else review()
