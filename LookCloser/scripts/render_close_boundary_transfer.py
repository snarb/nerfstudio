"""Matched current-recipe native RGB controls for earlier-time crack transfer."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT as CAL
from transfer_close_boundary_completion import ROOT,MOVIE,FRAMES
from review_jaw_repair_transfer import verified_image,panel


def render():
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2)
    parent=engine.verify_request(MOVIE)
    for frame in FRAMES:
        geometry=ROOT/frame/'interpolated'/frame
        r=read(geometry/'result.json');audit=read(geometry/'audit.json')
        complete=read(ROOT/frame/'controller_complete.json')
        assert complete['result_sha256']==sha(geometry/'result.json') and complete['audit_sha256']==sha(geometry/'audit.json')
        assert r['observed_guard_passed'] and audit['mesh_sha256']==sha(geometry/'mesh.ply')
        rows,_,_=cameras(frame)
        for view in ['moving','F004_E005_1210FP']:
            for variant in ['baseline','completion']:
                if view=='moving' and variant=='baseline':
                    continue
                q=deepcopy(parent);q['inventory']=[r for r in q['inventory'] if r['frame_id']==frame];entry=q['inventory'][0]
                if view!='moving':
                    entry['camera']=next(r for r in rows if r['physical_camera']==view)
                if variant=='completion':
                    entry.update(mesh=str(geometry/'mesh.ply'),mesh_sha256=sha(geometry/'mesh.ply'))
                q.update(partial_diagnostic_only=True,full_video_candidate=False,
                    geometry_changed=variant=='completion',geometry_transfer_variant=variant,
                    source_quality_implementation_sha256=implementation,
                    geometry_audit_sha256=sha(geometry/'audit.json'),
                    transfer_controller_request_sha256=sha(ROOT/(frame+'_controller_request.json')))
                q['script_hashes'][Path(__file__).name]=sha(__file__)
                dest=ROOT/'rgb'/frame/view/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
                if (dest/'request.json').exists() and read(dest/'request.json')!=q:
                    raise ValueError('Changed RGB request')
                atomic_json(dest/'request.json',q);engine.render(dest,[frame])


def review():
    records=[]
    profiles=read(CAL/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(CAL/'exposure.json')['fixed_exposure_gain']
    for frame in FRAMES:
        rows,_,_=cameras(frame)
        for view in ['moving','F004_E005_1210FP']:
            roots=[MOVIE if view=='moving' else ROOT/'rgb'/frame/view/'baseline',ROOT/'rgb'/frame/view/'completion']
            ims=[];rs=[];depths=[]
            for root in roots:
                im,r=verified_image(root,frame);ims.append(im);rs.append(r)
                depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
            for key in ['camera','source_cameras','fixed_exposure']:
                assert rs[0][key]==rs[1][key]
            hit0,hit1=depths[0]>0,depths[1]>0
            record=dict(frame=frame,view=view,new_depth_pixels=int((hit1&~hit0).sum()),
                lost_depth_pixels=int((hit0&~hit1).sum()),
                changed_rgb_pixels=int(np.any(ims[0]!=ims[1],2).sum()),
                new_black_pixels=int(((ims[0].max(2)>0)&(ims[1].max(2)==0)).sum()),
                removed_black_pixels=int(((ims[0].max(2)==0)&(ims[1].max(2)>0)).sum()),
                inputs={str(root/'frames'/frame/'complete.json'):sha(root/'frames'/frame/'complete.json') for root in roots},
                counts_are_integrity_not_anatomical_metrics=True)
            labels=['production','close-boundary transfer'];images=list(ims)
            if view!='moving':
                row=next(r for r in rows if r['physical_camera']==view)
                gt=np.rot90(np.rint(display(exr(row['file_path'])*gains[view],exposure)*255).clip(0,255).astype(np.uint8)).copy()
                gtpath=ROOT/'review'/frame/(view+'_gt.png');gtpath.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(gt).save(gtpath)
                record['gt_input_sha256']=sha(row['file_path']);record['gt_sha256']=sha(gtpath)
                images.insert(0,gt);labels.insert(0,'real train GT')
            for name,box in [('head',(170,450,970,1350)),('jaw',(400,850,900,1250)),('crown',(170,450,970,850))]:
                panel(ROOT/'review'/frame/(view+'_'+name+'.png'),images,labels,box)
            overlay=ims[1].copy();overlay[hit1&~hit0]=[255,50,50]
            overlay[np.any(ims[0]!=ims[1],2)&hit0&hit1]=[50,255,100]
            panel(ROOT/'review'/frame/(view+'_change_map.png'),[ims[0],overlay],
                ['production','red new depth / green RGB change'],(170,450,970,1350))
            records.append(record)
    atomic_json(ROOT/'review/result.json',dict(records=records,visual_status='pending',full_frame_quality_metrics=False,
        script_sha256=sha(__file__),production_updated=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);a=p.parse_args()
    render() if a.action=='render' else review()
