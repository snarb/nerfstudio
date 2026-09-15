"""Review frozen-method transfer at earlier face/hair canaries, not a new movie."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT as COLOR,exr,display
from run_neighborhood_completion_transfer import ROOT
from study_confidence_depth_prior import REGIONS,region_masks
from review_jaw_repair_transfer import verified_image,panel


def render_region(frame):
    import render_smooth_temporal_mesh_video as renderer
    from study_native_texture_footprint import install
    implementation=install();renderer.torch.set_num_threads(2)
    output=ROOT/frame;mesh=output/'interpolated'/frame/'mesh.ply';result=read(mesh.parent/'result.json')
    assert result['observed_guard_passed'] and sha(mesh)==result['hashes']['mesh.ply']
    parent=renderer.verify_request(Path('/mnt/data/dec5_phase30_early_texture_dynamic_150'))
    rows,_,_=cameras(frame);name=REGIONS[frame]['camera']
    for variant in ['baseline','repaired']:
        req=deepcopy(parent);req['inventory']=[r for r in req['inventory'] if r['frame_id']==frame];row=req['inventory'][0]
        row['camera']=next(r for r in rows if r['physical_camera']==name)
        if variant=='repaired':row.update(mesh=str(mesh),mesh_sha256=sha(mesh))
        req.update(partial_diagnostic_only=True,full_video_candidate=False,geometry_result_sha256=sha(mesh.parent/'result.json'),
            native_footprint_implementation_sha256=implementation,same_footprint_for_both_geometry_variants=True)
        for n in ['study_head_neighborhood_transfer.py','study_native_texture_footprint.py','native_texture_footprint.py']:
            req['script_hashes'][n]=sha(Path(__file__).with_name(n))
        dest=output/'region_rgb'/variant;dest.mkdir(parents=True,exist_ok=False);(dest/'frames').mkdir()
        atomic_json(dest/'request.json',req);renderer.render(dest,[frame])


def review(frame):
    output=ROOT/frame;dest=output/'head_review';dest.mkdir(exist_ok=False);records=[]
    rows,_,_=cameras(frame);profile=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(profile-profile.mean(0,keepdims=True));exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    for view in ['moving','F004_E005_1210FP',REGIONS[frame]['camera']]:
        roots=[output/'region_rgb'/v for v in ['baseline','repaired']] if view==REGIONS[frame]['camera'] else [output/'rgb'/view/v for v in ['baseline','repaired']]
        images=[];metadata=[];depth=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);metadata.append(r)
            depth.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        for k in ['camera','source_cameras','fixed_exposure']:assert metadata[0][k]==metadata[1][k]
        record=dict(view=view,inputs={str(p/'frames'/frame/'frame.png'):sha(p/'frames'/frame/'frame.png') for p in roots},
                    changed_rgb_pixels=int(np.any(images[0]!=images[1],axis=2).sum()))
        labels=['production + fixed footprint','completion + fixed footprint']
        if view!='moving':
            i=next(i for i,r in enumerate(rows) if r['physical_camera']==view);row=rows[i]
            gt=np.rot90(np.rint(display(exr(row['file_path'])*gain[i],exposure)*255).clip(0,255).astype(np.uint8)).copy()
            gtpath=dest/(view+'_gt.png');Image.fromarray(gt).save(gtpath)
            record.update(source_gt_sha256=sha(row['file_path']),display_gt_sha256=sha(gtpath),
                profile_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'))
            if view==REGIONS[frame]['camera']:
                regions={k:np.rot90(m) for k,m in region_masks(frame).items()}
                for name,mask in regions.items():
                    y,x=np.where(mask);box=(max(int(x.min())-20,0),max(int(y.min())-20,0),min(int(x.max())+21,1080),min(int(y.max())+21,1920))
                    record[name]=dict(pixels=int(mask.sum()),depth_misses=[int((mask&(d==0)).sum()) for d in depth],
                        black_rgb=[int((mask&(im.max(2)==0)).sum()) for im in images])
                    panel(dest/(name+'_native.png'),[gt,*images],['real train GT',*labels],box)
            images=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_head.png'),images,labels,(100,400,1060,1350));records.append(record)
    atomic_json(dest/'result.json',dict(frame=frame,script_sha256=sha(__file__),records=records,visual_status='pending',
        fixed_regions=REGIONS[frame],region_role='previous train annotations; coarse hair zero-depth is not anatomical hole count',
        full_frame_quality_metrics=False,heldout_used=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','review']);p.add_argument('--frame',choices=['001083','001123'],required=True)
    a=p.parse_args();{'render':render_region,'review':review}[a.action](a.frame)
