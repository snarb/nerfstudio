"""Matched geometry-only mask control; unchanged times reuse verified RGB."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT,exr,display
from review_jaw_repair_transfer import verified_image,panel
from study_jaw_repair_transfer import PARENT

BASE=Path('/mnt/data/dec5_jaw_repair_transfer')
OUT=Path('/mnt/data/dec5_jaw_measured_mask_control')
CAMERA='D004_D005_1210LZ'


def reuse(output):
    for frame in ['001083','001123','001195']:
        if sha(output/frame/'mesh.ply')!=sha(BASE/frame/'mesh.ply'):raise ValueError('Changed time cannot reuse RGB')
        for view in ['moving','F004_E005_1210FP']:
            for variant in ['baseline','repaired']:verified_image(BASE/'rgb'/frame/view/variant,frame)
        link=output/'rgb'/frame;link.parent.mkdir(exist_ok=True)
        if link.exists():
            if not link.is_symlink() or link.resolve()!=(BASE/'rgb'/frame).resolve():raise ValueError('Unexpected reuse destination')
        else:link.symlink_to(BASE/'rgb'/frame,target_is_directory=True)
        atomic_json(output/frame/'rgb_reuse.json',dict(frame=frame,source=str(BASE/'rgb'/frame),
            source_result_sha256=sha(BASE/frame/'result.json'),current_result_sha256=sha(output/frame/'result.json'),
            mesh_bytes_exact=True,rgb_rerendered=False))


def render_veto(output):
    import render_smooth_temporal_mesh_video as renderer
    from study_early_texture_prior import install
    install();renderer.torch.set_num_threads(2);parent=renderer.verify_request(PARENT);frame='001193'
    rows,_,_=cameras(frame)
    for variant,root in [('previous',BASE),('updated',output)]:
        request=deepcopy(parent);request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
        row=request['inventory'][0];row.update(camera=next(r for r in rows if r['physical_camera']==CAMERA),
            mesh=str(root/frame/'mesh.ply'),mesh_sha256=sha(root/frame/'mesh.ply'))
        request.update(partial_diagnostic_only=True,full_video_candidate=False,mask_control_variant=variant,
            geometry_result_sha256=sha(root/frame/'result.json'),source_texture_masks_unchanged=True)
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        dest=output/'veto_camera'/variant;dest.mkdir(parents=True,exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
        if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Changed veto-view request')
        atomic_json(dest/'request.json',request);renderer.render(dest,[frame])


def comparisons(output):
    frame='001193';dest=output/'matched_review';dest.mkdir(exist_ok=True);records=[]
    for view in ['moving','F004_E005_1210FP',CAMERA]:
        roots=[BASE/'rgb'/frame/view/'repaired',output/'rgb'/frame/view/'repaired'] if view!=CAMERA else [output/'veto_camera/previous',output/'veto_camera/updated']
        images=[];results=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);results.append(r)
        for key in ['camera','source_cameras','fixed_exposure']:
            if results[0][key]!=results[1][key]:raise ValueError('Unmatched mask control')
        changed=int(np.any(images[0]!=images[1],2).sum());labels=['previous repair','measured-mask repair']
        if view!='moving':
            rows,_,_=cameras(frame);row=next(r for r in rows if r['physical_camera']==view)
            parameters=np.load(ROOT/'parameters.npz')['log_gain'];index=next(i for i,r in enumerate(rows) if r['physical_camera']==view)
            gt=np.rot90(np.rint(display(exr(row['file_path'])*np.exp(parameters[index]),read(ROOT/'exposure.json')['fixed_exposure_gain'])*255).clip(0,255).astype(np.uint8)).copy()
            images=[gt,*images];labels=['real train GT',*labels]
        p=dest/(view+'_head.png');panel(p,images,labels,(170,450,970,1350))
        box=(570,1070,780,1240) if view=='F004_E005_1210FP' else ((480,1080,740,1260) if view==CAMERA else (370,1080,670,1320))
        q=dest/(view+'_detail.png');panel(q,images,labels,box)
        records.append(dict(view=view,changed_rgb_pixels=changed,paths={str(p):sha(p),str(q):sha(q)},
            source_prediction_sha256=[sha(r/'frames'/frame/'frame.png') for r in roots]))
    atomic_json(dest/'result.json',dict(records=records,geometry_only=True,texture_masks_unchanged=True,
        script_sha256=sha(__file__),visual_status='requires_actual_review',full_frame_quality_metrics=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['reuse','render_veto','panels'])
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    {'reuse':reuse,'render_veto':render_veto,'panels':comparisons}[a.action](a.output)
