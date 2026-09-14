"""Exact-wrapper jaw comparisons and one independently frozen held-out face ROI."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT
from study_jaw_repair_transfer import OUT,FRAMES

HELD=Path('/mnt/data/dec5_early_texture_prior_heldout')
GT=Path('/mnt/data/dec5_gradient_heldout_fidelity/evaluation/001193')


def verified_image(root,frame):
    folder=root/'frames'/frame; complete=read(folder/'complete.json')
    if complete['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed render request')
    for p,h in complete['hashes'].items():
        if sha(folder/p)!=h:raise ValueError('Changed render artifact')
    result=read(folder/'result.json')
    if result['target_rgb_read'] or result['rgb_averaging']:raise ValueError('RGB protocol violation')
    return np.array(Image.open(folder/'frame.png')),result


def panel(path,images,names,box):
    w,h=box[2]-box[0],box[3]-box[1];im=Image.new('RGB',(len(images)*w,h+24));draw=ImageDraw.Draw(im)
    for i,(array,name) in enumerate(zip(images,names)):
        im.paste(Image.fromarray(array).crop(box),(i*w,24));draw.text((i*w+3,3),name,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)


def review(output,frame):
    dest=output/'review'/frame;dest.mkdir(parents=True,exist_ok=True)
    pairs=[];notes=[]
    for view in ['moving','F004_E005_1210FP']:
        images=[];records=[];depths=[]
        for variant in ['baseline','repaired']:
            root=output/'rgb'/frame/view/variant
            image,result=verified_image(root,frame);images.append(image);records.append(result)
            depths.append(np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth']))
        for key in ['camera','source_cameras','fixed_exposure']:
            if records[0][key]!=records[1][key]:raise ValueError('Unmatched pair: '+key)
        names=['baseline','repaired'];input_hashes={}
        for variant in ['baseline','repaired']:
            p=output/'rgb'/frame/view/variant/'frames'/frame/'frame.png';input_hashes[str(p)]=sha(p)
        if view!='moving':
            rows,_,_=cameras(frame);row=next(r for r in rows if r['physical_camera']==view)
            profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
            gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
            gt=np.rot90(np.rint(display(exr(row['file_path'])*gains[view],gain)*255).clip(0,255).astype(np.uint8)).copy()
            Image.fromarray(gt).save(dest/(view+'_gt.png'));input_hashes[row['file_path']]=sha(row['file_path'])
            images=[gt,*images];names=['real train GT',*names]
        # Broad native head and jaw review; no crop-derived geometry or quality metric.
        path=dest/(view+'_head.png');panel(path,images,names,(170,450,970,1250));pairs.append(dict(path=str(path),sha256=sha(path)))
        jaw=dest/(view+'_jaw.png');panel(jaw,images,names,(400,850,900,1200));pairs.append(dict(path=str(jaw),sha256=sha(jaw)))
        box=(400,850,900,1200);x0,y0,x1,y1=box
        notes.append(dict(view=view,input_hashes=input_hashes,
            changed_rgb_pixels=int(np.any(images[-2]!=images[-1],2).sum()),
            jaw_box=box,depth_misses=[int((d[y0:y1,x0:x1]==0).sum()) for d in depths],
            diagnostic_box_contains_true_background_not_missing_anatomy_count=True))
    atomic_json(dest/'receipt.json',dict(frame=frame,panels=pairs,comparisons=notes,
        full_frame_quality_metrics=False,visual_status='requires_actual_review'))


def heldout(output):
    import render_smooth_temporal_mesh_video as renderer
    from study_early_texture_prior import install
    install();renderer.torch.set_num_threads(2)
    request=deepcopy(renderer.verify_request(HELD));frame='001193'
    request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
    result=read(output/frame/'result.json')
    if sha(output/frame/'mesh.ply')!=result['hashes']['mesh.ply']:raise ValueError('Changed candidate mesh')
    request['inventory'][0].update(mesh=str(output/frame/'mesh.ply'),mesh_sha256=sha(output/frame/'mesh.ply'))
    request.update(partial_diagnostic_only=True,full_video_candidate=False,
        matched_geometry_only_parent_sha256=sha(HELD/'request.json'),geometry_result_sha256=sha(output/frame/'result.json'))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    dest=output/'heldout';dest.mkdir(exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
    if (dest/'request.json').exists() and read(dest/'request.json')!=request:raise ValueError('Changed heldout request')
    atomic_json(dest/'request.json',request);renderer.render(dest,[frame])


def score(output):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from score_colmap_patchmatch_tsdf_face import load_manual_face_mask,masked_display_metrics
    frame='001193';gtpath=GT/'gt.png'
    if sha(gtpath)!=read(GT/'gt_receipt.json')['gt_sha256']:raise ValueError('Changed GT')
    gt=np.array(Image.open(gtpath));mask,_=load_manual_face_mask(GT/'roi.json',gtpath,gt.shape[:2])
    torch.set_num_threads(2);model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    images=[gt];rows=[];results=[]
    for variant,root in [('baseline',HELD),('repaired',output/'heldout')]:
        pred,result=verified_image(root,frame);images.append(pred);results.append(result)
        with torch.inference_mode():
            metric=masked_display_metrics(torch.from_numpy(pred.transpose(2,0,1).copy()).float().cuda()/255,
                torch.from_numpy(gt.transpose(2,0,1).copy()).float().cuda()/255,torch.from_numpy(mask).cuda(),model)
        if not all(np.isfinite(metric[k]) for k in ['face_psnr','face_ssim','face_lpips']):raise ValueError('Nonfinite face metric')
        rows.append(dict(frame=frame,variant=variant,**metric,prediction_sha256=sha(root/'frames'/frame/'frame.png')))
    for key in ['camera','source_cameras','fixed_exposure']:
        if results[0][key]!=results[1][key]:raise ValueError('Unmatched heldout pair')
    panel(output/'heldout/comparison.png',images,['heldout GT','baseline','repaired'],(170,450,970,1250))
    atomic_json(output/'heldout/metrics.json',dict(rows=rows,gt_sha256=sha(gtpath),roi_sha256=sha(GT/'roi.json'),
        protocol='previously frozen GT-only face polygon; no prediction-defined mask',parameters_tuned_on_heldout=False,
        full_frame_metrics=False,script_sha256=sha(__file__),visual_status='requires_actual_review'))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['panels','heldout','score'])
    p.add_argument('--frame',choices=FRAMES);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.action=='panels':review(a.output,a.frame)
    elif a.action=='heldout':heldout(a.output)
    else:score(a.output)
