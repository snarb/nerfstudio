"""Fixed held-out face gate for early source admission; reuses frozen GT ROIs."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
import render_smooth_temporal_mesh_video as renderer
from study_early_texture_prior import install

BASE=Path('/mnt/data/dec5_gradient_heldout_fidelity')
OUT=Path('/mnt/data/dec5_early_texture_prior_heldout')
FRAMES=['000899','000973','001193']


def initialize(output):
    request=deepcopy(renderer.verify_request(BASE));source_hash=install()
    request['recipe']['texture_source_prior']='target_angle_before_incidence_clip'
    request.update(study='early source admission heldout check',admission_transform_sha256=source_hash,
        matched_parent_sha256=sha(BASE/'request.json'),image_fidelity_parameters_tuned_on_heldout=False)
    for name in ['study_early_texture_prior.py',Path(__file__).name]:request['script_hashes'][name]=sha(Path(__file__).with_name(name))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Changed heldout request')
    atomic_json(output/'request.json',request)


def render(output,frame):
    request=renderer.verify_request(output);source_hash=install()
    if source_hash!=request['admission_transform_sha256']:raise ValueError('Changed admission implementation')
    renderer.torch.set_num_threads(2);renderer.render(output,[frame])


def score(output):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from score_colmap_patchmatch_tsdf_face import load_manual_face_mask,masked_display_metrics
    renderer.verify_request(output);torch.set_num_threads(2)
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();records=[]
    for frame in FRAMES:
        evaluation=BASE/'evaluation'/frame;gtpath=evaluation/'gt.png'
        if sha(gtpath)!=read(evaluation/'gt_receipt.json')['gt_sha256']:raise ValueError('Changed GT')
        gt=np.array(Image.open(gtpath));mask,_=load_manual_face_mask(evaluation/'roi.json',gtpath,gt.shape[:2])
        images=[Image.fromarray(gt)];depths=[]
        for label,root in [('late',BASE),('early',output)]:
            folder=root/'frames'/frame;receipt=read(folder/'complete.json')
            if sha(root/'request.json')!=receipt['request_sha256']:raise ValueError('Changed prediction request')
            for p,h in receipt['hashes'].items():
                if sha(folder/p)!=h:raise ValueError('Changed rendered output')
            result=read(folder/'result.json')
            if result['target_rgb_read'] or result['rgb_averaging'] or len(result['source_cameras'])!=62:raise ValueError('Prediction protocol changed')
            pred=np.array(Image.open(folder/'frame.png'));images.append(Image.fromarray(pred));depths.append(np.load(folder/'target_depth.npz')['depth'])
            with torch.inference_mode():
                metric=masked_display_metrics(torch.from_numpy(pred.transpose(2,0,1).copy()).float().cuda()/255,
                    torch.from_numpy(gt.transpose(2,0,1).copy()).float().cuda()/255,torch.from_numpy(mask).cuda(),model)
            if not all(np.isfinite(metric[k]) for k in ['face_psnr','face_ssim','face_lpips']):raise ValueError('Nonfinite face metric')
            records.append(dict(frame=frame,variant=label,**metric,gt_sha256=sha(gtpath),roi_sha256=sha(evaluation/'roi.json'),
                prediction_sha256=sha(folder/'frame.png')))
        if not np.array_equal(depths[0],depths[1]):raise ValueError('Changed geometry/depth in texture-only control')
        x0,y0,x1,y1=records[-1]['face_bbox_xyxy'];box=(max(0,x0-100),max(0,y0-240),min(1080,x1+100),min(1920,y1+100))
        w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(3*w,h+24));draw=ImageDraw.Draw(panel)
        for i,(im,label) in enumerate(zip(images,['GT','late angle prior','early angle prior'])):
            panel.paste(im.crop(box),(i*w,24));draw.text((i*w+3,3),label,fill='white')
        dest=output/'evaluation'/frame;dest.mkdir(parents=True,exist_ok=True);panel.save(dest/'native_face_hair_comparison.png')
    atomic_json(output/'metrics.json',dict(rows=records,script_sha256=sha(__file__),full_frame_metrics=False,
        target_depth_exact=True,heldout_rgb_used_for_prediction=False,roi_protocol='previously frozen GT-only face polygons',
        parameters_tuned_on_heldout=False,visual_status='pending'))
    print([{k:r[k] for k in ['frame','variant','face_psnr','face_ssim','face_lpips']} for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','render','score']);p.add_argument('--frame',choices=FRAMES)
    p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.action=='init':initialize(a.output)
    elif a.action=='render':render(a.output,a.frame)
    else:score(a.output)
