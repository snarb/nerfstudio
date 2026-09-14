"""Held-out fidelity check of bounded gradient color, using the actual movie renderer.

Geometry/profiles/foreground masks/view-prior stay frozen. GT is read only by
separate evaluation commands. Never tune color parameters from this evaluation.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import CALIBRATION, ROOT, SOURCE, read, sha, atomic_json, exr, display
from render_smooth_temporal_mesh_video import verify_request
from render_patchmatch_camera_path import normalize_frame

PARENT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_gradient_heldout_fidelity')
FRAMES=['000899','000973','001193']
CAMERA='F004_B005_1210O9'


def initialize(output):
    parent=verify_request(PARENT); request=deepcopy(parent); cal=read(CALIBRATION)
    held=next(r for r in cal['frames'] if r['physical_camera']==CAMERA)
    request['ordered_frame_ids']=FRAMES
    request['inventory']=[r for r in request['inventory'] if r['frame_id'] in FRAMES]
    request['source_rows']=[r for r in request['source_rows'] if Path(r['source_dataset']).name in FRAMES]
    for r in request['inventory']:
        r['camera']=normalize_frame(held,cal,read(r['metadata']))
    request.update(study='bounded gradient color fidelity at held-out camera',
                   heldout_camera=CAMERA, uses_heldout_rgb=False,
                   parent_sha256=sha(PARENT/'request.json'),
                   image_fidelity_parameters_tuned_on_heldout=False)
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True); (output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:
        raise ValueError('Immutable held-out request changed')
    atomic_json(output/'request.json',request)


def ground_truth(output, frame):
    # This path does not produce any renderer input; no held-out profile is fit.
    verify_request(output)
    row=next(r for r in read(SOURCE/frame/'transforms.json')['frames'] if r['physical_camera']==CAMERA)
    path=SOURCE/frame/row['file_path']; gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    rgb=np.rint(255*display(exr(path),gain)).clip(0,255).astype(np.uint8)
    dest=output/'evaluation'/frame; dest.mkdir(parents=True,exist_ok=False)
    Image.fromarray(np.rot90(rgb)).save(dest/'gt.png')
    atomic_json(dest/'gt_receipt.json',dict(source=str(path),source_sha256=sha(path),gt_sha256=sha(dest/'gt.png'),
        exposure_gain=gain,camera_gain_applied=False,display='fixed exposure -> Reinhard -> sRGB',
        evaluation_only=True,frame=frame,physical_camera=CAMERA))


def roi(output,frame,polygon):
    from score_colmap_patchmatch_tsdf_face import validate_polygon
    dest=output/'evaluation'/frame
    if (dest/'roi.json').exists():raise ValueError('Preserve frozen GT ROI')
    pts=validate_polygon(np.array(polygon).reshape(-1,2).tolist(),1080,1920)
    record=dict(selection_method='manual_polygon_on_heldout_gt_only',prediction_used_for_selection=False,
        ground_truth_sha256=sha(dest/'gt.png'),include_polygons=[pts],exclude_polygons=[],
        notes='Visible face polygon selected on fixed-exposure GT before reviewing the compared predictions; never candidate support.')
    atomic_json(dest/'roi.json',record)
    im=Image.open(dest/'gt.png').copy();ImageDraw.Draw(im).line(pts+[pts[0]],fill='magenta',width=3);im.save(dest/'roi_overlay.png')


def render(output):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2)
    install(renderer);install_source_masks(renderer)
    renderer.render(output)


def correct(output,frame):
    from study_temporal_gradient_seams import run
    from bound_temporal_gradient_offset import run as bound
    from audit_temporal_gradient_control import audit
    from audit_bounded_temporal_gradient_control import audit as audit_bound
    dest=output/'correction'/frame
    if not (dest/'raw/result.json').exists():
        run(output,dest/'raw',frame,[0,0,1080,1920],solver_device='cuda')
    audit(dest/'raw')
    if not (dest/'bounded/result.json').exists():bound(dest/'raw',dest/'bounded')
    audit_bound(dest/'bounded')


def score(output,frame):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from score_colmap_patchmatch_tsdf_face import load_manual_face_mask, masked_display_metrics
    torch.set_num_threads(2); dest=output/'evaluation'/frame
    gt=np.array(Image.open(dest/'gt.png')); mask,_=load_manual_face_mask(dest/'roi.json',dest/'gt.png',gt.shape[:2])
    if sha(dest/'gt.png')!=read(dest/'gt_receipt.json')['gt_sha256']:raise ValueError('Changed evaluation GT')
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).eval()
    variants={'baseline':output/'frames'/frame/'frame.png',
              'bounded':output/'correction'/frame/'bounded/corrected.png'}
    results=[];images=[Image.fromarray(gt)]
    for name,path in variants.items():
        pred=np.array(Image.open(path));images.append(Image.fromarray(pred))
        with torch.inference_mode():
            metrics=masked_display_metrics(torch.from_numpy(pred.transpose(2,0,1).copy()).float()/255,
                torch.from_numpy(gt.transpose(2,0,1).copy()).float()/255,torch.from_numpy(mask),model)
        results.append(dict(variant=name,prediction_sha256=sha(path),**metrics))
    atomic_json(dest/'metrics.json',dict(frame=frame,rows=results,roi_sha256=sha(dest/'roi.json'),
        gt_sha256=sha(dest/'gt.png'),protocol='exact face pixels PSNR; tight zero-outside ROI SSIM/AlexNet LPIPS',
        heldout_rgb_used_for_prediction=False,full_frame_metrics=False))
    box=results[0]['face_bbox_xyxy'];x0,y0,x1,y1=box
    # Context includes hair/lips, selected by GT face extent, not missing surfaces.
    box=(max(0,x0-100),max(0,y0-240),min(1080,x1+100),min(1920,y1+100))
    w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(w*3,h+24));draw=ImageDraw.Draw(panel)
    for i,(name,im) in enumerate(zip(['GT','hard baseline','bounded gradient'],images)):
        panel.paste(im.crop(box),(i*w,24));draw.text((i*w+3,3),name,fill='white')
    panel.save(dest/'native_face_hair_comparison.png');print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','gt','roi','render','correct','score'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frame',choices=FRAMES)
    p.add_argument('--polygon',nargs='+',type=int);a=p.parse_args()
    if a.action=='init':initialize(a.output)
    elif a.action=='render':render(a.output)
    elif a.action=='roi':roi(a.output,a.frame,a.polygon)
    elif a.action=='gt':ground_truth(a.output,a.frame)
    elif a.action=='correct':correct(a.output,a.frame)
    else:score(a.output,a.frame)
