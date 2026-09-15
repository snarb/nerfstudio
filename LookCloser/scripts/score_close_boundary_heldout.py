"""Frozen GT-only face metrics for a geometry-changing, matched-texture control."""
import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from joint_temporal_texture import read,sha,atomic_json
from evaluate_head_source_quality import GT
from render_close_boundary_completion import ROOT
from review_jaw_repair_transfer import verified_image,panel
from score_colmap_patchmatch_tsdf_face import load_manual_face_mask,masked_display_metrics


def run():
    root=ROOT/'heldout';frame='001193';gtpath=GT/'gt.png'
    if sha(gtpath)!=read(GT/'gt_receipt.json')['gt_sha256']:raise ValueError('Changed held-out GT')
    gt=np.array(Image.open(gtpath));mask,_=load_manual_face_mask(GT/'roi.json',gtpath,gt.shape[:2])
    torch.set_num_threads(2);model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    rows=[];images=[gt];records=[]
    for mode in ['baseline','close_boundary']:
        location=root/mode;pred,result=verified_image(location,frame);records.append(result);images.append(pred)
        if len(records)>1:
            for key in ['camera','source_cameras','fixed_exposure']:
                if result[key]!=records[0][key]:raise ValueError('Unmatched geometry comparison')
            if result['mesh_sha256']==records[0]['mesh_sha256']:raise ValueError('Geometry did not change')
        with torch.inference_mode():
            metrics=masked_display_metrics(torch.from_numpy(pred.transpose(2,0,1).copy()).float().cuda()/255,
                torch.from_numpy(gt.transpose(2,0,1).copy()).float().cuda()/255,torch.from_numpy(mask).cuda(),model)
        if not all(np.isfinite(metrics[k]) for k in ['face_psnr','face_ssim','face_lpips']):raise ValueError('Nonfinite face metric')
        rows.append(dict(mode=mode,**metrics,prediction_sha256=sha(location/'frames'/frame/'frame.png'),mesh_sha256=result['mesh_sha256']))
    panel(root/'comparison.png',images,['held-out GT','production mesh','no-minimum-gap completion'],(170,450,970,1250))
    atomic_json(root/'metrics.json',dict(rows=rows,frame=frame,gt_sha256=sha(gtpath),roi_sha256=sha(GT/'roi.json'),
        protocol='Existing frozen GT-only face ROI; same pose and texture recipe, changed mesh.',
        full_frame_metrics=False,loss_reported=False,geometry_changed=True,visual_status='pending',script_sha256=sha(__file__)))
    print(rows,flush=True)


if __name__=='__main__':run()
