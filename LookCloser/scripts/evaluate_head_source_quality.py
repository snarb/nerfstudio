"""Frozen held-out face check of source-quality controls on identical geometry."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import render_smooth_temporal_mesh_video as renderer
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_head_source_quality_heldout')
GT=Path('/mnt/data/dec5_gradient_heldout_fidelity/evaluation/001193')
BASE=Path('/mnt/data/dec5_central_train_pose_transfer/001193/native_G004_C005_121037')
HELD=Path('/mnt/data/dec5_early_texture_prior_heldout')
MODES=('baseline','incidence2','angular_only','pixel_angular')
FRAME='001193'


def render(mode):
    q=deepcopy(renderer.verify_request(BASE));held=renderer.verify_request(HELD)
    e=next(r for r in held['inventory'] if r['frame_id']==FRAME)
    assert e['metadata_sha256']==q['inventory'][0]['metadata_sha256']
    q['inventory'][0]['camera']=deepcopy(e['camera'])
    scripts=[Path(__file__).name]
    if mode=='baseline':
        from study_native_texture_footprint import install
    elif mode=='pixel_angular':
        from study_pixel_angular_head_texture import install
        scripts+=['view_consistent_source_quality.py','study_view_consistent_head_texture.py','study_pixel_angular_head_texture.py']
    else:
        from study_view_consistent_head_texture import install as installer
        install=lambda:installer(mode)
        scripts+=['view_consistent_source_quality.py','study_view_consistent_head_texture.py']
    implementation=install();renderer.torch.set_num_threads(2)
    q.update(heldout_source_quality_mode=mode,geometry_changed=False,heldout_camera_request_sha256=sha(HELD/'request.json'),
        implementation_sha256=implementation,partial_diagnostic_only=True,full_video_candidate=False,
        pixel_angle_prior=mode!='baseline',graph_labels_not_used_for_rgb=mode=='pixel_angular',
        graph_color_weight=0. if mode in ('angular_only','pixel_angular') else .5)
    for n in scripts:q['script_hashes'][n]=sha(Path(__file__).with_name(n))
    out=ROOT/mode;out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();atomic_json(out/'request.json',q)
    renderer.render(out,[FRAME])


def score():
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from score_colmap_patchmatch_tsdf_face import load_manual_face_mask,masked_display_metrics
    gtpath=GT/'gt.png';assert sha(gtpath)==read(GT/'gt_receipt.json')['gt_sha256']
    gt=np.array(Image.open(gtpath));mask,_=load_manual_face_mask(GT/'roi.json',gtpath,gt.shape[:2])
    torch.set_num_threads(2);model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    baseline=None;depth=None;rows=[];images=[gt]
    for mode in MODES:
        out=ROOT/mode;pred,result=verified_image(out,FRAME);images.append(pred)
        d=np.load(out/'frames'/FRAME/'target_depth.npz')['depth']
        if baseline is None:baseline=result;depth=d
        for k in ['camera','mesh_sha256','source_cameras','fixed_exposure']:assert result[k]==baseline[k]
        np.testing.assert_array_equal(d,depth)
        with torch.inference_mode():
            metrics=masked_display_metrics(torch.from_numpy(pred.transpose(2,0,1).copy()).float().cuda()/255,
                torch.from_numpy(gt.transpose(2,0,1).copy()).float().cuda()/255,torch.from_numpy(mask).cuda(),model)
        assert all(np.isfinite(metrics[k]) for k in ['face_psnr','face_ssim','face_lpips'])
        rows.append(dict(mode=mode,**metrics,prediction_sha256=sha(out/'frames'/FRAME/'frame.png')))
        panel(ROOT/(mode+'_heldout.png'),[gt,images[1],pred],['held-out GT','baseline',mode],(170,450,970,1250))
    atomic_json(ROOT/'metrics.json',dict(rows=rows,frame=FRAME,gt_sha256=sha(gtpath),roi_sha256=sha(GT/'roi.json'),
        protocol='Existing frozen GT-only face polygon; identical geometry, native footprint and pose; display-domain metrics only.',
        full_frame_metrics=False,loss_reported=False,depth_byte_equal=True,visual_status='pending'))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['render','score']);p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.action=='render':
        if a.mode is None:p.error('--mode required')
        render(a.mode)
    else:score()
