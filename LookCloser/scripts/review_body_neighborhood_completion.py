"""Matched body-prior RGB, fixed train-forearm ROI and native hand/torso panels."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT as COLOR
from study_body_neighborhood_completion import ROOT
from review_jaw_repair_transfer import verified_image,panel
from score_colmap_patchmatch_tsdf_face import masked_display_metrics
import study_forearm_plane_transfer_v3 as prior


def run(frame):
    root=ROOT/frame;dest=root/'review';dest.mkdir(exist_ok=False)
    prior.configure();torch.set_num_threads(2)
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    name='H004_A005_1210M6';mask=np.rot90(prior.v2.v1.masks(frame)[name]).copy()
    rows,_,_=cameras(frame);i=next(i for i,r in enumerate(rows) if r['physical_camera']==name)
    gains=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(gains-gains.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    gt=np.rot90(np.rint(display(exr(rows[i]['file_path'])*gain[i],exposure)*255).clip(0,255).astype(np.uint8)).copy()
    Image.fromarray(gt).save(dest/'train_gt.png');records=[]
    for view in ['moving',name]:
        images=[];metadata=[];depths=[];inputs={};metrics=[]
        for variant in ['baseline','repaired']:
            location=root/'rgb'/view/variant;im,result=verified_image(location,frame)
            d=np.rot90(np.load(location/'frames'/frame/'target_depth.npz')['depth'])
            images.append(im);metadata.append(result);depths.append(d)
            inputs[str(location/'frames'/frame/'frame.png')]=sha(location/'frames'/frame/'frame.png')
            if view==name:
                with torch.inference_mode():
                    m=masked_display_metrics(torch.from_numpy(im.transpose(2,0,1).copy()).float().cuda()/255,
                        torch.from_numpy(gt.transpose(2,0,1).copy()).float().cuda()/255,torch.from_numpy(mask).cuda(),model)
                m={k.removeprefix('face_'):v for k,v in m.items()}
                assert all(np.isfinite(m[k]) for k in ['psnr','ssim','lpips'])
                metrics.append(dict(variant=variant,**m,skin_depth_holes=int((mask&(d==0)).sum()),
                    skin_black_rgb=int((mask&(im.max(2)==0)).sum())))
        for key in ['camera','source_cameras','fixed_exposure']:assert metadata[0][key]==metadata[1][key]
        changed=np.any(images[0]!=images[1],axis=2)
        record=dict(view=view,input_hashes=inputs,metrics=metrics,changed_rgb_pixels=int(changed.sum()),
            changed_upper_1400=int(changed[:1400].sum()),
            black_with_geometry=[int(((im.max(2)==0)&(d>0)).sum()) for im,d in zip(images,depths)],
            full_image_counts_are_coverage_diagnostics_not_quality_metrics=True)
        labels=['protected baseline','body completion'];ims=images
        if view==name:ims=[gt,*images];labels=['real train GT',*labels]
        panel(dest/(view+'_hand.png'),ims,labels,(0,1380,470,1920))
        panel(dest/(view+'_torso.png'),ims,labels,(0,1180,1080,1920))
        records.append(record)
    atomic_json(dest/'result.json',dict(frame=frame,records=records,script_sha256=sha(__file__),
        source_gt_sha256=sha(rows[i]['file_path']),gt_sha256=sha(dest/'train_gt.png'),
        profile_sha256=sha(COLOR/'parameters.npz'),exposure_sha256=sha(COLOR/'exposure.json'),
        region_protocol_sha256=sha(prior.OUT/'protocol.json'),
        scope='fixed train forearm skin, not heldout face; no full-frame quality metrics',
        campaign_csv_changed=False,visual_status='pending'))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=['001029','001033','001037'],required=True)
    run(p.parse_args().frame)
