"""Matched early-texture forearm controls with fixed train-only skin metrics."""
from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image,ImageDraw
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from joint_temporal_texture import read,sha,atomic_json
from score_colmap_patchmatch_tsdf_face import masked_display_metrics
import study_forearm_plane_transfer_v3 as prior

ROOTS={'previous':Path('/mnt/data/dec5_forearm_early_texture_prior'),
       'pins_only':Path('/mnt/data/dec5_forearm_measured_boundary'),
       'pins_and_domain':Path('/mnt/data/dec5_forearm_measured_boundary_domain')}

def run(output,frames,annotation_only=False,variants=None):
    prior.configure();v1=prior.v2.v1;torch.set_num_threads(2)
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();records=[];inputs={}
    for frame in frames:
        name=v1.NAMES[1];gtpath=prior.OUT/frame/'rgb'/(name+'.png');gt=np.rot90(np.array(Image.open(gtpath))).copy()
        mask=np.rot90(v1.masks(frame)[name]).copy()
        labels=variants or (['previous','annotation_only'] if annotation_only else
                ['previous']+(['pins_only'] if frame=='001037' else [])+['pins_and_domain'])
        for view in ['moving',name]:
            images=[];results=[];metrics=[]
            for label in labels:
                root=ROOTS[label]/'rgb'/frame/view/'guarded';folder=root/'frames'/frame
                receipt=read(folder/'complete.json');request=read(root/'request.json')
                if receipt['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed render request')
                for p,h in receipt['hashes'].items():
                    if sha(folder/p)!=h:raise ValueError('Changed render artifact')
                result=read(folder/'result.json');results.append(result)
                if result['target_rgb_read'] or result['rgb_averaging'] or request['recipe']['texture_source_prior']!='target_angle_before_incidence_clip':raise ValueError('Unmatched texture policy')
                pred=np.array(Image.open(folder/'frame.png'));images.append(pred);inputs[str(folder/'frame.png')]=sha(folder/'frame.png')
                depth=np.rot90(np.load(folder/'target_depth.npz')['depth'])
                if view==name:
                    with torch.inference_mode():
                        m=masked_display_metrics(torch.tensor(pred.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),
                            torch.tensor(gt.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),torch.tensor(mask,device='cuda'),model)
                    m={k.removeprefix('face_'):v for k,v in m.items()}
                    if not all(np.isfinite(m[k]) for k in ['psnr','ssim','lpips']):raise ValueError('Nonfinite metric')
                    records.append(dict(frame=frame,variant=label,**m,skin_rgb_holes=int((mask&(pred.max(2)==0)).sum()),
                        skin_depth_holes=int((mask&(depth==0)).sum()),gt_sha256=sha(gtpath),prediction_sha256=sha(folder/'frame.png'),
                        train_reprojection_not_heldout=True))
            for r in results[1:]:
                for key in ['camera','source_cameras','fixed_exposure']:
                    if r[key]!=results[0][key]:raise ValueError('Unmatched geometry comparison: '+key)
            include_gt=view==name;columns=len(images)+int(include_gt);panel=Image.new('RGB',(430*columns,550));draw=ImageDraw.Draw(panel)
            ims=([gt] if include_gt else [])+images;names=(['real train GT'] if include_gt else [])+labels
            for i,(im,label) in enumerate(zip(ims,names)):
                panel.paste(Image.fromarray(im).crop((0,1400,430,1920)),(430*i,30));draw.text((430*i+3,5),label,fill='white')
            dest=output/frame;dest.mkdir(parents=True,exist_ok=True);panel.save(dest/(view+'_comparison.png'))
    atomic_json(output/'metrics.json',dict(frames=frames,rows=records,prediction_hashes=inputs,script_sha256=sha(__file__),
        region_protocol_sha256=sha(prior.OUT/'protocol.json'),scope='fixed manual train forearm skin; not heldout face or full-frame',
        main_face_campaign_csv_unchanged=True,visual_status='requires_separate_actual_review'))
    print(records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001029','001033','001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_measured_boundary_review'))
    p.add_argument('--annotation-only',action='store_true');a=p.parse_args()
    ROOTS['annotation_only']=Path('/mnt/data/dec5_forearm_annotation_domain_only')
    run(a.output,a.frames,a.annotation_only)
