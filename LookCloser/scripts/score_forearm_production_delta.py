"""Fixed real-train skin ROI checks, not held-out or full-frame quality scores."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from score_colmap_patchmatch_tsdf_face import masked_display_metrics
from joint_temporal_texture import read,sha,atomic_json
import study_forearm_plane_transfer_v3 as prior


def run(root):
    prior.configure();v1=prior.v2.v1;torch.set_num_threads(2)
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();records=[]
    for frame in ['001029','001033','001037']:
        name=v1.NAMES[1];gtpath=prior.OUT/frame/'rgb'/(name+'.png')
        gt=np.rot90(np.asarray(Image.open(gtpath).convert('RGB'))).copy();mask=np.rot90(v1.masks(frame)[name]).copy()
        if not mask.any():raise ValueError('No requested forearm skin; no substitute ROI allowed')
        panel=Image.new('RGB',(1620,550));draw=ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(gt).crop((0,1400,540,1920)),(0,30));draw.text((4,5),'real train reference',fill='white')
        for index,variant in enumerate(['baseline','guarded']):
            out=root/'rgb'/frame/name/variant;folder=out/'frames'/frame;receipt=read(folder/'complete.json')
            if receipt['request_sha256']!=sha(out/'request.json'):raise ValueError('Changed RGB request')
            for p,h in receipt['hashes'].items():
                if sha(folder/p)!=h:raise ValueError('Changed RGB output')
            pred=np.asarray(Image.open(folder/'frame.png').convert('RGB'))
            with torch.inference_mode():
                metric=masked_display_metrics(torch.tensor(pred.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),
                    torch.tensor(gt.transpose(2,0,1)/255,dtype=torch.float32,device='cuda'),torch.tensor(mask,device='cuda'),model)
            metric={k.removeprefix('face_'):v for k,v in metric.items()}
            if not all(np.isfinite(metric[k]) for k in ['psnr','ssim','lpips']):raise ValueError('Nonfinite ROI metric')
            records.append(dict(frame=frame,variant=variant,physical_camera=name,region='fixed_manual_forearm_skin',**metric,
                pixels=int(mask.sum()),prediction_sha256=sha(folder/'frame.png'),gt_sha256=sha(gtpath),
                source_region_protocol_sha256=sha(prior.OUT/'protocol.json'),train_reprojection_not_heldout=True))
            panel.paste(Image.fromarray(pred).crop((0,1400,540,1920)),((index+1)*540,30));draw.text(((index+1)*540+4,5),variant,fill='white')
        panel.save(root/frame/'train_reference_native.png')
    atomic_json(root/'metrics.json',dict(script_sha256=sha(__file__),rows=records,
        protocol='Fixed manual real-train skin ROI; display PSNR, bbox masked SSIM and AlexNet LPIPS; no loss or full-frame metrics',
        heldout_metrics=False,main_face_campaign_csv_unchanged=True))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
