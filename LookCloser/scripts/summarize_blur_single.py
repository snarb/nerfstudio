"""Compare saved, identically quantized synthetic probe renders on known teacher pixels."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from blur_runtime import write


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root=Path('/home/brans/lookcloser_artifacts/blur_ablation_fresh')
    meta=json.loads((root/'synthetic_single/transforms.json').read_text())
    row=next(r for r in meta['frames'] if 'train' in r['file_path'])
    gt=torch.tensor(np.array(Image.open(root/'synthetic_single'/row['file_path']))[::2,::2].copy(),device='cuda').float()/255
    mask=torch.tensor(np.array(Image.open(root/'synthetic_single'/row['mask_path']))[::2,::2].copy()>0,device='cuda')
    points=mask.nonzero();y0,x0=points.min(0).values.tolist();y1,x1=(points.max(0).values+1).tolist()
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    target=torch.where(mask[...,None],gt,0)[y0:y1,x0:x1].permute(2,0,1)[None]
    records=[]
    for path in sorted(root.glob('s[0-9]*/complete.json')):
        for step in [1000,2000,3000]:
            pred=torch.tensor(np.array(Image.open(path.parent/f'eval_{step:06d}/train_000.png')),device='cuda').float()/255
            image=torch.where(mask[...,None],pred,0)[y0:y1,x0:x1].permute(2,0,1)[None]
            records.append(dict(run=path.parent.name,step=step,
                psnr=float(-10*torch.log10((pred[mask]-gt[mask]).square().mean())),
                ssim=float(structural_similarity_index_measure(image,target,data_range=1.)),lpips=float(lpips(image,target))))
    write(Path(__file__).resolve().parents[1]/'experiments/assets/blur_ablation_fresh/single_metrics.json',
        dict(protocol='8-bit PNG, stride2; teacher-valid train PSNR; tight zero-masked SSIM/LPIPS. Diagnostic only.',results=records))
    for r in records:
        if r['step']==2000:print(json.dumps(r))


if __name__=='__main__':main()
