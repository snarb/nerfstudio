"""Supplement frozen ROI metrics with face interiors excluding known mask holes.

Primary full-frame selection and existing ROI metrics remain unchanged. These
additional metrics use saved 8-bit PNGs for every checkpoint being compared.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from blur_runtime import metrics,write

BOXES = {
    'cam_012_000470.png':[719,330,752,397],
    'cam_095_000470.png':[675,345,850,545],
    'cam_150_000470.png':[90,530,335,820],
    'cam_011_000470.png':[770,290,818,365],
    'cam_097_000470.png':[645,320,800,500],
    'cam_151_000470.png':[980,815,1230,1080],
}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    torch.set_num_threads(2)
    model=SimpleNamespace(ssim=structural_similarity_index_measure,lpips=LearnedPerceptualImagePatchSimilarity(normalize=True))
    rows=[];panels=[]
    for run in args.runs:
        request=json.loads((run/'request.json').read_text());result=json.loads((run/'history.json').read_text())[-1]
        for view in result['per_view']:
            name=view['image'];box=BOXES[name]
            gt=Image.open(Path(request['data'])/'images'/name).convert('RGB').crop(box)
            pred=Image.open(Path(result['render_dir'])/f'{view["split"]}_{view["index"]:03d}.png').convert('RGB').crop(box)
            a=torch.tensor(np.array(pred).astype('float32')/255);b=torch.tensor(np.array(gt).astype('float32')/255)
            values=metrics(model,a,b);rows.append(dict(run=str(run),step=result['step'],image=name,split=view['split'],**values))
            panel=Image.new('RGB',(gt.width*2,gt.height+24),(35,35,35));panel.paste(gt,(0,24));panel.paste(pred,(gt.width,24));ImageDraw.Draw(panel).text((3,3),f'{run.name} {name[:7]}',fill='white');panels.append(panel)
    write(args.output/'metrics.json',dict(protocol='Additional fixed face interiors, saved PNG RGB; does not select checkpoints',boxes=BOXES,results=rows))
    sheet=Image.new('RGB',(max(i.width for i in panels),sum(i.height for i in panels)),(35,35,35));y=0
    for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
    sheet.save(args.output/'contact.jpg',quality=95)
    for run in args.runs:
        for split in ['train','eval']:
            subset=[r for r in rows if r['run']==str(run) and r['split']==split]
            print(json.dumps(dict(run=str(run),split=split,**{k:float(np.mean([r[k] for r in subset])) for k in ['psnr','ssim','lpips']})))

if __name__=='__main__':main()
