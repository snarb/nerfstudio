"""Isolated MatAnyone2 pilot on existing previous-flight neural renders."""
import argparse,json,time
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image,ImageDraw
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int,default=69);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
for d in ['alpha','mat_black','safe_black','review']: (a.out/d).mkdir(exist_ok=True)
torch.set_num_threads(8); model=get_matanyone2_model(a.checkpoint,torch.device('cuda')); proc=InferenceCore(model,cfg=model.cfg,device='cuda')
mask=np.load(a.root/f'{a.first:03}'/'strict.npz')['accumulation'][...,0].clip(0,1)*255
mask=torch.from_numpy(mask).float().cuda();rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
 for ti,i in enumerate([a.first]*10+list(range(a.first,a.last+1))):
  raw=np.array(Image.open(a.root/f'{i:03}'/'raw.png').convert('RGB')); inp=torch.from_numpy(raw.copy()).permute(2,0,1).cuda().float()/255
  if ti==0:proc.step(inp,mask,objects=[1])
  prob=proc.step(inp,first_frame_pred=ti<=10);alpha=proc.output_prob_to_mask(prob).float().cpu().numpy().clip(0,1)
  if ti<10:continue
  old=np.load(a.root/f'{i:03}'/'strict.npz')['accumulation'][...,0].clip(0,1); strict=np.array(Image.open(a.root/f'{i:03}'/'strict.png').convert('RGB'))
  # Conservative diagnostic: preserve all interior more than four pixels from old support edge.
  dist=cv2.distanceTransform((old>.5).astype('uint8'),cv2.DIST_L2,5); protect=np.clip((dist-2)/2,0,1)
  ratio=np.minimum(1,alpha/np.maximum(old,1e-4));safe=1-(1-ratio)*(1-protect)
  mat=np.rint(raw*alpha[...,None]).astype('uint8');out=np.rint(strict*safe[...,None]).astype('uint8')
  Image.fromarray(np.rint(alpha*255).astype('uint8')).save(a.out/'alpha'/f'{i:03}.png');Image.fromarray(mat).save(a.out/'mat_black'/f'{i:03}.png');Image.fromarray(out).save(a.out/'safe_black'/f'{i:03}.png')
  if i%5==0 or i==a.last:
   panel=Image.new('RGB',(1920,1104));d=ImageDraw.Draw(panel)
   for k,(im,label) in enumerate([(strict,'Strict baseline'),(mat,'Raw x MatAnyone2'),(out,'Boundary-only conservative')]):
    panel.paste(Image.fromarray(im).resize((640,360)),(k*640,24));d.text((k*640+5,4),f'{i:03} {label}',fill='white')
    panel.paste(Image.fromarray(im).crop((300,120,1000,800)).resize((640,680)),(k*640,400))
   panel.save(a.out/'review'/f'{i:03}.jpg',quality=95)
  rows.append({'frame':i,'mat_removed_strict_gt09':int(((old>.9)&(alpha<.1)).sum()),'safe_changed_rgb_pixels':int(np.any(out!=strict,axis=2).sum()),'max_core_rgb_change':int(np.abs(out.astype('int16')-strict.astype('int16'))[dist>=4].max()) if (dist>=4).any() else 0}); print(json.dumps(rows[-1]),flush=True)
(a.out/'receipt.json').write_text(json.dumps({'source':str(a.root),'frames':rows,'note':'Diagnostic; no ground-truth alpha. Safe preserves 4px interior, not yet approved hair/prop boundary.'},indent=2))
