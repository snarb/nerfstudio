#!/usr/bin/env python3
"""Heldout geometry-defined background quality, never used to fit weights.

PSNR and local-window SSIM use exterior box support. LPIPS uses only entire
256px tiles within that support; no masked image or GT compositing is used.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from marshawn_ray_background_provider import FrozenMarshawnRayBackground,sha
from prepare_marshawn_ray_background import mask_for_row

def run(a):
 from skimage.metrics import structural_similarity
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 torch.set_num_threads(2);cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False);meta=json.loads((a.data/'transforms.json').read_text());frames=[r for r in meta['frames'] if r['file_path'] in meta['val_filenames']];assert len(frames)==3
 support=json.loads(a.support.read_text());assert support['data_sha256']==sha(a.data/'transforms.json');box=np.array(support['subject_box']);band=support['protective_band_hd_pixels']-1
 provider=FrozenMarshawnRayBackground(a.checkpoint,device=a.device,data=a.data);lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).to(a.device);rows=[];tiles=[]
 for row in frames:
  h,w=row['h'],row['w'];pose=torch.tensor(row['transform_matrix'],device=a.device,dtype=torch.float32);pred=np.empty((h*w,3),np.float32)
  for start in range(0,h*w,65536):
   flat=torch.arange(start,min(start+65536,h*w),device=a.device);x=flat%w;y=flat//w;d=torch.stack(((x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-torch.ones_like(x)),1).float()@pose[:3,:3].T;origin=pose[:3,3].expand(len(flat),3)
   with torch.inference_mode():rgb=provider.predict(origin,d)
   pred[start:start+len(flat)]=rgb.cpu().numpy()
  pred=pred.reshape(h,w,3);gt=np.array(Image.open(a.data/row['file_path']).convert('RGB')).astype(np.float32)/255;valid=np.array(Image.open(a.data/row['mask_path']))>0;mask,hull=mask_for_row(row,box,valid,band);name=row['physical_camera'];Image.fromarray(np.rint(pred.clip(0,1)*255).astype(np.uint8)).save(a.output/(name+'_background_prediction.png'));Image.fromarray(mask.astype(np.uint8)*255).save(a.output/(name+'_metric_support.png'))
  mse=np.mean((pred[mask]-gt[mask])**2);_,ssim=structural_similarity(gt,pred,channel_axis=-1,data_range=1.,win_size=11,full=True);local_mask=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool);ssim_value=float(ssim.mean(-1)[local_mask].mean());perceptual=[];patches=[]
  for y in range(0,h-255,128):
   for x in range(0,w-255,128):
    if not mask[y:y+256,x:x+256].all():continue
    p=torch.from_numpy(pred[y:y+256,x:x+256].copy()).permute(2,0,1)[None].to(a.device);g=torch.from_numpy(gt[y:y+256,x:x+256].copy()).permute(2,0,1)[None].to(a.device)
    with torch.inference_mode():score=float(lpips(p,g))
    lpips.reset();perceptual.append(score);patches.append([x,y,256,256])
  rows.append(dict(camera=name,PSNR=float(-10*np.log10(mse)),SSIM=ssim_value,LPIPS=float(np.mean(perceptual)) if perceptual else None,background_pixels=int(mask.sum()),lpips_entire_background_tiles=len(perceptual),tile_rectangles=patches,source_sha256=sha(a.data/row['file_path']),prediction_sha256=sha(a.output/(name+'_background_prediction.png'))))
  show=Image.fromarray(np.rint(pred.clip(0,1)*255).astype(np.uint8));reference=Image.open(a.data/row['file_path']).convert('RGB');difference=Image.fromarray(np.rint(abs(pred-gt).clip(0,1)*255).astype(np.uint8));overlay=np.rint(pred.clip(0,1)*255).astype(np.uint8);overlay[~mask]=(.45*overlay[~mask]+.55*np.array([255,0,150])).astype(np.uint8)
  panel=Image.new('RGB',(4*480,280));d=ImageDraw.Draw(panel)
  for i,(im,label) in enumerate([(reference,'GT reference only'),(show,'Predicted frozen B'),(difference,'Absolute difference'),(Image.fromarray(overlay),'Metric exclusion magenta')]):im.thumbnail((480,253));panel.paste(im,(i*480,24));d.text((i*480+4,3),f'{name} {label}',fill='white')
  panel.save(a.output/(name+'_review.jpg'),quality=90);tiles.append(panel)
 provider.verify();sheet=Image.new('RGB',(1920,280*len(tiles)))
 for i,t in enumerate(tiles):sheet.paste(t,(0,280*i))
 sheet.save(a.output/'all_heldout_review.jpg',quality=90)
 report=dict(checkpoint_sha256=sha(a.checkpoint),support_sha256=sha(a.support),transforms_sha256=sha(a.data/'transforms.json'),uses_eval_rgb_for_metrics_only=True,uses_eval_rgb_for_fitting_or_selection=False,gt_compositing=False,rows=rows,mean={k:float(np.mean([r[k] for r in rows])) for k in ['PSNR','SSIM','LPIPS']},component_gate_passed=False,reason='Metrics and visuals supplied for review; no automatic quality acceptance',metric_protocol='PSNR exterior valid support; SSIM mean local11window map on eroded exterior support; LPIPS mean of untouched256x256 fully exterior tiles at128stride',limitations=['Geometry-only support excludes projected subject box; does not evaluate hidden background inside that box','LPIPS tile-average is not interchangeable with full-frame LPIPS','Frozen ray background has no geometric depth or occupancy'])
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['mean']))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','support','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--device',default='cuda');run(p.parse_args())
