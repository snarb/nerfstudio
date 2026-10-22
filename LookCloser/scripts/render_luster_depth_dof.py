"""Depth-only postprocess DoF. Never reads silhouette masks; preserves source images.

Relative monocular disparity is not metric depth. Focus uses a central torso
rectangle, normalized depth span and a broad in-focus band. Existing hybrid
appearance residuals are retained, without carrying over silhouette defocus.
"""
from pathlib import Path
import argparse,json,sys,time,subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter1d
R=Path('/fsx/tmp/luster_gt_flythrough_20261022_v11_dof')
RAW=Path('/fsx/tmp/luster_gt_flythrough_20261021_v4_no_blur')
H=Path('/fsx/tmp/luster_gt_flythrough_20261021_v8_transfer')
B=Path('/fsx/tmp/luster_gt_flythrough_20261021_v9_hair')

def rgb(i):
 a=cv2.imread(str(RAW/f'frames/wide/{i:05d}.png'))
 if PLAN['records'][i]['kind']=='flight':
  delta=cv2.imread(str(H/f'frames/wide/{i:05d}.png')).astype(np.float32)-cv2.imread(str(B/f'frames/wide/{i:05d}.png')).astype(np.float32)
  a=np.rint(a.astype(np.float32)+delta).clip(0,255).astype(np.uint8)
 return a

def infer(indices):
 import torch
 sys.path.insert(0,str(R/'runtime/Depth-Anything-V2'))
 from depth_anything_v2.dpt import DepthAnythingV2
 torch.set_num_threads(4)
 model=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
 model.load_state_dict(torch.load(R/'runtime/depth_anything_v2_vitl.pth',map_location='cpu',weights_only=True));model=model.cuda().eval()
 for i in indices:
  p=R/f'depth/{i:05d}.npy'
  if p.exists():continue
  t=time.time()
  with torch.inference_mode():d=model.infer_image(rgb(i),input_size=756)
  np.save(p,d.astype(np.float16))
  vis=cv2.applyColorMap(np.rint((d-d.min())/(np.ptp(d)+1e-6)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
  cv2.imwrite(str(R/f'depth/{i:05d}.jpg'),vis)
  print(f'depth {i}/313 {time.time()-t:.1f}s',flush=True)

def params(indices):
 vals=[]
 for i in indices:
  d=np.load(R/f'depth/{i:05d}.npy').astype(np.float32);h,w=d.shape
  focus=float(min(np.median(d[int(y*h)-20:int(y*h)+20,int(.48*w):int(.53*w)]) for y in [.25,.32,.50,.60,.75]))
  span=float(np.percentile(d,95)-np.percentile(d,5))
  vals.append([focus,span])
 return np.array(vals)

def render(i,focus,span):
 a=rgb(i);d=np.load(R/f'depth/{i:05d}.npy').astype(np.float32)
 # Broad depth-of-field keeps hands, face and clothing sharp despite body depth.
 q=(d-focus)/max(span,1e-5)
 if ALL_PARAMS is not None:
  sw,sh=375,512
  gray=cv2.cvtColor(cv2.resize(a,(sw,sh)),cv2.COLOR_BGR2GRAY)
  accum=q.copy();norm=np.ones_like(q)
  for j in (i-1,i+1):
   if j<0 or j>=314:continue
   other=rgb(j);g=cv2.cvtColor(cv2.resize(other,(sw,sh)),cv2.COLOR_BGR2GRAY)
   flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray,g,None)
   flow=cv2.resize(flow,(a.shape[1],a.shape[0]))*4
   yy,xx=np.indices(q.shape,dtype=np.float32)
   dj=np.load(R/f'depth/{j:05d}.npy').astype(np.float32)
   qj=(dj-ALL_PARAMS[j,0])/max(ALL_PARAMS[j,1],1e-5)
   warped=cv2.remap(qj,xx+flow[...,0],yy+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
   weight=.3*np.clip(1-np.abs(warped-q)/.10,0,1)
   accum+=weight*warped;norm+=weight
  q=accum/norm
 coc=np.clip(np.maximum(-q-.12,q-.65)/.45,0,1)*24
 # Defocused contributors only: reduces bright/colored foreground halos.
 weight=np.clip(np.maximum(-q-.07,q-.60)/.16,0,1).astype(np.float32)
 af=a.astype(np.float32);out=af.copy();prev=0.
 for sigma in [2.,4.,8.,12.,18.,24.]:
  den=cv2.GaussianBlur(weight,(0,0),sigma)
  num=cv2.GaussianBlur(af*weight[...,None],(0,0),sigma)
  blurred=num/np.maximum(den[...,None],1e-5)
  blurred=np.where((den>.01)[...,None],blurred,af)
  # Interpolate adjacent blur levels using per-pixel circle of confusion.
  amount=np.clip((coc-prev)/(sigma-prev),0,1)[...,None]
  out=out*(1-amount)+blurred*amount;prev=sigma
 cv2.imwrite(str(R/f'frames/{i:05d}.png'),np.rint(out).clip(0,255).astype(np.uint8))
 return dict(index=i,focus=focus,span=span,coc_p50=float(np.median(coc)),coc_p95=float(np.percentile(coc,95)))

def review(indices):
 thumbs=[]
 for i in indices:
  im=Image.open(R/f'frames/{i:05d}.png');im.thumbnail((300,410));t=Image.new('RGB',(300,434));t.paste(im,(0,24));ImageDraw.Draw(t).text((3,3),str(i),fill='white');thumbs.append(t)
 for start in range(0,len(thumbs),8):
  sheet=Image.new('RGB',(1200,868))
  for j,t in enumerate(thumbs[start:start+8]):sheet.paste(t,((j%4)*300,(j//4)*434))
  sheet.save(R/f'review/contact_{indices[start]:03d}.jpg',quality=94)

ALL_PARAMS=None

def main():
 global ALL_PARAMS
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 ids=[0,33,84,112,146,194,241,253,277,301,313] if args.pilot else list(range(314))
 for name in ['depth','frames','review']: (R/name).mkdir(parents=True,exist_ok=True)
 infer(ids);p=params(ids)
 if not args.pilot:
  p=gaussian_filter1d(p,2,axis=0);ALL_PARAMS=p
 receipts=[render(i,*v) for i,v in zip(ids,p)]
 (R/('pilot.json' if args.pilot else 'render.json')).write_text(json.dumps(receipts,indent=2));review(ids)
 if args.pilot:return
 ff=RAW/'runtime/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2';video=R/'luster_hybrid_depth_dof.mp4'
 with (R/'encode.log').open('w') as f:subprocess.run([str(ff),'-hide_banner','-y','-framerate','24','-i',str(R/'frames/%05d.png'),'-frames:v','314','-c:v','libx264','-preset','slow','-crf','14','-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(video)],stdout=f,stderr=f,check=True)
 cap=cv2.VideoCapture(str(video));n=0
 while True:
  ok,a=cap.read()
  if not ok:break
  assert a.shape==(2048,1500,3);n+=1
 cap.release();assert n==314
 (R/'validation.json').write_text(json.dumps(dict(decoded_frames=n,fps=24,depth_model='Depth Anything V2 Large',silhouette_masks_read=False,depth_metric=False,source='v4 unblurred + v8-v9 hybrid appearance residual',visual_review_pending=True),indent=2))
 print('DONE',flush=True)
PLAN=json.loads((RAW/'plan.json').read_text())
if __name__=='__main__':
 cv2.setNumThreads(4);main()
