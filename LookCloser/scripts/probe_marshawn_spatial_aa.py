"""Frozen native-field pixel-footprint supersampling diagnostic; no NN changes."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');S=R/'native_png_MVSdepth14k_STAGED'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inverse_display(x):
 x=np.asarray(x,np.float64)
 if not np.isfinite(x).all()or np.any(x<0)or np.any(x>1):raise ValueError('Finite native encoded RGB in [0,1] required')
 y=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 with np.errstate(divide='ignore',invalid='ignore'):return y/(1-y)
def display(x):
 with np.errstate(invalid='ignore'):y=np.where(np.isposinf(x),1,x/(1+x))
 return np.where(y<=.0031308,12.92*y,1.055*np.maximum(y,0)**(1/2.4)-.055).astype(np.float32)
def area(x,factor):
 h,w=x.shape[:2];shape=(h//factor,factor,w//factor,factor)+x.shape[2:];return x.reshape(shape).mean(axis=(1,3))
def save(p,x):Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(p)
def main(a):
 for p in reversed([R/'code_native',R/'code_native/LookCloser/scripts',R/'code_sfm/LookCloser/scripts',R/'final_pair/code',S/'code']):sys.path.insert(0,str(p))
 import torch,cv2
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import digest_model
 from marshawn_release_common import native_bounds
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory);cv2.setNumThreads(2)
 gridrec=json.loads((S/'grids/step-000016000_grid.json').read_text());ck=Path(gridrec['checkpoint']);assert sha(ck)=='5aa08c7cc725cb38e1de33141acc9a5dfd77041d9e808d7d9d97c3832a4c11f2'
 def configure(c):c.eval_checkpoint=ck;c.load_checkpoint=None;c.load_dir=None;c.load_step=None;c.load_config=None;return c
 cfg,pipe,loaded,step=eval_setup(S/'config.yml',test_mode='val',eval_num_rays_per_chunk=a.chunk,update_config_callback=configure);assert Path(loaded).resolve()==ck.resolve()and step==16000;pipe.eval();m=pipe.model;ds=pipe.datamanager.eval_dataset;bounds=native_bounds(m);before=digest_model(m)
 with np.load(S/'grids/step-000016000_grid.npz')as z:core=z['active'].copy();ema=z['ema'].copy()
 assert np.array_equal(core,m.occupancy_grid.binaries.cpu().numpy())and np.array_equal(ema.reshape(-1),m.occupancy_grid.occs.cpu().numpy())
 assert m.config.render_step_size==.003 and cfg.pipeline.final_grid_radius==0
 a.output.mkdir(exist_ok=False);rows=[]
 for name,bbox in [('KA',[60,630,600,900]),('HC',[1250,650,1550,900])]:
  if a.camera and name not in a.camera:continue
  idx=[Path(p).stem.rsplit('_',1)[-1]for p in ds.image_filenames].index(name);source=ds.get_image_float32(idx).cpu().numpy()[...,:3];x0,y0,x1,y1=bbox;gt=source[y0:y1,x0:x1];H,W=gt.shape[:2];original=copy.deepcopy(ds.cameras[idx:idx+1]).to(m.device);K0={k:getattr(original,k).cpu().tolist()for k in ['fx','fy','cx','cy','width','height']};results={};saved_path=S/'gate16k_HD'/('eval_'+name+'_buffers.npz')
  with np.load(saved_path)as z:saved=z['rgb'][y0:y1,x0:x1].copy();saved_gt=z['gt'][y0:y1,x0:x1].copy()
  assert np.array_equal(saved_gt,gt)
  for scale in [1,2,4]:
   cam=copy.deepcopy(original);cam.fx*=scale;cam.fy*=scale;cam.cx=(cam.cx-x0)*scale;cam.cy=(cam.cy-y0)*scale;cam.width.fill_(W*scale);cam.height.fill_(H*scale);assert torch.equal(cam.camera_to_worlds,original.camera_to_worlds)
   start=time.monotonic()
   with torch.inference_mode():outputs=m.get_outputs_for_camera(cam)
   rgb=outputs['rgb'][...,:3].detach().float().cpu().numpy();alpha=outputs['accumulation'].detach().float().cpu().numpy().squeeze(-1);depth=outputs['depth'].detach().float().cpu().numpy().squeeze(-1);assert rgb.shape==(H*scale,W*scale,3)and np.isfinite(rgb).all()
   if scale==1:
    parity=float(np.max(np.abs(rgb-saved)));assert parity<2e-5,('SavedHDparity',parity)
   encoded=area(rgb,scale).astype(np.float32);linear=display(area(inverse_display(rgb),scale));lowalpha=area(alpha,scale).astype(np.float32)
   np.savez_compressed(a.output/(name+'_SS'+str(scale)+'_buffers.npz'),rgb=rgb,alpha=alpha,depth=depth,encoded_area=encoded,inverse_display_linear_area=linear,alpha_area=lowalpha,gt_1x=gt)
   result=dict(scale=scale,ray_count=H*W*scale**2,render_seconds=time.monotonic()-start,render_K={k:getattr(cam,k).cpu().tolist()for k in ['fx','fy','cx','cy','width','height']},camera_to_world=cam.camera_to_worlds.cpu().tolist(),supersample_raw_float_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),alpha_min=float(alpha.min()),encoded_RGB_equal1_count=int((rgb==1).sum()),metrics={})
   for label,pred in [('encoded_area',encoded),('linear_area',linear)]:
    p=torch.tensor(pred,device=m.device).permute(2,0,1)[None];t=torch.tensor(gt,device=m.device).permute(2,0,1)[None]
    with torch.inference_mode():metrics=dict(psnr=float(-10*torch.log10((p-t).square().mean())),ssim=float(m.ssim(p,t,data_range=1.0)),lpips=float(m.lpips(p,t)))
    residual=pred-gt;blur=cv2.GaussianBlur(residual,(0,0),8,borderType=cv2.BORDER_REFLECT_101);Y=blur@np.array([.2126,.7152,.0722]);metrics.update(lowpass_sigma8_RGB_std=float(blur.std()),lowpass_sigma8_RGB_rmse=float(np.sqrt((blur**2).mean())),lowpass_sigma8_luma_std=float(Y.std()),lowpass_sigma8_luma_mean=float(Y.mean()));result['metrics'][label]=metrics;save(a.output/(name+'_SS'+str(scale)+'_'+label+'.png'),pred);results[(scale,label)]=pred
   if scale==1:result['baseline_max_abs_parity_to_saved_HD']=parity
   rows.append(dict(camera=name,bbox=bbox,source_K=K0,source_image_sha256=sha(ds.image_filenames[idx]),saved_HD_buffer_sha256=sha(saved_path),**result));(a.output/'progress.json').write_text(json.dumps(dict(completed=[(q['camera'],q['scale'])for q in rows]))+'\n');print(json.dumps(rows[-1]),flush=True)
  # Prediction-only outputs are distinct from GT contacts. All comparisons use fixed original GT.
  labels=['GT','1x','2x encoded','4x encoded','2x linear','4x linear'];panels=[gt,results[(1,'encoded_area')],results[(2,'encoded_area')],results[(4,'encoded_area')],results[(2,'linear_area')],results[(4,'linear_area')]];canvas=Image.new('RGB',(W*3,(H+24)*2),'#202020');draw=ImageDraw.Draw(canvas)
  for j,(label,pred)in enumerate(zip(labels,panels)):
   xx=(j%3)*W;yy=(j//3)*(H+24);canvas.paste(Image.fromarray(np.rint(np.clip(pred,0,1)*255).astype(np.uint8)),(xx,yy+24));draw.text((xx+4,yy+4),name+' '+label,fill='white')
  canvas.save(a.output/(name+'_spatial_AA_contact.jpg'),quality=95)
 assert digest_model(m)==before and np.array_equal(core,m.occupancy_grid.binaries.cpu().numpy())and np.array_equal(ema.reshape(-1),m.occupancy_grid.occs.cpu().numpy())
 report=dict(checkpoint=str(ck),checkpoint_sha256=sha(ck),config_sha256=sha(S/'config.yml'),source_grid_sha256=sha(S/'grids/step-000016000_grid.npz'),helper_sha256=sha(__file__),step=16000,default_core0=True,render_step_world=.003,chunk=a.chunk,live_bounds=bounds,rows=rows,formula='K:fx/fy*S,cx=(cx-x0)*S,cy=(cy-y0)*S; same pose. Box area mean of encoded RGB or inverse sRGB+inverse Reinhard linear light then forward display. Fixed camera gain/EV cancels.',linear_inversion_encoded1_policy='Exact infinite inverse-Reinhard at encoded1; limit maps back to1 after averaging. No arbitrary exposure fitting/capping.',frozen_state_unchanged=True,state_sha256=before,no_GT_compositing=True,no_training_or_worker_resume=True,optimizer_steps=0,metric_space='Original1x encoded sRGB target; float beforePNG quantization',limitations=['Diagnostic supersampling only; no change to learned field, occupancy or along-raydt.','Lowpass residual sigma8 is a descriptive crop statistic, not a separate ground truth.']);(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',a.output,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--chunk',type=int,default=2048);p.add_argument('--camera',action='append');main(p.parse_args())
