"""Read-only fixed10k/12k actual train/eval floor RGB diagnosis; geometry-only pixel selection."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np,cv2
R=Path('/fsx/tmp/marshawn_001660_hd');OUT=R/'native_png_random10k_floor_band_PRIOR_STAGED/floor_train_colour_diagnosis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',R/'final_pair/code',R/'native_png_batch1M_resume8k_randomBG_ALL3_STAGED/code',R/'native_png_random10k_floor_band_PRIOR_STAGED/code']:sys.path.insert(0,str(p))
 import torch
 from PIL import Image,ImageDraw
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import digest_model
 from audit_marshawn_geometry import projection,project_points
 torch.set_num_threads(2);cv2.setNumThreads(1);OUT.mkdir(exist_ok=False);data=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';tf=json.loads((data/'transforms.json').read_text());rows={q['physical_camera']:q for q in tf['frames']};proposal=json.loads((R/'floor_band_prior_PROPOSAL/proposal.json').read_text());polys={q['camera']:np.array(q['clear_floor_polygon'],np.float32)for q in proposal['floor_source_visuals']};z=np.load(R/'floor_band_prior_PROPOSAL/finite_domain_triangles_PROPOSAL.npz');n=z['normal'];d=float(z['offset']);half=float(z['halfwidth']);origin=z['grid_xy_origin'];step=float(z['grid_xy_step']);cellset={tuple(k)for k in z['valid_xy_cell_indices']};verts=z['triangles_xyz'].reshape(-1,3);selections={};names=['MA','NA','NB','AB','BB','KA'];rng=np.random.RandomState(42)
 for name in names:
  row=rows[name];uv,_=project_points(row,np.r_[verts-half*n,verts+half*n]);low=np.maximum(np.floor(uv.min(0)).astype(int),0);high=np.minimum(np.ceil(uv.max(0)).astype(int),[1920,1012]);xx,yy=np.meshgrid(np.arange(low[0],high[0]),np.arange(low[1],high[1]));pixels=np.c_[xx.ravel(),yy.ravel()];pose=np.array(row['transform_matrix']);v=np.c_[(pixels[:,0]+.5-row['cx'])/row['fl_x'],-(pixels[:,1]+.5-row['cy'])/row['fl_y'],-np.ones(len(pixels))]@pose[:3,:3].T;v/=np.linalg.norm(v,axis=1)[:,None];t=-(pose[:3,3]@n+d)/(v@n);hit=pose[:3,3]+t[:,None]*v;cells=np.floor((hit[:,:2]-origin)/step).astype(int);valid=np.array([tuple(k)in cellset for k in cells])&(t>0)
  if name in polys:
   for side in [-half,0,half]:
    u,_=project_points(row,hit+side*n);valid&=np.array([cv2.pointPolygonTest(polys[name],tuple(a),True)>8 for a in u])
  keep=np.flatnonzero(valid);ids=rng.choice(keep,min(1536,len(keep)),replace=False);pixels=pixels[ids];assert len(pixels)>0;selections[name]=dict(pixels=pixels.tolist(),candidate_pixels=int(valid.sum()),source=str(data/row['file_path']),source_sha256=sha(data/row['file_path']),domain_bbox=[*low.tolist(),*high.tolist()],semantic_train_floor_only=name!='KA',selector_uses_GT_RGB=False)
 (OUT/'fixed_pixel_selection.json').write_text(json.dumps(selections,indent=2)+'\n');models={
 'randomBG10k':(R/'native_png_batch1M_resume8k_randomBG_ALL3_STAGED',10000,'marshawn_001660_resume8k_randomBG_ALL3','warm_parent8000'),
 'floorPrior12k':(R/'native_png_random10k_floor_band_PRIOR_STAGED',12000,'marshawn_001660_random10k_finite_floor_PRIOR','warm_parent10000')};stats=[];saved={}
 for label,(stage,iteration,exp,stamp)in models.items():
  ckpt=stage/'run'/exp/'instant-ngp-bounded'/stamp/'nerfstudio_models'/f'step-{iteration:09d}.ckpt';config=stage/'config.yml'
  def configure(cfg):
   cfg.eval_checkpoint=ckpt;cfg.load_dir=None;cfg.load_step=None;cfg.load_checkpoint=None;cfg.load_config=None;return cfg
  _,pipe,_,loaded=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);pipe.eval();model=pipe.model;before=digest_model(model);saved[label]={}
  for name in names:
   ds=pipe.datamanager.eval_dataset if name=='KA'else pipe.datamanager.train_dataset;ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index(name);camera=ds.cameras[ci:ci+1].to(model.device);pix=np.array(selections[name]['pixels']);coords=torch.as_tensor(pix[:,::-1].copy()+.5,device=model.device);ray=camera.generate_rays(0,coords=coords,keep_shape=False);values={}
   with torch.inference_mode():
    chunks=[]
    for start in range(0,len(pix),512):chunks.append(model(ray[start:start+512]))
   for key in ['rgb','accumulation','depth']:values[key]=torch.cat([q[key]for q in chunks]).detach().float().cpu().numpy()
   source_gt=ds.get_image_float32(ci).cpu().numpy();gt=source_gt[pix[:,1],pix[:,0],:3];pred=values['rgb'];err=pred-gt;stat=dict(model=label,camera=name,split='eval'if name=='KA'else'train',pixels=len(pix),RMSE=float(np.sqrt(np.mean(err**2))),PSNR=float(-10*np.log10(np.mean(err**2))),mean_error_RGB=err.mean(0).tolist(),median_error_RGB=np.median(err,0).tolist(),GT_mean_RGB=gt.mean(0).tolist(),pred_mean_RGB=pred.mean(0).tolist(),median_opacity=float(np.median(values['accumulation'])),min_opacity=float(values['accumulation'].min()),checkpoint_sha256=sha(ckpt));stats.append(stat);saved[label][name]=dict(gt=gt,**values);np.savez_compressed(OUT/f'{label}_{name}_samples.npz',pixels=pix,gt=gt,**values)
   # Actual native context crop, same bbox for both snapshots; no GT-conditioned replacement.
   x0,y0,x1,y1=selections[name]['domain_bbox'];cam=copy.deepcopy(camera);cam.cx-=int(x0);cam.cy-=int(y0);cam.width.fill_(int(x1-x0));cam.height.fill_(int(y1-y0))
   with torch.inference_mode():outputs=model.get_outputs_for_camera(cam)
   rgb=outputs['rgb'].detach().float().cpu().numpy();np.savez_compressed(OUT/f'{label}_{name}_crop.npz',rgb=rgb,gt=source_gt[y0:y1,x0:x1,:3],bbox=[x0,y0,x1,y1]);Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(OUT/f'{label}_{name}_crop.png');(OUT/'progress.json').write_text(json.dumps(dict(model=label,camera=name)))
  assert digest_model(model)==before;del pipe,model;torch.cuda.empty_cache()
 for name in names:
  old=np.load(OUT/f'randomBG10k_{name}_crop.npz');new=np.load(OUT/f'floorPrior12k_{name}_crop.npz');imlist=[old['gt'],old['rgb'],new['rgb'],np.abs(new['rgb']-old['gt'])*5];h,w=old['gt'].shape[:2];sheet=Image.new('RGB',(w*4,h+25),'black');draw=ImageDraw.Draw(sheet)
  for j,(im,title)in enumerate(zip(imlist,['GT','parent10k','floorprior12k','5xabs12kerror'])):sheet.paste(Image.fromarray(np.rint(np.clip(im,0,1)*255).astype(np.uint8)),(j*w,25));draw.text((j*w+2,4),name+' '+title,fill='white')
  sheet.save(OUT/(name+'_contact.png'))
 result=dict(rows=stats,source_selection_sha256=sha(OUT/'fixed_pixel_selection.json'),script_sha256=sha(__file__),same_pixels_both_snapshots=True,actual_native_rays_and_core0=True,full_model_state_unchanged=True,no_training_or_colourgains_fit=True,no_GT_compositing=True,scope='EncodedSDR localfloor photometric residual; camera footprint/support differ. Semantictrainpolygons and finiteprior3Ddomain select pixels without GTtexture/exposure fitting; KAdomain usesgeometryonly.',limitations=['Chosen floor depth is soft scene prior, not measured correspondence.','Differentviewdirection/footprint and radiometry confound directpercamera causality.','NativegridcancontributesecondaryRGBerror; doesnotidentifyBRDF.']);(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
