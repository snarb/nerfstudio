"""208 TRAIN MVS rays: measured RGB, native volume, direct field and band color."""
from pathlib import Path
import argparse,json,sys,hashlib,copy
import numpy as np
R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 for p in [R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',a.stage/'code',Path(__file__).parent]:sys.path.insert(0,str(p))
 import torch,cv2
 from PIL import Image,ImageDraw
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.cameras.rays import RaySamples,Frustums,RayBundle
 from nerfstudio.field_components.field_heads import FieldHeadNames
 from audit_marshawn_geometry import project_points
 from probe_marshawn_occupancy import digest_model,integrate
 torch.set_num_threads(2)
 def configure(c):c.eval_checkpoint=a.checkpoint;c.load_checkpoint=c.load_dir=c.load_step=c.load_config=None;return c
 cfg,pipe,actual,step=eval_setup(a.config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure)
 assert actual.resolve()==a.checkpoint.resolve();model=pipe.model;pipe.eval();model.current_train_step=step;before=digest_model(model);grid=model.occupancy_grid.binaries.clone();ds=pipe.datamanager.train_dataset;names=[Path(p).stem.rsplit('_',1)[-1]for p in ds.image_filenames];data=Path(cfg.pipeline.datamanager.dataparser.data);tf=json.loads((data/'transforms.json').read_text());by={f['physical_camera']:f for f in tf['frames']};archive=Path(cfg.pipeline.MVS_depth_manifest);manifest=json.loads(archive.read_text());Z=np.load(manifest['archive']);xyz=Z['xyz'];normal=Z['normal'];cameras=Z['cameras'].tolist();ids=Z['candidate_ids'];assert cameras==['MA','NA','NB','LB'];assert set(cameras)<=set(names)
 origins=Z['origins'].reshape(-1,3);directions=Z['directions'].reshape(-1,3);truth=Z['depth'].reshape(-1);points=np.tile(xyz,(4,1));ci=np.repeat([names.index(c)for c in cameras],52);device=model.device
 rays=RayBundle(origins=torch.tensor(origins,device=device,dtype=torch.float32),directions=torch.tensor(directions,device=device,dtype=torch.float32),pixel_area=torch.ones((208,1),device=device),camera_indices=torch.tensor(ci[:,None],device=device))
 capture={}
 def field_hook(_m,args,out):
  s=args[0];capture.update(starts=s.frustums.starts.detach().float().cpu().numpy().reshape(-1),ends=s.frustums.ends.detach().float().cpu().numpy().reshape(-1),sigma=out[FieldHeadNames.DENSITY].detach().float().cpu().numpy().reshape(-1),rgb_samples=out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3))
 def depth_hook(_m,args,kwargs):capture.update(weights=kwargs['weights'].detach().float().cpu().numpy().reshape(-1),ray_indices=kwargs['ray_indices'].detach().cpu().numpy().reshape(-1))
 handles=[model.field.register_forward_hook(field_hook),model.renderer_depth.register_forward_pre_hook(depth_hook,with_kwargs=True)]
 try:
  with torch.no_grad():out=model(rays)
 finally:
  for h in handles:h.remove()
 native=out['rgb'].detach().float().cpu().numpy();opacity=out['accumulation'].detach().float().cpu().numpy().reshape(-1);offsets=np.array([-.005,-.001,0.,.001,.005]);positions=points[:,None,:]+offsets[None,:,None]*normal[None,None,:];D=rays.directions[:,None,:].expand(208,5,3).reshape(-1,3);cameraids=rays.camera_indices[:,None,:].expand(208,5,1).reshape(-1,1)
 sample=RaySamples(frustums=Frustums(origins=torch.tensor(positions.reshape(-1,3),device=device,dtype=torch.float32),directions=D,starts=torch.zeros((1040,1),device=device),ends=torch.zeros((1040,1),device=device),pixel_area=torch.ones((1040,1),device=device)),camera_indices=cameraids)
 with torch.no_grad():direct=model.field(sample)
 direct_rgb=direct[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(208,5,3);direct_sigma=direct[FieldHeadNames.DENSITY].detach().float().cpu().numpy().reshape(208,5);rows=[];GT=[];UV=[];images={};maxw=0.;reconstructed=[];bandweights=np.zeros_like(capture['weights'],dtype=np.float64)
 for cam in cameras:images[cam]=np.asarray(Image.open(data/by[cam]['file_path']).convert('RGB'),dtype=np.float32)/255.
 def gtmean(im,uv):
  delta=np.array([(x,y)for y in range(-2,3)for x in range(-2,3)]);xy=np.asarray(uv)[None]- .5+delta;return cv2.remap(im,xy[:,0].astype('float32')[:,None],xy[:,1].astype('float32')[:,None],cv2.INTER_LINEAR).reshape(-1,3).mean(0)
 for i in range(208):
  cam=cameras[i//52];j=i%52;uv=project_points(by[cam],xyz[j:j+1])[0][0];UV.append(uv);gt=gtmean(images[cam],uv);GT.append(gt);shift=np.array([gtmean(images[cam],uv+[x,y])for y in [-1,0,1]for x in [-1,0,1]]);sel=capture['ray_indices']==i;st,en,sig,w=[capture[k][sel].astype(float)for k in ['starts','ends','sigma','weights']];rgb=capture['rgb_samples'][sel];_,_,refw=integrate(st,en,sig);maxw=max(maxw,float(np.max(abs(w-refw),initial=0)));recon=(w[:,None]*rgb).sum(0);reconstructed.append(recon);cosine=abs(float(directions[i]@normal));lo,hi=truth[i]-.005/cosine,truth[i]+.005/cosine;begin=np.maximum(st,lo);end=np.minimum(en,hi);length=np.maximum(end-begin,0);alpha=-np.expm1(-sig*(en-st));T=np.divide(w,alpha,out=np.zeros_like(w),where=alpha>1e-30);bw=T*np.exp(-sig*np.maximum(begin-st,0))*(-np.expm1(-sig*length));bandweights[sel]=bw;mass=bw.sum();band=(bw[:,None]*rgb).sum(0)/mass if mass>1e-12 else None;perp=np.linalg.norm(points[i]-origins[i]-truth[i]*directions[i]);assert perp<2e-6
  # The generated camera ray is independently checked against the cached primitives.
  ray=ds.cameras[ci[i]:ci[i]+1].to(device).generate_rays(0,coords=torch.tensor(uv[::-1].copy()[None],device=device,dtype=torch.float32),keep_shape=False);dirdiff=float(np.linalg.norm(ray.directions[0].cpu().numpy()-directions[i]));assert dirdiff<2e-6
  row=dict(camera=cam,candidate_id=int(ids[j]),candidate_index=j,uv_COLMAP=uv.tolist(),GT_RGB_5x5=gt.tolist(),native_volume_RGB=native[i].tolist(),native_opacity=float(opacity[i]),native_conditional_RGB=(native[i]/opacity[i]).tolist()if opacity[i]>1e-12 else None,direct_XYZ_RGB=direct_rgb[i,2].tolist(),normal_offsets=offsets.tolist(),RGB_at_normal_offsets=direct_rgb[i].tolist(),sigma_at_normal_offsets=direct_sigma[i].tolist(),normal005_band_termination_mass=float(mass),band_conditional_RGB=band.tolist()if band is not None else None,GT_shift_plusminus1px_meanRGB_range=(shift.max(0)-shift.min(0)).tolist(),ray_XYZ_perpendicular_world_error=float(perp),provided_generated_direction_error=dirdiff,native_absolute_RGB_error=(native[i]-gt).tolist(),direct_absolute_RGB_error=(direct_rgb[i,2]-gt).tolist(),band_absolute_RGB_error=(band-gt).tolist()if band is not None else None)
  rows.append(row)
 maxrgb=float(np.max(abs(np.asarray(reconstructed)-native)));assert maxrgb<2e-5
 a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'cached_geometry_native_profiles.npz',**capture,partial_band_weights=bandweights,xyz=points,origins=origins,directions=directions,MVS_ray_distance=truth,uv=np.array(UV),GT_5x5=np.array(GT),native_RGB=native,direct_RGB_at_normal_offsets=direct_rgb,direct_sigma=direct_sigma,camera_names=np.repeat(np.array(cameras),52),candidate_ids=np.tile(ids,4))
 scores={}
 for label,key in [('native_volume','native_volume_RGB'),('direct_XYZ','direct_XYZ_RGB'),('band_conditional','band_conditional_RGB')]:
  valid=[r for r in rows if r[key]is not None];e=np.array([np.array(r[key])-r['GT_RGB_5x5']for r in valid]);scores[label]=dict(rows=len(valid),signed_channel_mean=e.mean(0).tolist(),channel_MAE=np.abs(e).mean(0).tolist(),channel_RMSE=np.sqrt((e*e).mean(0)).tolist(),median_perray_RGB_MAE=float(np.median(np.abs(e).mean(1))),p90_perray_RGB_MAE=float(np.quantile(np.abs(e).mean(1),.9)))
 point_error=np.array([np.mean([np.abs(np.array(r['native_volume_RGB'])-r['GT_RGB_5x5']).mean()for r in rows if r['candidate_index']==j])for j in range(52)]);worst=np.argsort(point_error)[-6:][::-1];typical=[j for j in np.argsort(abs(point_error-np.median(point_error)))if j not in worst][:6];saved={}
 for group,selected in [('worst6',worst),('typical6',typical)]:
  sheet=Image.new('RGB',(4*250,6*145),'#171717');draw=ImageDraw.Draw(sheet)
  for yy,j in enumerate(selected):
   for xx,cam in enumerate(cameras):
    i=xx*52+int(j);uv=np.array(UV[i]);x0,y0=np.floor(uv-.5).astype(int)-10;camera=copy.deepcopy(ds.cameras[ci[i]:ci[i]+1]).to(device);camera.cx-=int(x0);camera.cy-=int(y0);camera.width.fill_(21);camera.height.fill_(21)
    with torch.no_grad():result=model.get_outputs_for_camera(camera)
    prediction=result['rgb'].detach().float().cpu().numpy();target=images[cam][y0:y0+21,x0:x0+21];saved[f'{group}_p{int(ids[j])}_{cam}_GT']=target;saved[f'{group}_p{int(ids[j])}_{cam}_native']=prediction
    for k,im in enumerate([target,prediction]):tile=Image.fromarray(np.rint(np.clip(im,0,1)*255).astype('uint8')).resize((84,84),Image.Resampling.NEAREST);sheet.paste(tile,(xx*250+k*86,yy*145+25))
    draw.text((xx*250+2,yy*145+3),f'p{int(ids[j])} {cam} GT | native',fill='white');r=rows[i]
    for k,key in enumerate(['GT_RGB_5x5','native_volume_RGB','direct_XYZ_RGB','band_conditional_RGB']):
     color=r[key]if r[key]is not None else [1,0,1];draw.rectangle((xx*250+k*58,yy*145+113,xx*250+k*58+55,yy*145+131),fill=tuple(np.rint(np.clip(color,0,1)*255).astype(int)))
    draw.text((xx*250+2,yy*145+132),'GT / volume / XYZ / band',fill='white')
  sheet.save(a.output/(group+'_TRAIN_contacts.jpg'),quality=97)
 np.savez_compressed(a.output/'selected_actual_native_patches.npz',**saved);assert digest_model(model)==before and torch.equal(grid,model.occupancy_grid.binaries)
 report=dict(checkpoint=str(actual),checkpoint_sha256=sha(actual),config_sha256=sha(a.config),source_sha256=sha(__file__),MVS_archive_sha256=sha(Path(manifest['archive'])),transforms_sha256=sha(data/'transforms.json'),source_images_sha256={c:sha(data/by[c]['file_path'])for c in cameras},geometry_cache='cached_geometry_native_profiles.npz',rows=rows,scores=scores,native_RGB_from_actual_weights_colors_maxerror=maxrgb,native_weights_density_maxerror=maxw,selected_worst6_ids=[int(ids[j])for j in worst],selected_typical6_ids=[int(ids[j])for j in typical],field_grid_unchanged=True,uses_eval_RGB=False,optimizer_updates=0,view_color_averaging=False,band_halfwidth_world_normal=.005,limitations=['MVS52 are soft TRAIN photometric candidates, not scannerGT or calibrated confidence intervals','GT5x5mean versus infinitesimal directfield has pixelfootprint/registration/reflectance uncertainty; sourceplusminus1px ranges retained','21x21patchrenders preserveeach actualcameradirection; no normalaveraging','Bandcolor undefined for zero terminationmass, explicitly reported ratherthanfilled'])
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(scores=scores,RGBerror=maxrgb,weightserror=maxw,field_grid_unchanged=True)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();
 for k in ['config','checkpoint','stage','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
