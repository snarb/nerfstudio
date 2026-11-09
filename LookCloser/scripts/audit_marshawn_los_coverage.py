"""Readonly observed TRAIN LOS optical-thickness census; no training or carving."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
R=Path('/fsx/tmp/marshawn_001660_hd')

def main(a):
 for p in[R/'code_sfm/LookCloser/scripts',R/'code_native/LookCloser/scripts',R/'code_native',a.stage/'code']:sys.path.insert(0,str(p))
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import digest_model,file_sha
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 a.output.mkdir(exist_ok=False);z=np.load(a.stage/'geometry/scene_medium.npz');o=z['origins'];v=z['directions'];start=z['t_start'];end=z['t_end'];L=end-start;N=len(start)
 def config(c):c.eval_checkpoint=a.checkpoint;return c
 cfg,pipe,loaded,step=eval_setup(a.stage/'config.yml',test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=config);pipe.eval();model=pipe.model;before=digest_model(model);results={};peaks={}
 def integrate(ids,n):
  tau=[];mx=[];tp=[]
  for begin in range(0,len(ids),32):
   ix=ids[begin:begin+32];unit=(torch.arange(n,device='cuda',dtype=torch.float32)+.5)/n;ss=torch.tensor(start[ix],dtype=torch.float32,device='cuda');length=torch.tensor(L[ix],dtype=torch.float32,device='cuda');tt=ss[:,None]+length[:,None]*unit;points=torch.tensor(o[ix],dtype=torch.float32,device='cuda')[:,None,:]+torch.tensor(v[ix],dtype=torch.float32,device='cuda')[:,None,:]*tt[...,None]
   with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):sigma=model.field.density_fn(points.reshape(-1,3)).float().reshape(len(ix),n)
   if not torch.isfinite(sigma).all():raise FloatingPointError('Nonfinite true field sigma')
   val=sigma.mean(1)*length;maxval,loc=sigma.max(1);pt=tt.gather(1,loc[:,None]).squeeze(1);tau.extend(val.cpu().numpy());mx.extend(maxval.cpu().numpy());tp.extend(pt.cpu().numpy())
  return np.array(tau),np.array(mx),np.array(tp)
 for n in[64,256,1024]:
  tau,mx,tt=integrate(np.arange(N),n);results[str(n)]=tau;peaks[str(n)]=(mx,tt);(a.output/'progress.json').write_text(json.dumps({'completed_resolution':n,'rays':N})+'\n')
 top=np.argsort(results['1024'])[-128:][::-1];refined,mx,tt=integrate(top,4096)
 sigma_max,tmax=peaks['1024'];q=np.array(json.loads(a.peak.read_text())['peak']['xyz']);tq=np.sum((q-o)*v,axis=1);tq=np.clip(tq,start,end);distance=np.linalg.norm(o+v*tq[:,None]-q,axis=1);near=np.argsort(distance)[:128]
 def row(i,tau=None,peak=None):
  return dict(interval_index=int(i),camera=str(z['camera_names'][int(z['camera_idx'][i])]),point_id=int(z['point_idx'][i]),UV_calibrated_reprojection=z['uv'][i].tolist(),witnessXYZ=z['witness_xyz'][i].tolist(),tau64=float(results['64'][i]),tau256=float(results['256'][i]),tau1024=float(results['1024'][i]),tau4096=None if tau is None else float(tau),max_sigma1024=float(sigma_max[i]),max_sigmaXYZ=(o[i]+v[i]*tmax[i]).tolist(),distance_to_q7_peak=float(distance[i]))
 summary={n:{'mean':float(x.mean()),'quantiles':{str(q):float(np.quantile(x,q))for q in[0,.1,.5,.9,.95,.99,.999,1]},'count_tau_gt01':int((x>.1).sum()),'count_tau_gt1':int((x>1).sum())}for n,x in results.items()}
 from PIL import Image,ImageDraw
 tf=json.loads((R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json').read_text());by={f['physical_camera']:f for f in tf['frames']};selected=[];seen=set()
 for j,i in enumerate(top):
  r=row(i,refined[j]);key=(r['camera'],round(r['UV_calibrated_reprojection'][0],1),round(r['UV_calibrated_reprojection'][1],1))
  if key not in seen:selected.append(r);seen.add(key)
  if len(selected)==24:break
 for page in range((len(selected)+11)//12):
  canvas=Image.new('RGB',(4*300,3*220));dr=ImageDraw.Draw(canvas)
  for j,r in enumerate(selected[page*12:page*12+12]):
   f=by[r['camera']];im=Image.open(R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png'/f['file_path']).convert('RGB');u,vv=r['UV_calibrated_reprojection'];xx=int(u)-75;yy=int(vv)-45;crop=im.crop((xx,yy,xx+150,yy+90));dd=ImageDraw.Draw(crop);dd.ellipse((u-xx-3,vv-yy-3,u-xx+3,vv-yy+3),outline='cyan',width=1);x0=j%4*300;y0=j//4*220;canvas.paste(crop.resize((300,180)),(x0,y0+40));dr.text((x0+4,y0+3),'%s interval%d p%d'%(r['camera'],r['interval_index'],r['point_id']),fill='white');dr.text((x0+4,y0+19),'tau4096 %.3g'%r['tau4096'],fill='white')
  canvas.save(a.output/('top_LOS_sources_%02d.jpg'%(page+1)))
 if digest_model(model)!=before:raise ValueError('Full field/grid/EMA state changed')
 np.savez_compressed(a.output/'tau_arrays.npz',tau64=results['64'],tau256=results['256'],tau1024=results['1024'],sigma_max1024=sigma_max,peak_t1024=tmax,distance_q7_peak=distance,top128_indices=top,top128_tau4096=refined,top128_sigma4096=mx,top128_peak_t4096=tt)
 report=dict(kind='READONLY_ALL44938_OBSERVED_TRAIN_LOS_CENSUS',checkpoint_sha256=file_sha(a.checkpoint),script_sha256=file_sha(__file__),geometry_sha256=file_sha(a.stage/'geometry/scene_medium.npz'),rays=N,summary=summary,top128=[row(i,refined[j])for j,i in enumerate(top)],near_q7_128=[row(i)for i in near],source_contact_rows=selected,original_field_EMA_grid_unchanged=True,optimizer_updates=0,all_ray_samples_inside_original_intervals=True,no_empty_volume_tubes=True,training_actual_sampling_coverage='UNKNOWN: no per-interval selected-index logs; no exact count inferred fromseed/expectedprobability',cap_GiB=6,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),tau_quadrature_is_diagnostic_not_physical_emptyspace_certificate=True,time=time.time())
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'summary':summary,'peak_reserved':report['peak_reserved_bytes']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in['stage','checkpoint','peak','output']:p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
