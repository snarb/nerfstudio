#!/usr/bin/env python3
"""Train-SfM floor plane and frozen Gaussian geometry diagnosis, no optimization."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
import torch
from audit_marshawn_geometry import project_points

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit_floor(points):
 candidates=points[points[:,2]<np.quantile(points[:,2],.40)];rng=np.random.default_rng(42);best=None
 for _ in range(6000):
  triplet=candidates[rng.choice(len(candidates),3,replace=False)];normal=np.cross(triplet[1]-triplet[0],triplet[2]-triplet[0]);length=np.linalg.norm(normal)
  if length<1e-10:continue
  normal/=length
  if abs(normal[2])<np.cos(np.deg2rad(35)):continue
  offset=-normal@triplet[0];inside=abs(candidates@normal+offset)<.003;count=inside.sum()
  if best is None or count>best[0]:best=(count,inside)
 if best is None:raise ValueError('No supported up-facing floor plane')
 selected=candidates[best[1]];center=selected.mean(0);_,_,vt=np.linalg.svd(selected-center);normal=vt[-1];normal*=np.sign(normal[2]);offset=-normal@center
 return normal,offset,abs(points@normal+offset)<.003

def rotation(q):
 q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1).reshape(-1,3,3)

def run(a):
 a.output.mkdir(parents=True,exist_ok=False);meta=json.loads((a.data/'transforms.json').read_text());train=set(meta['train_filenames']);points=np.load(a.points)['xyz'];normal,offset,inlier=fit_floor(points)
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['pipeline'];get=lambda key:state[next(k for k in state if k.endswith('gauss_params.'+key))].numpy()
 means=get('means');scales=np.exp(get('scales'));opacity=1/(1+np.exp(-get('opacities').reshape(-1)));q=get('quats');R=rotation(q);normal_local=np.einsum('nij,i->nj',R,normal);sigma_n=np.sqrt(np.sum((normal_local*scales)**2,axis=1));height=means@normal+offset;aspect=scales.max(1)/np.maximum(scales.min(1),1e-12)
 np.savez_compressed(a.output/'floor_and_gaussians.npz',normal=normal,offset=offset,points=points,plane_inliers=inlier,means=means,scales=scales,opacity=opacity,signed_height=height,normal_sigma=sigma_n,scale_aspect=aspect)
 pages=[];panels=[]
 for row in meta['frames']:
  if row['file_path'] not in train:continue
  image=Image.open(a.data/row['file_path']).convert('RGB');draw=ImageDraw.Draw(image);uv,front=project_points(row,points);inside=front&(uv[:,0]>=0)&(uv[:,0]<row['w'])&(uv[:,1]>=0)&(uv[:,1]<row['h'])
  for u,v in uv[inside&inlier]:draw.ellipse((u-2,v-2,u+2,v+2),fill=(0,255,50))
  image.thumbnail((640,337));draw=ImageDraw.Draw(image);draw.text((3,3),row['physical_camera']+' train-floor inliers green',fill='yellow');image.save(a.output/(row['physical_camera']+'_plane_inliers.jpg'));small=image.copy();small.thumbnail((320,169));panels.append((row['physical_camera'],small))
 for start in range(0,len(panels),21):
  canvas=Image.new('RGB',(960,1330));draw=ImageDraw.Draw(canvas)
  for j,(name,im) in enumerate(panels[start:start+21]):x=j%3*320;y=j//3*190;canvas.paste(im,(x,y+20));draw.text((x+3,y+3),name,fill='white')
  name=f'plane_train_contacts_{start//21+1:02d}.jpg';canvas.save(a.output/name,quality=92);pages.append(name)
 rois={'DC_mat':('DC',[160,560,850,920]),'HC_center_floor':('HC',[600,650,1400,1012]),'KA_feet':('KA',[780,480,1340,960])};regions=[]
 for name,(camera,bbox) in rois.items():
  row=next(r for r in meta['frames'] if r['physical_camera']==camera);uv,front=project_points(row,means);x0,y0,x1,y1=bbox;selected=front&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)&(opacity>.01)
  sph,spfront=project_points(row,points);sp=spfront&(sph[:,0]>=x0)&(sph[:,0]<x1)&(sph[:,1]>=y0)&(sph[:,1]<y1)
  h=height[selected];s=sigma_n[selected];regions.append(dict(region=name,camera=camera,bbox=bbox,projected_gaussian_centers=int(selected.sum()),mean_height_quantiles=np.quantile(h,[0,.1,.25,.5,.75,.9,1]).tolist(),normal_sigma_quantiles=np.quantile(s,[0,.5,.9,1]).tolist(),scale_aspect_quantiles=np.quantile(aspect[selected],[0,.5,.9,.99,1]).tolist(),center_above_floor_001=int((h>.01).sum()),center_three_sigma_above_floor=int((h>np.maximum(.01,3*s)).sum()),projected_SfM_points=int(sp.sum()),SfM_signed_height_quantiles=np.quantile(points[sp]@normal+offset,[0,.1,.5,.9,1]).tolist(),SfM_floor_inliers=int((sp&inlier).sum()),caveat='Projected centers can be occluded; counts are not pixel-contribution weights. Mat height above floor is genuine and not itself a floater.'))
 report=dict(checkpoint_sha256=sha(a.checkpoint),points_sha256=sha(a.points),transforms_sha256=sha(a.data/'transforms.json'),gaussians=len(means),plane=dict(normal=normal.tolist(),offset=float(offset),inliers=int(inlier.sum()),threshold=.003,normal_up_angle_degrees=float(np.degrees(np.arccos(normal[2]))),method='Train9022lower40%z,6000RANSACtriples,normalwithin35degup,3e-3normalized residual,least-squareinlierrefit',inlier_extent=[points[inlier].min(0).tolist(),points[inlier].max(0).tolist()],no_eval_RGB_used_for_fitting=True),regions=regions,pages=pages,limitations=['Floor normal constrained by prepared-world up; manual train-view overlays must confirm physical floor','A geometric floor plane does not establish thickness/shape of the padded mat','Gaussian centers alone do not prove which Gaussians generate colored pixels; native raster conic/depth contribution audit required'])
 (a.output/'geometry_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','points','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
 run(p.parse_args())
