#!/usr/bin/env python3
"""Current-frame, train-only calibrated SIFT witnesses and provisional AABBs.

Bounds are evidence envelopes, NOT certified silhouettes. Scene extrema and
subject ROIs need human review before approving boxes for training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw


def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def projection(row):
 pose=np.asarray(row['transform_matrix'],float)
 r=np.diag([1.,-1.,-1.])@pose[:3,:3].T
 t=-r@pose[:3,3]
 k=np.array([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1.]])
 return k@np.column_stack([r,t]),k,r,t

def fundamental(a,b):
 _,ka,ra,ta=projection(a);_,kb,rb,tb=projection(b)
 r=rb@ra.T;t=tb-r@ta
 cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
 return np.linalg.inv(kb).T@cross@r@np.linalg.inv(ka)

def sampson(f,a,b):
 aa=np.column_stack([a,np.ones(len(a))]);bb=np.column_stack([b,np.ones(len(b))])
 fa=aa@f.T;fb=bb@f
 return np.abs(np.sum(bb*fa,axis=1))/np.sqrt(np.maximum(fa[:,0]**2+fa[:,1]**2+fb[:,0]**2+fb[:,1]**2,1e-30))

def triangulate(a,b,ua,ub):
 pa,*_=projection(a);pb,*_=projection(b)
 xy=cv2.triangulatePoints(pa,pb,ua.T,ub.T).T
 xyz=xy[:,:3]/xy[:,3:4]
 ca=(pa@np.column_stack([xyz,np.ones(len(xyz))]).T).T
 cb=(pb@np.column_stack([xyz,np.ones(len(xyz))]).T).T
 error=np.maximum(np.linalg.norm(ca[:,:2]/ca[:,2:3]-ua,axis=1),np.linalg.norm(cb[:,:2]/cb[:,2:3]-ub,axis=1))
 centers=[np.asarray(x['transform_matrix'])[:3,3] for x in [a,b]]
 va=xyz-centers[0];vb=xyz-centers[1]
 cosine=np.sum(va*vb,axis=1)/np.maximum(np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1),1e-30)
 angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
 valid=np.isfinite(xyz).all(1)&(ca[:,2]>0)&(cb[:,2]>0)&(error<1)&(angle>1)&(angle<80)
 return xyz,valid,error,angle

def project_points(row,xyz):
 p,*_=projection(row);v=np.column_stack([xyz,np.ones(len(xyz))])@p.T
 return v[:,:2]/np.maximum(v[:,2:3],1e-20),v[:,2]>0

def envelope(xyz):
 # Keep all geometrically validated witnesses; extremes are manually reviewed.
 low=xyz.min(0);high=xyz.max(0);center=(low+high)/2;span=np.maximum(high-low,.001)
 return {name:np.stack([center-span*s/2,center+span*s/2]).tolist() for name,s in [('tight',1.05),('medium',1.15),('loose',1.30)]}

def audit(data,out,workers,neighbors,roi_file):
 cv2.setNumThreads(1);out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((data/'transforms.json').read_text());rows=[r for r in meta['frames'] if r['file_path'] in meta['train_filenames']]
 if len(rows)!=63:raise ValueError('Expected exactly 63 train images')
 feat=out/'features';feat.mkdir(exist_ok=True)
 def features(row):
  name=row['physical_camera'];p=data/row['file_path'];cache=feat/(name+'.npz');sha=digest(p)
  if cache.exists():
   z=np.load(cache)
   if str(z['sha'])==sha:return z['uv'],z['des']
  im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);mask=cv2.imread(str(data/row['mask_path']),cv2.IMREAD_GRAYSCALE)
  kp,des=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.025).detectAndCompute(im,mask)
  uv=np.array([k.pt for k in kp],np.float32)+.5
  np.savez_compressed(cache,uv=uv,des=des,sha=sha)
  print('features',name,len(uv),flush=True);return uv,des
 with ThreadPoolExecutor(max_workers=workers) as pool:features_all=list(pool.map(features,rows))
 centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
 distance=np.linalg.norm(centers[:,None]-centers[None],axis=-1)
 pairs=sorted(set(tuple(sorted((i,int(j)))) for i in range(len(rows)) for j in np.argsort(distance[i])[1:neighbors+1]))
 points=[];records=[];source_pairs=[];edges=[]
 matcher=cv2.BFMatcher()
 for i,j in pairs:
  ua,da=features_all[i];ub,db=features_all[j]
  matches=matcher.knnMatch(da,db,k=2);good=[m for m,n in matches if m.distance<.65*n.distance]
  if len(good)<8:continue
  a=ua[[m.queryIdx for m in good]];b=ub[[m.trainIdx for m in good]]
  ep=sampson(fundamental(rows[i],rows[j]),a,b);qualified=ep<1.
  xyz,valid,err,ang=triangulate(rows[i],rows[j],a,b);valid&=qualified
  points.append(xyz[valid]);source_pairs.extend([[i,j]]*int(valid.sum()))
  edges.extend([((i,good[k].queryIdx),(j,good[k].trainIdx)) for k in np.flatnonzero(valid)])
  records.append(dict(a=rows[i]['physical_camera'],b=rows[j]['physical_camera'],matches=len(good),accepted=int(valid.sum()),epipolar_median=float(np.median(ep)),epipolar_p90=float(np.quantile(ep,.9)),geometric_inlier_fraction=float(qualified.mean())))
  print('pair',rows[i]['physical_camera'],rows[j]['physical_camera'],'points',int(valid.sum()),flush=True)
 if not points:raise ValueError('No triangulated points')
 # Require >=3 distinct-camera feature tracks, with consistent final reprojection.
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for x,y in edges:
  a=find(x);b=find(y)
  if a!=b:parent[a]=b
 tracks={}
 for x in parent:tracks.setdefault(find(x),[]).append(x)
 strict=[];strict_sources=[]
 for track in tracks.values():
  if len(track)<3 or len(set(i for i,k in track))!=len(track):continue
  equations=[]
  for i,k in track:
   p,*_=projection(rows[i]);u,v=features_all[i][0][k];equations.extend([u*p[2]-p[0],v*p[2]-p[1]])
  _,_,vt=np.linalg.svd(np.asarray(equations),full_matrices=False);hom=vt[-1]
  if abs(hom[3])<1e-12:continue
  x=hom[:3]/hom[3];ok=True
  for i,k in track:
   p,*_=projection(rows[i]);q=p@np.r_[x,1.];uv=features_all[i][0][k]
   if q[2]<=0 or np.linalg.norm(q[:2]/q[2]-uv)>1.:ok=False;break
  if ok:strict.append(x);strict_sources.append([i for i,k in track])
 xyz=np.asarray(strict);source_pairs=np.array([v[:2] for v in strict_sources])
 if len(xyz)<100:raise ValueError('Insufficient consistent three-view geometry')
 write(out/'strict_tracks.json',dict(points=len(xyz),camera_ids=strict_sources))
 np.savez_compressed(out/'train_witnesses.npz',xyz=xyz,source_pairs=source_pairs,cameras=np.array([r['physical_camera'] for r in rows]))
 scopes={'scene':envelope(xyz)};roi_note='subject missing: require reviewed train-only rectangles'
 if roi_file:
  rois=json.loads(roi_file.read_text());votes=np.zeros(len(xyz),int)
  for name,rect in rois.items():
   r=next(r for r in rows if r['physical_camera']==name);uv,front=project_points(r,xyz);x0,y0,x1,y1=rect
   votes+=front&(uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1)
  sub=xyz[votes>=2]
  if len(sub)<30:raise ValueError('Insufficient subject witnesses')
  scopes['subject']=envelope(sub);np.savez_compressed(out/'subject_witnesses.npz',xyz=sub);roi_note='subject requires visual review; two-view rectangular support is not segmentation'
 boxes=dict(coordinate_system='prepared_world',normalization_sha256=digest(data/'preparation.json'),approved=False,scopes=scopes,caveat='Current train-only SIFT evidence envelope; sparse features do not guarantee limb/hair/requisite containment. '+roi_note)
 write(out/'boxes_candidates.json',boxes)
 write(out/'audit.json',dict(train=63,eval_used=False,points=len(xyz),pairs=records,transforms_sha256=digest(data/'transforms.json'),script_sha256=digest(__file__),bounds_certified=False))
 for r in rows:
  im=Image.open(data/r['file_path']).convert('RGB');draw=ImageDraw.Draw(im);uv,front=project_points(r,xyz)
  valid=front&(uv[:,0]>=0)&(uv[:,0]<r['w'])&(uv[:,1]>=0)&(uv[:,1]<r['h'])
  for u,v in uv[valid][::max(1,int(valid.sum())//3000)]:draw.ellipse((u-1,v-1,u+1,v+1),fill=(0,255,100))
  for scope,variants in scopes.items():
   lo,hi=np.array(variants['tight']);corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]]);p,front=project_points(r,corners)
   for k in range(8):
    for bit in [1,2,4]:
     n=k^bit
     if n>k and front[k] and front[n]:draw.line([tuple(p[k]),tuple(p[n])],fill='yellow' if scope=='scene' else 'red',width=3)
  im.thumbnail((960,600));im.save(out/(r['physical_camera']+'_witnesses.jpg'))
 # Compact complete inventory for manual review; generated artifacts are diagnostics.
 files=sorted(out.glob('*_witnesses.jpg'))
 for page in range((len(files)+20)//21):
  canvas=Image.new('RGB',(1280,1330));draw=ImageDraw.Draw(canvas)
  for j,f in enumerate(files[page*21:(page+1)*21]):
   im=Image.open(f);im.thumbnail((320,170));xx=j%3*420;yy=j//3*190;canvas.paste(im,(xx,yy+20));draw.text((xx,yy),f.stem,fill='white')
  canvas.save(out/f'contact_{page+1}.jpg')
 print(json.dumps(dict(points=len(xyz),boxes=str(out/'boxes_candidates.json'))),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--neighbors',type=int,default=4);p.add_argument('--subject-rois',type=Path);a=p.parse_args();audit(a.data,a.output,a.workers,a.neighbors,a.subject_rois)
