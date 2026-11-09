"""Localized raw8K train-only SfM. Approximate plane only scopes image crops."""
from pathlib import Path
import json,hashlib,itertools,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
from audit_marshawn_geometry import projection,fundamental,sampson
R=Path('/fsx/tmp/marshawn_001660_hd');OUT=R/'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit/raw8k_floor_sfm';RAW=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local');RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
NAMES=['AA','AB','BB','BC','LA','MA','NA','NB']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2)+'\n')
def main():
 cv2.setNumThreads(2);OUT.mkdir(exist_ok=False,parents=True);tf=json.loads((R/'data/transforms.json').read_text());by={q['physical_camera']:q for q in tf['frames'] if q['file_path']in tf['train_filenames']};assert set(NAMES)<=set(by)
 support=json.loads((OUT.parent/'floor_reference_support.json').read_text());target=np.array([q['reference_plane_XYZ']for q in support['rows']]);scope=np.concatenate([target+np.array([dx,dy,0])for dx,dy in itertools.product([-.16,.16],repeat=2)])
 cams=read_cameras_text(RIG/'cameras.txt');ims=read_images_text(RIG/'images.txt');rig={q.name.split('/')[0]:cams[q.camera_id]for q in ims.values()};rows=[];uvs=[];descs=[];crops=[];sources=[]
 for name in NAMES:
  row=dict(by[name]);cam=rig[name];fx,fy,cx,cy,*dist=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  row.update(fl_x=float(fx),fl_y=float(fy),cx=float(cx),cy=float(cy),w=8192,h=4320);P,*_=projection(row);q=np.c_[scope,np.ones(len(scope))]@P.T;q=q[:,:2]/q[:,2:];low=np.maximum(np.floor(q.min(0)-180).astype(int),0);high=np.minimum(np.ceil(q.max(0)+180).astype(int),[8192,4320]);x0,y0=low;x1,y1=high
  # Keep native resolution: only translate destination principal point for crop.
  srcK=K.copy();srcK[:2,2]-=.5;dstK=srcK.copy();dstK[:2,2]-=low
  mx,my=cv2.initUndistortRectifyMap(srcK,np.array(dist),None,dstK,(int(x1-x0),int(y1-y0)),cv2.CV_32FC1)
  bx=max(0,int(np.floor(mx.min()))-2);by0=max(0,int(np.floor(my.min()))-2);ex=min(8192,int(np.ceil(mx.max()))+3);ey=min(4320,int(np.ceil(my.max()))+3)
  src=RAW/name/f'{name}.000408.tif';raw=cv2.imread(str(src),cv2.IMREAD_UNCHANGED);assert raw.shape==(4320,8192,3) and raw.dtype==np.uint16
  # Decode Log3G10 before interpolating. Preserve all signed RGB until grayscale luminance.
  patch=raw[by0:ey,bx:ex].astype(np.float32)/65535;del raw
  linear=np.expm1(patch*(np.log(10)/.224282))/155.975327-.01;del patch
  # Monotone positive camera-linear channel average for detector, no learned color fit.
  gray=np.maximum(linear.mean(2),0);del linear
  gray=cv2.remap(gray,mx-bx,my-by0,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);gray=4*gray;gray=gray/(1+gray);gray=np.where(gray<=.0031308,12.92*gray,1.055*gray**(1/2.4)-.055);gray=np.rint(np.clip(gray,0,1)*255).astype(np.uint8)
  sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.002,edgeThreshold=15);kp,desc=sift.detectAndCompute(gray,None);uv=np.array([q.pt for q in kp],float)+low+.5
  rows.append(row);uvs.append(uv);descs.append(desc);crops.append(gray);Image.fromarray(gray).save(OUT/(name+'_crop.png'));np.savez_compressed(OUT/(name+'_features.npz'),detected_uv=uv,descriptors=desc,crop_offset=low)
  sources.append(dict(camera=name,raw=str(src),raw_sha256=sha(src),bbox=[int(x0),int(y0),int(x1),int(y1)],features=len(kp)));save('progress.json',dict(stage='features',camera=name,features=len(kp)));print(name,gray.shape,len(kp),flush=True)
 # Match real detections; no projected/reconstructed UV substituted for observations.
 edges=[];pairlog=[];bf=cv2.BFMatcher()
 for i,j in itertools.combinations(range(len(rows)),2):
  aa=bf.knnMatch(descs[i],descs[j],k=2);bb=bf.knnMatch(descs[j],descs[i],k=2);back={m.queryIdx:m.trainIdx for m,n in bb if m.distance<.75*n.distance};matches=[m for m,n in aa if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx];a=np.array([m.queryIdx for m in matches],int);b=np.array([m.trainIdx for m in matches],int)
  err=sampson(fundamental(rows[i],rows[j]),uvs[i][a],uvs[j][b])if len(a)else np.empty(0);keep=err<1.5
  edges.extend([((i,int(x)),(j,int(y)))for x,y in zip(a[keep],b[keep])]);pairlog.append(dict(cameras=[NAMES[i],NAMES[j]],mutual_ratio=len(a),epipolar_pass=int(keep.sum())))
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for a,b in edges:
  pa,pb=find(a),find(b)
  if pa!=pb:parent[pb]=pa
 groups={}
 for x in parent:groups.setdefault(find(x),[]).append(x)
 points=[];tracks=[];reject={'fewer3':0,'duplicate_camera':0,'reprojection_cheirality_angle':0}
 for group in groups.values():
  if len(group)<3:reject['fewer3']+=1;continue
  if len({x[0]for x in group})!=len(group):reject['duplicate_camera']+=1;continue
  A=[]
  for c,k in group:
   P,*_=projection(rows[c]);u,v=uvs[c][k];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
  _,_,vh=np.linalg.svd(A);X=vh[-1,:3]/vh[-1,3];errors=[];depth=[]
  for c,k in group:
   P,*_=projection(rows[c]);q=P@np.r_[X,1];errors.append(float(np.linalg.norm(q[:2]/q[2]-uvs[c][k])));depth.append(q[2])
  origins=np.array([np.array(rows[c]['transform_matrix'])[:3,3]for c,k in group]);directions=X-origins;directions/=np.linalg.norm(directions,axis=1)[:,None];angle=np.degrees(np.arccos(np.clip(directions@directions.T,-1,1))).max()
  if not np.isfinite(X).all()or max(errors)>=1.5 or min(depth)<=0 or angle<1 or angle>100:reject['reprojection_cheirality_angle']+=1;continue
  points.append(X);tracks.append(dict(id=len(points)-1,xyz=X.tolist(),max_detected_uv_reprojection_rawpixels=max(errors),max_angle_degrees=float(angle),observations=[dict(camera=NAMES[c],camera_index=c,keypoint_index=k,detected_uv=uvs[c][k].tolist(),reprojection_error_rawpixels=e)for(c,k),e in zip(group,errors)]))
 xyz=np.array(points).reshape(-1,3);np.savez_compressed(OUT/'candidate_points.npz',xyz=xyz)
 save('tracks.json',dict(points=tracks,pairs=pairlog,rejections=reject,sources=sources,true_detected_uv=True,minimum_track_cameras=3,maximum_reprojection_rawpixels=1.5,pixel_convention='SIFT crop0-index pixel centers +cropoffset+.5 => COLMAPhalf-integercenters',scope='Old plane ONLY image ROI scope; no plane depth constraints or filtering',poses_unchanged=True,uses_eval_RGB=False,source_sha256={str(p):sha(p)for p in [Path(__file__),R/'data/transforms.json',RIG/'cameras.txt',RIG/'images.txt']},manual_floor_classification_pending=True))
 # Rank spatially near target for human object classification; preserve all points.
 order=np.argsort(np.linalg.norm(xyz[:,None,:]-target[None,:,:],axis=2).min(1))if len(xyz)else []
 chosen=list(order[:60]);save('contact_point_order.json',[int(i)for i in chosen])
 for page in range((len(chosen)+4)//5):
  sheet=Image.new('RGB',(900,5*285),(25,25,25));draw=ImageDraw.Draw(sheet)
  for rr,idx in enumerate(chosen[page*5:page*5+5]):
   t=tracks[idx];obs=t['observations'];selected=[obs[0]]
   while len(selected)<3:
    left=[o for o in obs if o not in selected];selected.append(max(left,key=lambda o:min(np.linalg.norm(np.array(rows[o['camera_index']]['transform_matrix'])[:3,3]-np.array(rows[z['camera_index']]['transform_matrix'])[:3,3])for z in selected)))
   for col,o in enumerate(selected):
    c=o['camera_index'];low=np.array(sources[c]['bbox'][:2]);x,y=np.array(o['detected_uv'])-low-.5;im=Image.fromarray(crops[c]).crop((int(x)-128,int(y)-128,int(x)+128,int(y)+128)).convert('RGB');d=ImageDraw.Draw(im);d.ellipse((123,123,133,133),outline='red',width=2);sheet.paste(im,(col*300,rr*285+25));draw.text((col*300+3,rr*285+3),f"p{idx} {o['camera']} err{o['reprojection_error_rawpixels']:.2f}px",fill='white')
  sheet.save(OUT/f'contact_{page+1:02d}.jpg',quality=94)
 save('complete.json',dict(candidate_points=len(points),contacts=len(chosen),pages=(len(chosen)+4)//5,manual_review_pending=True,tracks_sha256=sha(OUT/'tracks.json')));print('COMPLETE',len(points),reject,flush=True)
if __name__=='__main__':main()
