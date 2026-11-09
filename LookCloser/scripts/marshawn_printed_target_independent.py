#!/usr/bin/env python3
"""Independent raw ArUco extraction; no existing SfM image observations used."""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
ROOT=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local')
RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
OUT=Path('/fsx/tmp/marshawn_001660_hd/printed_target_calibration_INDEPENDENT')
def rig():
 from scipy.spatial.transform import Rotation
 cs={}
 for s in (RIG/'cameras.txt').read_text().splitlines():
  if not s or s.startswith('#'):continue
  q=s.split();p=np.array(q[4:],float);cs[int(q[0])]=(np.array([[p[0],0,p[2]-.5],[0,p[1],p[3]-.5],[0,0,1]]),p[4:])
 ims={};lines=(RIG/'images.txt').read_text().splitlines();i=0
 while i<len(lines):
  s=lines[i];i+=1
  if not s or s.startswith('#'):continue
  q=s.split();i+=1;v=np.array(q[1:8],float);R=Rotation.from_quat(v[[1,2,3,0]]).as_matrix();K,d=cs[int(q[8])];ims[q[9].split('/')[0]]=dict(K=K,d=d,R=R,t=v[4:])
 return ims
def extract(frame,cam):
 target=OUT/'detections_v2'/f'{cam}_{frame}.json'
 if target.exists():return json.loads(target.read_text())
 p=ROOT/cam/f'{cam}.{frame:06}.tif'
 if not p.exists():return dict(camera=cam,frame=frame,error='missing')
 a=cv2.imread(str(p),-1);g=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);lo,hi=np.percentile(g[::8,::8],[1,99]);g=np.uint8(np.clip((g.astype(float)-lo)*255/max(hi-lo,1),0,255)); del a
 params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX;params.cornerRefinementWinSize=5;params.adaptiveThreshWinSizeMax=101;params.adaptiveThreshWinSizeStep=10;params.errorCorrectionRate=.4
 detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),params)
 found={};duplicates=[]
 for variant,im in [('linear',g),('half',cv2.resize(g,None,fx=.5,fy=.5))]:
  corners,ids,_=detector.detectMarkers(im)
  if ids is None:continue
  for c,mid in zip(corners,ids.ravel()):
   mid=int(mid);c=c.reshape(4,2)
   if variant=='half':
    c=(c+.5)*2-.5
    cv2.cornerSubPix(g,c,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
   if mid in found:
    if np.max(np.linalg.norm(c-np.array(found[mid]['corners']),axis=1))>5:duplicates.append(mid)
    continue
   found[mid]=dict(id=mid,corners=c.tolist(),variant=variant)
 for mid in duplicates:found.pop(mid,None)
 row=dict(camera=cam,frame=frame,source=str(p),dictionary='DICT_6X6_1000 (nested subsets indistinguishable)',markers=list(found.values()),duplicate_ids=duplicates)
 target.parent.mkdir(exist_ok=True,parents=True);target.write_text(json.dumps(row));
 if frame==408 and found:
  preview=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
  cv2.aruco.drawDetectedMarkers(preview,[np.float32(z['corners'])[None] for z in found.values()],np.int32(list(found))[:,None]);cv2.imwrite(str(OUT/f'overlay_{cam}.jpg'),cv2.resize(preview,(2048,1080)))
 return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frames',nargs='+',type=int,default=[408,448,488]);p.add_argument('--workers',type=int,default=4);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 tasks=[(f,c) for f in a.frames for c in rig()]
 with ThreadPoolExecutor(a.workers) as pool:
  for r in pool.map(lambda x:extract(*x),tasks):print(r['camera'],r['frame'],len(r.get('markers',[])),flush=True)
