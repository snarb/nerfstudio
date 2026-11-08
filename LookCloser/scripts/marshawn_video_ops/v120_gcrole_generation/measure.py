from pathlib import Path
import json,os,sys,hashlib,time,concurrent.futures,multiprocessing
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import cv2,numpy as np
cv2.setNumThreads(1)
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations/semantic_actor_role_v120';sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from crew_guard_plan_v53 import candidates

def one(t):
 frame,cam,records=t;f=V/'frames'/f'{frame:06d}';p=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}'/'applications'/cam/('old_'+cam+'.npz');p=p if p.exists()else f/'native_masks'/(cam+'.npz')
 with np.load(p)as z:A=z['black']|z['grey'];H=z['qualified_actor_rgb']if'qualified_actor_rgb'in z.files else np.zeros_like(A)
 near=cv2.distanceTransform((~A).astype('uint8'),cv2.DIST_L2,5)<=32
 cp,meta=candidates(V,frame,cam)
 with np.load(cp)as z:masks=z['masks']
 ids=[x for x in meta['selected'].values()if x is not None];raw=masks[ids].any(0)
 bs=json.loads((V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'bindings.json').read_text())['records'];out=[];cache={};labels={}
 for i,r in records:
  q=r['NN_pair'];roi=r['ROI'];x0,y0,x1,y1=roi;key=tuple(roi)
  if key not in cache:
   row=next(x for x in bs if x['camera']==cam and x['ROI']==roi);allowed=np.zeros((1008,1008),bool)
   for pr in row['components']:
    inst=pr['instance']
    if inst not in labels:labels[inst]=cv2.connectedComponents(masks[inst].astype('uint8'),8)[1]
    allowed|=labels[inst][y0:y1,x0:x1]==pr['native_component']
   cache[key]=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64
  ms=[]
  for e in[q['a'],q['b']]:
   ap=Path(e['archive']);assert sha(ap)==e['archive_sha256']
   with np.load(ap)as z:ms.append(z['masks'][e['index']].copy())
  n,lab=cv2.connectedComponents((ms[0]&ms[1]).astype('uint8'),8);part=(lab==q['NN_intersection_component'])&cache[key];R=part&~A[y0:y1,x0:x1];new=part&~H[y0:y1,x0:x1]
  assert int(R.sum())==q['new_actor_union_pixels']
  def frac(m,scope):return float((m&scope).sum()/max(int(scope.sum()),1))
  out.append(dict(frame=frame,camera=cam,index=i,family=q['family'],ROI=roi,part_pixels=int(part.sum()),R_pixels=int(R.sum()),R_near32_fraction=frac(near[y0:y1,x0:x1],R),R_RAW_fraction=frac(raw[y0:y1,x0:x1],R),part_RAW_fraction=frac(raw[y0:y1,x0:x1],part),part_existing_A_fraction=frac(A[y0:y1,x0:x1],part),old_new_near32=q['new_current_actor32native_fraction'],pre_native=str(p),pre_native_sha256=sha(p),current_candidates_sha256=sha(cp),selected=meta['selected'],part_sha256=hashlib.sha256(part.tobytes()).hexdigest()))
 return out
if __name__=='__main__':
 ts=[]
 for frame in[1690,1695,1700,1705]:
  rows=json.loads((V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json').read_text())['records'];cams=sorted(set(x['camera']for x in rows))
  ts +=[(frame,cam,[(i,x)for i,x in enumerate(rows)if x['camera']==cam])for cam in cams]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:rows=sum(list(pool.map(one,ts)),[])
 write(O/'current_role_metrics.json',dict(kind='IMMUTABLE_PREAPPLICATION_CURRENT_ACTOR_ROLE_METRICS',records=rows,created=time.time()))
 print('DONE',len(rows),flush=True)
 for r in rows:
  if r['R_pixels'] and(r['R_near32_fraction']<.8 or r['R_RAW_fraction']<.1):print(json.dumps(r),flush=True)
