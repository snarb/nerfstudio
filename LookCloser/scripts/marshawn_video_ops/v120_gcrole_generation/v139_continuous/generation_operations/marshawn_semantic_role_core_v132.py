"""Same current NN qualification, followed by explicit target-human role gate."""
from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np
BASE_ROOT=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');V=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(BASE_ROOT/'operations'),str(BASE_ROOT/'code/LookCloser/scripts'),str(BASE_ROOT/'operations/semantic_actor_role_v120')]
import marshawn_current_parts_v105 as original
from marshawn_current_part_role_policy_v128 import decide_current_role
from marshawn_prior_family_role_cache_v132 import support
from marshawn_union_resolution_v5 import array_sha
from prepare_marshawn_tuning import sha,write
from crew_guard_plan_v53 import candidates
CREW_CACHE=original.CREW_CACHE
ROLE_CACHE={}

def near32_with_halo(Aall,box):
 x0,y0,x1,y1=box;l=max(0,x0-32);t=max(0,y0-32);r=min(Aall.shape[1],x1+32);b=min(Aall.shape[0],y1+32)
 halo=cv2.distanceTransform((~Aall[t:b,l:r]).astype('uint8'),cv2.DIST_L2,5)<=32
 return halo[y0-t:y1-t,x0-l:x1-l]

def qualify_tile(frame,row):
 original.V=V;cam=row['camera'];native=V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz');before=sha(native)
 with np.load(native)as z:Aall=z['black']|z['grey']
 x0,y0,x1,y1=row['ROI'];A=Aall[y0:y1,x0:x1];near=near32_with_halo(Aall,row['ROI']);cp,meta=candidates(V,frame,cam)
 with np.load(cp)as z:raw=z['masks'][[x for x in meta['selected'].values()if x is not None]].any(0)[y0:y1,x0:x1]
 records=original.qualify_tile(frame,row);output=[]
 for index,q in enumerate(records):
  part=q['mask'];R=part&~A;count=int(R.sum());aa=int((part&A).sum());rr=int((part&raw).sum());nf=int((R&near).sum())/max(count,1);d=decide_current_role(aa,rr,nf,0,count);role=None;supported=0
  if not d['accepted']:
   key=(frame,cam,q['family'])
   if key not in ROLE_CACHE:ROLE_CACHE[key]=support(V,frame,cam,q['family'])
   flow,role=ROLE_CACHE[key];supported=int((R&flow[y0:y1,x0:x1]).sum());d=decide_current_role(aa,rr,nf,supported,count)
  bootstrap=None
  if not d['accepted']and frame==1705 and cam=='FB':
   for bp in(BASE_ROOT/'operations/semantic_actor_role_v120/policy_v128').glob('FB1705_*_positive_bootstrap.json'):
    b=json.loads(bp.read_text());full=np.zeros((4320,8192),bool);full[y0:y1,x0:x1]=R
    if b['scope_array_sha256']==array_sha(full)and b['source_RGB_receipt_sha256']==sha(V/'sources/1705/FB/receipt.json')and b['NN_pair']['a']['archive_sha256']==q['a']['archive_sha256']and b['NN_pair']['b']['archive_sha256']==q['b']['archive_sha256']:
     assert sha(b['scope_archive'])==b['scope_archive_sha256']and sha(b['source_photo_review'])==b['source_photo_review_sha256'];bootstrap=dict(path=str(bp),sha256=sha(bp));d=decide_current_role(aa,rr,nf,supported,count,True);break
  q['target_human_role']=dict(decision=d,current_preapplication_native_sha256=before,current_candidate_archive_sha256=sha(cp),current_part_A_pixels=aa,current_part_RAW_pixels=rr,current_residual_near32_fraction=nf,current_residual_pixels=count,prior_same_family_supported_pixels=supported,rule_helper_sha256=sha(BASE_ROOT/'operations/semantic_actor_role_v120/marshawn_current_part_role_policy_v128.py'),prior_role_receipt=role,finite_exact_photo_bootstrap=bootstrap)
  if not d['accepted']:
   O=V/'operations/current_role_UNKNOWN_v132'/f'frame{frame:06d}'/cam;O.mkdir(parents=True,exist_ok=True);key=hashlib.sha256((str(row['ROI'])+q['family']+str(q['a'])+str(q['b'])+str(index)).encode()).hexdigest()[:20];p=O/(key+'.npz')
   with p.with_suffix('.npz.tmp').open('wb')as f:np.savez_compressed(f,role_unknown=R,crop_xyxy=np.asarray(row['ROI']))
   p.with_suffix('.npz.tmp').replace(p);q['ROLE_UNKNOWN_only']=True;q['role_unknown_current_NN_archive']=str(p);q['role_unknown_current_NN_archive_sha256']=sha(p);q['originally_proposed_actor_union_pixels']=count;q['new_actor_union_pixels']=0;q['new_pixels']=0;q['mask']=np.zeros_like(part)
  output.append(q)
 assert sha(native)==before,'Semantic role gate requires immutable preapplication CURRENT A';return output
