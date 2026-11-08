from pathlib import Path
import os,sys,json,itertools,time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import cv2,numpy as np
from PIL import Image
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
import marshawn_union_tracking_proposals_v1 as tracker
from marshawn_current_parts_v105 import FAMILIES

def assess(frame,cam,family,record_index):
 previous=frame-5;p=G/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz');p=p if p.exists()else V/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz')
 with np.load(p)as z:A=(z['black']|z['grey'])&~z['source_ineligible']
 binding=V/'operations/current_native_parts_v105'/f'frame{previous:06d}'/'bindings.json';rows=[r for r in json.loads(binding.read_text())['records']if r['camera']==cam]if binding.exists()else[];role=np.zeros_like(A);proof=[]
 for row in rows:
  s=json.loads(Path(row['spec']).read_text());C=Path(row['candidates']);receipt=json.loads((C/'receipt.json').read_text());assert receipt['source_spec_sha256']==sha(row['spec']);x0,y0,x1,y1=s['crop_xyxy'];ms=[]
  for prompt in FAMILIES[family]:
   ap=C/(prompt.replace(' ','_')+'.npz');r=next(z for z in receipt['records']if z['prompt']==prompt);assert sha(ap)==r['archive_sha256']
   with np.load(ap)as z:
    for i in np.flatnonzero(z['scores']>=.7):ms.append(dict(prompt=prompt,index=int(i),mask=z['masks'][i].copy(),score=float(z['scores'][i]),archive=str(ap),sha256=sha(ap)))
  for a,b in itertools.combinations(ms,2):
   if a['prompt']==b['prompt']:continue
   part=a['mask']&b['mask'];iou=int(part.sum())/max(int((a['mask']|b['mask']).sum()),1)
   if iou<.85:continue
   n,lab=cv2.connectedComponents(part.astype('uint8'),8)
   for k in range(1,n):
    c=lab==k;area=int(c.sum());over=int((c&A[y0:y1,x0:x1]).sum());fraction=over/max(area,1);accepted=area>=32 and fraction>=.8
    if accepted:role[y0:y1,x0:x1]|=c&A[y0:y1,x0:x1]
    proof.append(dict(ROI=s['crop_xyxy'],a={z:w for z,w in a.items()if z!='mask'},b={z:w for z,w in b.items()if z!='mask'},IoU=iou,component=k,pixels=area,prior_qualified_actor_overlap=over,prior_actor_fraction=fraction,role_support_accepted=accepted))
 images=[]
 for f in[previous,frame]:
  r=json.loads((V/'sources'/str(f)/cam/'receipt.json').read_text());images.append(np.array(Image.open(r['raw_native_image']).convert('L').resize(tracker.SIZE,Image.Resampling.BILINEAR)))
 flowed=tracker.flow_channels(images[0],images[1],{'complete_union':tracker.pool(role)})['flowed_complete_union'];full=cv2.resize(flowed.astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0
 qpath=V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json';row=json.loads(qpath.read_text())['records'][record_index];assert row['camera']==cam and row['NN_pair']['family']==family;q=row['NN_pair'];x0,y0,x1,y1=row['ROI'];ms=[]
 for e in[q['a'],q['b']]:
  assert sha(e['archive'])==e['archive_sha256']
  with np.load(e['archive'])as z:ms.append(z['masks'][e['index']].copy())
 _,lab=cv2.connectedComponents((ms[0]&ms[1]).astype('uint8'),8);part=lab==q['NN_intersection_component']
 from crew_guard_plan_v53 import candidates
 cp,meta=candidates(V,frame,cam)
 with np.load(cp)as z:masks=z['masks']
 current_bind=json.loads((V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'bindings.json').read_text())['records'];br=next(x for x in current_bind if x['camera']==cam and x['ROI']==row['ROI']);allowed=np.zeros_like(part)
 for pr in br['components']:
  _,lab=cv2.connectedComponents(masks[pr['instance']].astype('uint8'),8);allowed|=lab[y0:y1,x0:x1]==pr['native_component']
 allowed=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64;part&=allowed
 pre=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}'/'applications'/cam/('old_'+cam+'.npz');pre=pre if pre.exists()else V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz')
 with np.load(pre)as z:current_A=(z['black']|z['grey'])[y0:y1,x0:x1]
 R=np.zeros((4320,8192),bool);R[y0:y1,x0:x1]=part&~current_A;assert int(R.sum())==q['new_actor_union_pixels']
 return dict(frame=frame,camera=cam,family=family,current_semantic_record_index=record_index,prior_source_native=str(p),prior_source_native_sha256=sha(p),prior_cached_NN_bindings_sha256=sha(binding)if binding.exists()else None,prior_NN_ROIs=len(rows),qualified_prior_ROLE_pixels=int(role.sum()),new_actor_residual_pixels=int(R.sum()),new_residual_supported_by_FLOW_SAME_ROLE_pixels=int((R&full).sum()),new_residual_same_ROLE_fraction=int((R&full).sum())/max(int(R.sum()),1),records=proof,no_NN_or_source_modified=True,private_not_policy_authority=True)
if __name__=='__main__':
 cv2.setNumThreads(1);cases=[(1695,'GC','hand',30),(1695,'MD','foot',21),(1695,'MD','foot',25),(1700,'GC','hand',28),(1700,'ED','hand',151),(1700,'NC','foot',34),(1705,'GC','hand',51),(1705,'FB','hand',29),(1705,'FB','hand',34),(1705,'GD','foot',117),(1705,'DE','foot',2)];out=[]
 for c in cases:
  q=assess(*c);out.append(q);print(c,q['prior_NN_ROIs'],q['qualified_prior_ROLE_pixels'],q['new_residual_same_ROLE_fraction'],flush=True)
 write(V/'operations/semantic_actor_role_v120/cached_prior_ROLE_goldens_v121.json',dict(kind='PRIVATE_CACHED_SAME_FAMILY_ROLE_ASSESSMENT',records=out,private_not_policy_authority=True,no_NN_or_source_modified=True,created=time.time()))
