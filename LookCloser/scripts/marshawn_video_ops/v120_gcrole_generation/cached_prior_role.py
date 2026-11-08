from pathlib import Path
import os,sys,json,itertools,time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import cv2,numpy as np
from PIL import Image
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
import marshawn_union_tracking_proposals_v1 as tracker
from marshawn_current_parts_v105 import FAMILIES

def assess(frame,cam,family):
 previous=frame-5;p=G/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz');p=p if p.exists()else V/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz')
 with np.load(p)as z:A=(z['black']|z['grey'])&~z['source_ineligible']
 binding=V/'operations/current_native_parts_v105'/f'frame{previous:06d}'/'bindings.json';rows=[r for r in json.loads(binding.read_text())['records']if r['camera']==cam];role=np.zeros_like(A);proof=[]
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
 current=G/'frames'/f'{frame:06d}'/'generation_corrections/GC_visible_CREW_hand/retracted_wrong_human_NN_component.npz'
 with np.load(current)as z:R=z['wrong_CREW_actor_part']
 return dict(frame=frame,camera=cam,family=family,prior_source_native=str(p),prior_source_native_sha256=sha(p),prior_cached_NN_bindings_sha256=sha(binding),prior_NN_ROIs=len(rows),qualified_prior_ROLE_pixels=int(role.sum()),new_wrong_CREW_pixels=int(R.sum()),wrong_CREW_supported_by_FLOW_SAME_ROLE_pixels=int((R&full).sum()),wrong_CREW_same_ROLE_fraction=int((R&full).sum())/max(int(R.sum()),1),records=proof,no_NN_or_source_modified=True,private_not_policy_authority=True)
if __name__=='__main__':
 cv2.setNumThreads(1);r=assess(1695,'GC','hand');write(V/'operations/semantic_actor_role_v120/GC1695_cached_prior_HAND_role.json',r);print({k:v for k,v in r.items()if k!='records'})
