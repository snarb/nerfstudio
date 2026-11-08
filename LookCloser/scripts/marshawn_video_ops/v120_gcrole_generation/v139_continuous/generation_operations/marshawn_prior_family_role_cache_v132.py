"""Source-bound prior FAMILY role guidance; never emits current foreground."""
from pathlib import Path
import sys,json,time,itertools,fcntl
import cv2,numpy as np
from PIL import Image
BASE=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(BASE/'operations'),str(BASE/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_current_parts_v105 import FAMILIES
import marshawn_union_tracking_proposals_v1 as tracker

def support(root,frame,camera,family):
 root=Path(root);prior=frame-5;out=root/'operations/prior_family_ROLE_cache_v132'/f'frame{frame:06d}';out.mkdir(parents=True,exist_ok=True);key=camera+'_'+family;archive=out/(key+'.npz');receipt=out/(key+'.json');lock=open(out/(key+'.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX)
 extra_registry=root/'operations/prior_family_ROLE_extra_inputs_v132.json'
 previous=root/'frames'/f'{prior:06d}';native=previous/'native_masks'/(camera+'.npz');marker=previous/'native_semantic_ready.json';m=json.loads(marker.read_text());assert m['native_masks'][camera]['archive_sha256']==sha(native)
 bindings=root/'operations/current_native_parts_v105'/f'frame{prior:06d}'/'bindings.json';binding=dict(frame=frame,prior=prior,camera=camera,family=family,prior_source_marker_sha256=sha(marker),prior_actor_native_sha256=sha(native),prior_actor_metadata_sha256=sha(native.with_suffix('.json')),prior_semantic_NN_bindings_sha256=sha(bindings)if bindings.exists()else None,current_RAW_receipt_sha256=sha(root/'sources'/str(frame)/camera/'receipt.json'),previous_RAW_receipt_sha256=sha(root/'sources'/str(prior)/camera/'receipt.json'),implementation_sha256=sha(__file__),tracker_sha256=sha(tracker.__file__),extra_registry_sha256=sha(extra_registry)if extra_registry.exists()else None)
 if receipt.exists()and archive.exists():
  r=json.loads(receipt.read_text())
  if r['binding']==binding and r['archive_sha256']==sha(archive):
   with np.load(archive)as z:return z['flowed_same_family_role_native'].copy(),r
 with np.load(native)as z:A=(z['black']|z['grey'])&~z['source_ineligible']
 role=np.zeros_like(A);proofs=[];rows=json.loads(bindings.read_text())['records']if bindings.exists()else[]
 if extra_registry.exists():
  for e in json.loads(extra_registry.read_text())['records']:
   if e['prior_frame']==prior and e['camera']==camera and e['family']==family:
    assert sha(e['spec'])==e['spec_sha256'];rows.append(dict(camera=camera,spec=e['spec'],candidates=e['candidates']))
 for row in rows:
  if row['camera']!=camera:continue
  sp=Path(row['spec']);spec=json.loads(sp.read_text());C=Path(row['candidates']);nr=json.loads((C/'receipt.json').read_text());assert spec['frame']==prior and spec['camera']==camera and spec['raw_native_image_sha256']==json.loads((root/'sources'/str(prior)/camera/'receipt.json').read_text())['raw_native_image_sha256']and nr['source_spec_sha256']==sha(sp)and nr['source_receipt_sha256']==binding['previous_RAW_receipt_sha256'];x0,y0,x1,y1=spec['crop_xyxy'];candidates=[]
  for prompt in FAMILIES[family]:
   ap=C/(prompt.replace(' ','_')+'.npz');entry=next(x for x in nr['records']if x['prompt']==prompt);assert sha(ap)==entry['archive_sha256']
   with np.load(ap)as z:
    for i in np.flatnonzero(z['scores']>=.7):candidates.append(dict(prompt=prompt,index=int(i),score=float(z['scores'][i]),mask=z['masks'][i].copy(),archive=str(ap),archive_sha256=sha(ap)))
  for a,b in itertools.combinations(candidates,2):
   if a['prompt']==b['prompt']:continue
   h=a['mask']&b['mask'];iou=int(h.sum())/max(int((a['mask']|b['mask']).sum()),1)
   if iou<.85:continue
   n,lab=cv2.connectedComponents(h.astype('uint8'),8)
   for k in range(1,n):
    c=lab==k;count=int(c.sum());overlap=int((c&A[y0:y1,x0:x1]).sum());fraction=overlap/max(count,1)
    if count<32 or fraction<.8:continue
    role[y0:y1,x0:x1]|=c&A[y0:y1,x0:x1];proofs.append(dict(spec=str(sp),spec_sha256=sha(sp),NN_receipt=str(C/'receipt.json'),NN_receipt_sha256=sha(C/'receipt.json'),a={k:v for k,v in a.items()if k!='mask'},b={k:v for k,v in b.items()if k!='mask'},IoU=iou,component=k,prior_actor_fraction=fraction,pixels=count))
 images=[]
 for f in[prior,frame]:
  source=json.loads((root/'sources'/str(f)/camera/'receipt.json').read_text());assert sha(Path(source['raw_native_image']))==source['raw_native_image_sha256'],'Actual RAW-native PNG changed before ROLE flow';images.append(np.array(Image.open(source['raw_native_image']).convert('L').resize(tracker.SIZE,Image.Resampling.BILINEAR)))
 coarse=tracker.flow_channels(images[0],images[1],{'complete_union':tracker.pool(role)})['flowed_complete_union'];flowed=cv2.resize(coarse.astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0;tmp=archive.with_suffix('.npz.tmp')
 with tmp.open('wb')as f:np.savez_compressed(f,prior_current_family_role=role,flowed_same_family_role_native=flowed)
 tmp.replace(archive);r=dict(kind='QUALIFIED_PRIOR_SAME_FAMILY_ROLE_GUIDANCE_V132',binding=binding,archive_sha256=sha(archive),prior_role_pixels=int(role.sum()),prior_role_array_sha256=array_sha(role),current_flowed_role_array_sha256=array_sha(flowed),proofs=proofs,prior_ROLE_only_not_current_FG=True,created=time.time());write(receipt,r);return flowed,r
