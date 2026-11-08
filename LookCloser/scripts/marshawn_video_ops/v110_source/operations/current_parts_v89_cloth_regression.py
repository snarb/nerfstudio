from pathlib import Path
import sys,json,numpy as np,cv2,itertools,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
O=V/'operations/NB1685_boot_prop_v81';C=O/'remote_results/candidates';spec=json.loads((O/'spec.json').read_text());receipt=json.loads((C/'receipt.json').read_text());x0,y0,x1,y1=spec['crop_xyxy'];f=V/'frames/001685';before=sha(f/'native_masks/NB.npz')
with np.load(f/'qualified_nonactor_inputs/NB_PRE78/mask.npz')as z:P=z['qualified_NONACTOR_current_RAW_PRE'][y0:y1,x0:x1]
assert P.sum()==78;rows=[]
for prompt in ['shoe','gray shoe','boot']:
 p=C/(prompt.replace(' ','_')+'.npz');assert sha(p)==next(x['archive_sha256']for x in receipt['records']if x['prompt']==prompt)
 with np.load(p)as z:
  assert np.array_equal(z['crop_xyxy'],spec['crop_xyxy'])
  for i in np.flatnonzero(z['scores']>=.7):rows.append(dict(prompt=prompt,index=int(i),score=float(z['scores'][i]),mask=z['masks'][i],archive=str(p),archive_sha256=sha(p)))
accepted=[]
for a,b in itertools.combinations(rows,2):
 if a['prompt']==b['prompt']:continue
 m=a['mask']&b['mask'];iou=m.sum()/max(1,(a['mask']|b['mask']).sum())
 if iou<.85:continue
 n,lab,stats,_=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
 for k in range(1,n):
  part=lab==k
  if part.sum()<32:continue
  assert not(part&P).any(),'Semantic pair wrongly accepted source-certified floor-cloth RAW_PRE78'
  accepted.append(dict(a={k:v for k,v in a.items()if k!='mask'},b={k:v for k,v in b.items()if k!='mask'},IoU=float(iou),component=k,pixels=int(part.sum()),certified_floor_prop_overlap=0))
assert accepted;assert before==sha(f/'native_masks/NB.npz');write(V/'operations/current_parts_v89_cloth_regression.json',dict(kind='ACTUAL_CURRENT_NATIVE_MIXED_NB_BOOT_CLOTH_SEMANTIC_NEGATIVE_REGRESSION',frame=1685,camera='NB',status='pass',current_source_receipt_sha256=sha(V/'sources/1685/NB/receipt.json'),NN_receipt_sha256=sha(C/'receipt.json'),spec_sha256=sha(O/'spec.json'),exact_certified_prop_negative_archive_sha256=sha(f/'qualified_nonactor_inputs/NB_PRE78/mask.npz'),ROOT_prop_photo_approval_proof_sha256=sha(f/'qualified_nonactor_inputs/NB_PRE78/proof.json'),negative_pixels=78,all_accepted_pair_components_avoid_certified_prop=True,records=accepted,no_source_or_model_inputs_changed=True,current_semantic_pair_not_whole_mixed_component=True,broader_unlabelled_cloth_NOT_claimed_manual_groundtruth=True,created=time.time()))
print('PASS',len(accepted),'current semantic components, zero overlap with78 exact photo-certified floor-cloth negatives')
