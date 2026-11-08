"""Current native semantic consensus; weak/occluded hair stays geometry UNKNOWN."""
from pathlib import Path
import json,itertools
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write

DREAD={'dreadlocks','long dreadlocks','blond dreadlocks','braids','braided hair'}

def qualify(root,frame,camera,specpath,candidates,policy):
 specpath=Path(specpath);candidates=Path(candidates);s=json.loads(specpath.read_text());r=json.loads((candidates/'receipt.json').read_text());source=Path(root)/'sources'/str(frame)/camera/'receipt.json';f=Path(root)/'frames'/f'{frame:06d}'
 if r['source_spec_sha256']!=sha(specpath)or r['source_receipt_sha256']!=sha(source):raise ValueError('Foreign current head NN source')
 assert s['frame']==frame and s['camera']==camera and r['checkpoint_sha256']==policy['checkpoint_sha256']
 box=s['crop_xyxy'];x0,y0,x1,y1=box
 with np.load(f/'native_masks'/(camera+'.npz'))as z:body=z['black'][y0:y1,x0:x1].copy()
 rows=[]
 for record in r['records']:
  prompt=record['prompt'];path=candidates/(prompt.replace(' ','_')+'.npz')
  if sha(path)!=record['archive_sha256']:raise ValueError('Current native semantic NN output changed')
  archive_sha=sha(path)
  with np.load(path)as z:
   assert np.array_equal(z['crop_xyxy'],box)
   masks=z['masks'];scores=z['scores']
   for index in np.flatnonzero(scores>=policy['weak_min_score']):
    index=int(index);score=scores[index];m=masks[index].copy();area=int(m.sum());overlap=int((m&body).sum());fraction=overlap/max(area,1)
    if area and float(score)>=policy['weak_min_score']and fraction>=policy['minimum_existing_actor_overlap_fraction']:
     rows.append(dict(prompt=prompt,index=index,score=float(score),area=area,existing_actor_overlap_fraction=fraction,archive=str(path),archive_sha256=archive_sha,mask=m))
 strong=[x for x in rows if x['score']>=policy['positive_min_score']];pairs=[]
 for a,b in itertools.combinations(strong,2):
  if a['prompt']==b['prompt']or not({a['prompt'],b['prompt']}&DREAD):continue
  intersection=int((a['mask']&b['mask']).sum());iou=intersection/max(int((a['mask']|b['mask']).sum()),1)
  if iou>=policy['minimum_semantic_pair_IoU']:pairs.append((min(a['score'],b['score']),iou,a,b))
 pairs.sort(key=lambda x:(x[0],x[1]),reverse=True);fg=np.zeros((y1-y0,x1-x0),bool);geometry=None;selected=[]
 if pairs:
  _,iou,a,b=pairs[0];fg=a['mask']&b['mask'];geometry=a['mask']|b['mask'];selected=[{k:v for k,v in x.items()if k!='mask'}for x in [a,b]];status='CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR'
 else:
  weak=sorted(rows,key=lambda x:(x['prompt']in DREAD,x['score']),reverse=True)
  if weak:geometry=weak[0]['mask'];selected=[{k:v for k,v in weak[0].items()if k!='mask'}];status='GEOMETRY_UNKNOWN_ONLY_WEAK_CURRENT_HAIR'
  else:geometry=np.ones_like(fg);status='GEOMETRY_UNKNOWN_ONLY_CURRENT_HEAD_ROI_OCCLUDED_OR_UNRESOLVED'
  iou=None
 # A current component reaching a tile border requests another native tile.
 # Image/FOV edges are recorded, never hallucinated outside the actual raster.
 edges=[]
 if rows:
  for name,touch,expandable in [('left',geometry[:,:2].any(),x0>0),('right',geometry[:,-2:].any(),x1<8192),('top',geometry[:2].any(),y0>0),('bottom',geometry[-2:].any(),y1<4320)]:
   if touch and expandable:edges.append(name)
 return dict(frame=frame,camera=camera,status=status,crop_xyxy=box,selected=selected,semantic_pair_IoU=iou,qualified_current_foreground_pixels=int(fg.sum()),source_spec_sha256=sha(specpath),source_receipt_sha256=sha(source),NN_receipt_sha256=sha(candidates/'receipt.json'),checkpoint_sha256=r['checkpoint_sha256'],native_actor_archive_sha256=sha(f/'native_masks'/(camera+'.npz')),geometry_UNKNOWN_radius_native=policy['geometry_UNKNOWN_radius_native'],required_native_overlap_edges=edges,no_previous_bitmap_as_foreground=True,previous_head_location_only_ROI_proposal=True),fg,geometry
