"""Current-frame same-fighter LOWER-component qualification; no old pixels copied.

Requires trusted earlier actor-fragment identity, >=95% previous/current SAME
actor upper-anchor agreement, lower-body topology with explicit other-actor
occlusion context, and no independent person/crew conflict. UNKNOWN is retained.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from prepare_marshawn_tuning import sha,write
from crew_guard_plan import small,prior_actors,candidates

def qualify(root,frame,camera,instance,native_override=None):
 f=root/'frames'/f'{frame:06d}';path,meta=candidates(root,frame,camera);native=Path(native_override)if native_override else f/'native_masks'/f'{camera}.npz';record=json.loads((f/'native_masks'/f'{camera}.json').read_text())
 with np.load(native)as z:current={k:z[k].copy()for k in ['black','grey','source_ineligible','unknown']}
 with np.load(path)as z:masks=z['masks'];mask=masks[instance]
 s=small(mask);yy,xx=np.where(s);anchor=s&(np.indices(s.shape)[0]<=np.percentile(yy,20));prior,prior_sha=prior_actors(root,frame,camera);current_small={a:small(current[a])for a in ['black','grey']};anchors={}
 for actor in ['black','grey']:
  allow=cv2.dilate(prior[actor].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0;anchors[actor]=dict(previous=float((anchor&allow).sum()/max(anchor.sum(),1)),current=float((anchor&current_small[actor]).sum()/max(anchor.sum(),1)))
 actor=next((a for a,q in anchors.items()if q['previous']>=.95 and q['current']>=.95),None)
 if actor is None:return dict(frame=frame,camera=camera,instance=instance,actor=None,anchor_coarse_pixels=int(anchor.sum()),anchor_agreement=anchors,status='UNKNOWN_ACTOR_IDENTITY',components=[],current_candidate_archive_sha256=sha(path),native_input_sha256=sha(native))
 previous=f.parent/f'{frame-5:06d}'/'crew_guard'/f'{camera}_people.json';roles=json.loads(previous.read_text())['people']if previous.exists()else[];prevpath,_=candidates(root,frame-5,camera)
 with np.load(prevpath)as z:ps=[small(m)for m in z['masks']]
 ious=np.asarray([(s&m).sum()/max((s|m).sum(),1)for m in ps]);k=int(ious.argmax());old=next((p for p in roles if p['instance']==k),None);prior_actor_role=bool(old and ('actor_fragment'in old['status']or old['status'].startswith('fighter_'))and ious[k]>=.15)
 base=dict(frame=frame,camera=camera,instance=instance,actor=actor,anchor_coarse_pixels=int(anchor.sum()),anchor_agreement=anchors,previous_person_instance=k,previous_person_IoU=float(ious[k]),previous_person_role=None if old is None else old['status'],prior_actor_role=prior_actor_role,current_candidate_archive_sha256=sha(path),prior_actor_source_sha256=prior_sha,native_input_sha256=sha(native))
 if actor is None or not prior_actor_role:return dict(base,status='UNKNOWN_ACTOR_IDENTITY',components=[])
 # Independently tracked CREW/UNKNOWN person evidence has priority over anchors.
 current_roles=f/'crew_guard'/f'{camera}_people.json';people=json.loads(current_roles.read_text())['people']if current_roles.exists()else[];crew_indices=[p['instance']for p in people if p['instance']!=instance and ('verified_other_person'in p['status']or'verified_crew'in p['status']or'NEEDS_CURRENT_NATIVE_CREW'in p['status']or p['status'].startswith('UNKNOWN'))]
 other='grey'if actor=='black'else'black';count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8);components=[];minimum_y=int(np.where(mask)[0].min());height=int(np.where(mask)[0].max()-minimum_y+1)
 for component in range(1,count):
  m=labels==component;new=m&~current['black']&~current['grey']&~current['source_ineligible'];new_pixels=int(new.sum())
  if new_pixels<4096:continue
  y,x=np.where(m);lower_fraction=float((y.min()-minimum_y)/height);cm=small(m);novel_conflict=[]
  for ck in crew_indices:
   conflict=int((cm&small(masks[ck])).sum())
   if conflict>8 and conflict/max(int(cm.sum()),1)>.01:novel_conflict.append(dict(person_instance=ck,coarse_pixels=conflict))
  bridge=s|current_small[actor];_,bl=cv2.connectedComponents(bridge.astype('uint8'),8);head_labels=set(bl[anchor]);leg_labels=set(bl[cm]);direct=bool((head_labels&leg_labels)-{0})
  bridge_other=bridge|current_small[other];_,ol=cv2.connectedComponents(bridge_other.astype('uint8'),8);through_other=bool((set(ol[anchor])&set(ol[cm]))-{0})
  veto_pixels=int((m&current['source_ineligible']).sum());qualified=lower_fraction>=.55 and not novel_conflict and(direct or through_other)and veto_pixels/max(int(m.sum()),1)<.01
  components.append(dict(component=component,pixels=int(m.sum()),new_visible_native_pixels=new_pixels,lower_body_fraction=lower_fraction,direct_same_actor_connection=direct,connection_only_through_current_other_actor=through_other and not direct,independent_CREW_or_UNKNOWN_conflicts=novel_conflict,crew_veto_pixels=veto_pixels,qualified=qualified,status='QUALIFIED_CURRENT_SAME_FIGHTER_LOWER_FRAGMENT'if qualified else'UNKNOWN_COMPONENT'))
 return dict(base,status='QUALIFIED_CURRENT_COMPONENT_AVAILABLE'if any(c['qualified']for c in components)else'UNKNOWN_NO_QUALIFIED_LOWER_COMPONENT',components=components,policy='No background-gap interpolation. All final added pixels must come from the same current NN component; preserve all existing actors/gray/veto/UNKNOWN. Independently tracked crew conflict cannot be waived.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--camera',required=True);p.add_argument('--instance',type=int,required=True);p.add_argument('--native-override',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=qualify(a.root,a.frame,a.camera,a.instance,a.native_override);write(a.output,r);print(json.dumps(r))
