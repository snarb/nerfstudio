"""Current-frame person guard and conservative add-only lower-fragment recovery.

Known crew/UNKNOWN conflicts never receive an anchor exemption. If a source-
qualified current component cannot pass the released regression-tested rule,
publication blocks rather than deleting pixels or inventing actor ownership.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image,ImageDraw
from prepare_marshawn_tuning import sha,write
from crew_guard_plan_v53 import run,candidates
from crew_reference_advance import advance
from actor_fragment_qualify_v5 import qualify
from actor_fragment_apply import apply

def main(a):
 root=a.root;f=root/'frames'/f'{a.frame:06d}';cameras=json.loads((root/'source_inventory.json').read_text())['cameras'];policy=json.loads(a.policy.read_text())
 if policy.get('authorized') is not True or policy.get('kind')!='CURRENT_SAME_ACTOR_LOWER_FRAGMENT_RULE_V5':raise ValueError('No released automatic actor-fragment policy')
 regression=root/'operations'/policy['regression_suite']
 if sha(regression)!=policy['regression_suite_sha256']:raise ValueError('Released real regression evidence changed')
 proof=json.loads(regression.read_text())
 if proof['status']!='pass'or proof['qualifier_sha256']!=sha(root/'operations/actor_fragment_qualify_v5.py')or proof['guard_plan_sha256']!=sha(root/'operations/crew_guard_plan_v53.py'):raise ValueError('Current guard differs from qualified real regressions')
 advance(root,a.frame);run(root,a.frame,cameras);plan=json.loads((f/'crew_guard/plan_receipt.json').read_text())
 for row in plan['reports']:
  if row['status']=='pass':continue
  camera=row['camera'];report=json.loads((f/'crew_guard'/(camera+'_people.json')).read_text());out=f/'crew_guard/fragment_qualification';out.mkdir(exist_ok=True)
  for instance in row['unknown']:
   person=next(p for p in report['people']if p['instance']==instance)
   if 'CREW'in person['status']or person['status'].startswith('verified_crew'):continue
   q=qualify(root,a.frame,camera,instance);qp=out/f'{camera}_{instance}_qualification.json';write(qp,q)
   if q.get('actor') is None:continue
   path,_=candidates(root,a.frame,camera);native=f/'native_masks'/(camera+'.npz');receipt=root/'sources'/str(a.frame)/camera/'receipt.json';source=json.loads(receipt.read_text())
   # Full source context is always persisted together with component boxes.
   context=Image.open(source['raw_native_image']).convert('RGB');draw=ImageDraw.Draw(context)
   import cv2
   with np.load(path)as z:mask=z['masks'][instance]
   _,labels,_,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
   for c in q['components']:
    if c['qualified']:
     yy,xx=np.where(labels==c['component']);draw.rectangle([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],outline='yellow',width=12)
   context.thumbnail((2048,1080));contextpath=out/f'{camera}_{instance}_full_source_context.jpg';context.save(contextpath,quality=96)
   for c in q['components']:
    if not c['qualified']:continue
    spec=dict(campaign=str(root),frame=a.frame,camera=camera,actor=q['actor'],person_instance=instance,component=c['component'],component_pixels=c['pixels'],candidate_archive=str(path),candidate_archive_sha256=sha(path),source_receipt_sha256=sha(receipt),input_native_sha256=sha(native),approved=True,source_review_evidence=dict(automatic_qualified_current_component=True,policy=str(a.policy),policy_sha256=sha(a.policy),qualification=str(qp),qualification_sha256=sha(qp),source_context=str(contextpath),source_context_sha256=sha(contextpath),human_review_claim=False),head_anchor_evidence=q['anchor_agreement'],occlusion_context=c)
    specpath=out/f'{camera}_{instance}_component{c["component"]}_apply.json';write(specpath,spec);apply(specpath)
 if plan['status']!='pass':run(root,a.frame,cameras)
 final=json.loads((f/'crew_guard/plan_receipt.json').read_text())
 if final['status']!='pass':raise RuntimeError('UNKNOWN current person/component remains; native_semantic_ready withheld')
 write(f/'crew_guard/autonomous_guard_ready.json',dict(frame=a.frame,status='pass',policy_sha256=sha(a.policy),guard_sha256=sha(f/'crew_guard/plan_receipt.json'),no_prior_bitmap_final_pixels=True,created=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--policy',type=Path,required=True);main(p.parse_args())
