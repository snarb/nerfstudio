"""Current native semantic parts anchored by complete 2D union; never transfer FG."""
from pathlib import Path
import argparse,json,sys,time,subprocess,os,shlex,uuid,itertools,concurrent.futures
import cv2,numpy as np
import hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
import marshawn_union_tracking_proposals_v1 as tracker
FAMILIES={'pants':['pants','trousers','dark pants'],'foot':['shoe','gray shoe','black shoe','boot'],'hand':['hand','hands'],'forearm':['arm','forearm'],'head':['head','human head','face'],'gray_actor':['person in gray clothes','man in gray clothing','gray jacket','gray shirt','gray clothes']}
PROMPTS=sum(FAMILIES.values(),[])+['folded cloth','black cloth','fabric on floor','knee pad','black knee pad']
def load(p):return json.loads(Path(p).read_text())
def plans(frame,cameras):
 O=V/'operations/current_native_parts_v105'/f'frame{frame:06d}';O.mkdir(parents=True,exist_ok=True)
 if(O/'ROI_proposals.json').exists():return O,load(O/'ROI_proposals.json')['records']
 guide=V/'operations/union_tracking_v1'/f'frame{frame:06d}';records=[]
 def proposal(cam):return tracker.propose(V,frame,cam,guide)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:reports=list(pool.map(proposal,cameras))
 for q in reports:
  for p in q['records']:
   area=p['native_pixels'];fraction=p['current_actor_pixels_native']/max(area,1)
   if area<256 or fraction>=.8:continue
   if not p['direct_flow_anchor_present']and p['current_actor_pixels_native']<128:continue
   x0,y0,x1,y1=p['ROI_native_xyxy']
   def axis(a,b,limit):
    if b-a<=1008:return[max(0,min(limit-1008,(a+b)//2-504))]
    return sorted(set(list(range(a,b-1008+1,672))+[min(limit-1008,b-1008)]))
   xs=axis(x0,x1,8192);ys=axis(y0,y1,4320)
   for x in sorted(set(xs)):
    for y in sorted(set(ys)):
     x=max(0,min(8192-1008,x));y=max(0,min(4320-1008,y));roi=[x,y,x+1008,y+1008]
     found=next((r for r in records if r['camera']==q['camera']and r['ROI']==roi),None)
     if found:found['components'].append(p);continue
     records.append(dict(camera=q['camera'],ROI=roi,components=[p],proposal=str(guide/(q['camera']+'_proposals.json')),proposal_sha256=sha(guide/(q['camera']+'_proposals.json')),guidance_archive=str(guide/(q['camera']+'_channels.npz'))))
 # Highest genuine missing-union evidence first. Additional unresolved regions
 # stay UNKNOWN; the proposal cap never creates foreground or free labels.
 records.sort(key=lambda r:sum(x['missing_current_union_flow_pixels_coarse']for x in r['components']),reverse=True);records=records[:64];write(O/'ROI_proposals.json',dict(frame=frame,records=records,all_source_proposals=len(records),no_FG_pixels_emitted=True));return O,records

def launch(frame,O,records):
 if not records:return
 recipe=load(V/'operations/sam3_assist/v5/full_release.json');reuse={}
 for p in [(V/'operations/current_native_parts_v96'/f'frame{frame:06d}'/'bindings.json')]:
  if p.exists():
   for row in load(p)['records']:
    cs=load(row['spec']);cp=Path(row['candidates']);rp=cp/'receipt.json'
    if rp.exists()and cs['frame']==frame and cs['prompts']==PROMPTS:
     cr=load(rp)
     if cr['source_spec_sha256']==sha(Path(row['spec']))and cr['source_receipt_sha256']==sha(V/'sources'/str(frame)/row['camera']/'receipt.json')and all(sha(cp/(x['prompt'].replace(' ','_')+'.npz'))==x['archive_sha256']for x in cr['records']):reuse[(row['camera'],tuple(cs['crop_xyxy']))]=row
 SV=Path(recipe['remote_root']);SO=SV/'operations/current_native_parts_v105'/f'frame{frame:06d}';jobs=[];bindings=[]
 for i,row in enumerate(records):
  cam=row['camera'];cached=reuse.get((cam,tuple(row['ROI'])))
  if cached:
   bindings.append(dict(row,key=cached['key'],spec=cached['spec'],candidates=cached['candidates'],current_frame_current_ROI_NN_adopted=True));continue
  key=cam+'_'+hashlib.sha256(json.dumps(row['ROI']).encode()).hexdigest()[:12];local=O/key;local.mkdir(exist_ok=True);remote=SO/key;sp=V/'sources'/str(frame)/cam/'receipt.json';source=load(sp);spec=dict(overlay=str(SV/'tools/sam3_overlay'),frozen_scripts=str(SV/'code/LookCloser/scripts'),raw_native_image=str(SV/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'tasks'/cam/'raw_native.png'),raw_native_image_sha256=source['raw_native_image_sha256'],source_receipt_sha256=sha(sp),checkpoint_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',crop_xyxy=row['ROI'],output=str(remote/'candidates'),frame=frame,camera=cam,prompts=PROMPTS,current_native_pixels_only=True);write(local/'spec.json',spec);bindings.append(dict(row,key=key,spec=str(local/'spec.json'),candidates=str(O/'remote_results'/key/'candidates')));jobs.append(dict(name=f'CURRENT{frame}_{key}_parts_v105',host='silver',command=[recipe['remote_python'],str(SV/'operations/native_text_actor_parts_v6.py'),'--spec',str(remote/'spec.json')],env={'CREW_PILOT_LEASE_RECEIPT':'{job_root}/receipt.json'},metrics_file=str(remote/'candidates/receipt.json')))
 if not jobs:write(O/'bindings.json',dict(frame=frame,records=bindings,current_frame_qualified_NN_reuse_only=True));return
 if not(O/'launch.json').exists():
  subprocess.run(['ssh','-o','BatchMode=yes',recipe['remote_host'],shlex.join(['mkdir','-p',str(SO)])],check=True);subprocess.run(['rsync','-rt','--include=*/','--include=spec.json','--exclude=*',str(O)+'/',recipe['remote_host']+':'+str(SO)+'/'],check=True);host=dict(address=recipe['remote_host'],python=recipe['remote_python'],controller=str(SV/'code/LookCloser/scripts/marshawn_tuning_controller.py'),root=str(SO/'dispatch'),gpus=list(range(8)));write(O/'plan.json',dict(schema_version=1,poll_seconds=5,hosts={'silver':host},jobs=jobs));request=V/'operations/native10_requests'/f'{frame}_native_parts_v105.json';write(request,dict(status='pending',host_gpus={'silver':list(range(8))},frame=frame,nonce=uuid.uuid4().hex,source_bound_plan=str(O/'plan.json'),source_bound_plan_sha256=sha(O/'plan.json'),created=time.time()));argv=[sys.executable,str(V/'operations/native_ROI_pool_controller_v6.py'),str(O/'controller'),str(O/'plan.json'),str(request)]
  with(O/'controller.log').open('ab')as log:p=subprocess.Popen(argv,env=dict(os.environ,PYTHONPATH=str(V/'operations/sam3_assist/code')+':'+str(V/'code/LookCloser/scripts')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(O/'launch.json',dict(pid=p.pid,argv=argv,plan_sha256=sha(O/'plan.json')))
 write(O/'bindings.json',dict(frame=frame,records=bindings))
 while True:
  p=O/'controller/state.json'
  if p.exists()and all(x['status']=='complete'for x in load(p)['jobs'].values()):break
  launch_record=load(O/'launch.json');pid=launch_record['pid'];cmd=Path(f'/proc/{pid}/cmdline')
  if not cmd.exists()or b'native_ROI_pool_controller_v6.py'not in cmd.read_bytes():
   with(O/'controller.log').open('ab')as log:p=subprocess.Popen(launch_record['argv'],env=dict(os.environ,PYTHONPATH=str(V/'operations/sam3_assist/code')+':'+str(V/'code/LookCloser/scripts')),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   launch_record.update(pid=p.pid,restarted=time.time());write(O/'launch.json',launch_record)
  time.sleep(10)
 subprocess.run(['rsync','-rt','--include=*/','--include=*.json','--include=*.npz','--exclude=*',recipe['remote_host']+':'+str(SO)+'/',str(O/'remote_results')+'/'],check=True)

CREW_CACHE={}
def crew_core(frame,cam,masks):
 key=(frame,cam)
 if key in CREW_CACHE:return CREW_CACHE[key]
 from marshawn_actor_conflict_v51 import qualified_current_crew_core
 previous=V/'frames'/f'{frame-5:06d}';roles=load(previous/'crew_guard'/(cam+'_people.json'))['people'];out=np.zeros((4320,8192),bool);proofs=[]
 with np.load(previous/'person_candidates'/(cam+'.npz'))as z:old=z['masks']
 with np.load(previous/'native_masks'/(cam+'.npz'))as z:oldH=z['qualified_actor_rgb']if'qualified_actor_rgb'in z.files else np.zeros((4320,8192),bool)
 with np.load(V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz'))as z:H=z['qualified_actor_rgb']if'qualified_actor_rgb'in z.files else np.zeros((4320,8192),bool)
 known=[(r,tracker.pool(old[r['instance']]&~oldH))for r in roles if r['status'].startswith(('verified_other_person','verified_crew'))]
 for i,m in enumerate(masks):
  visible=m&~H;coarse=tracker.pool(visible);rank=[(int((coarse&o).sum())/max(int((coarse|o).sum()),1),r,o)for r,o in known]
  if not rank:continue
  score,r,o=max(rank,key=lambda x:x[0])
  if score<.25:continue
  core,proof=qualified_current_crew_core(visible,o,dict(status=r['status'],IoU=score,instance=r['instance']))
  if proof.get('qualified'):
   out|=(cv2.resize(core.astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0)&visible;proofs.append(dict(instance=i,previous=r['instance'],proof=proof))
 CREW_CACHE[key]=(out,proofs);return out,proofs

def qualify_tile(frame,row):
 cam=row['camera'];sp=Path(row['spec']);C=Path(row['candidates']);s=load(sp);r=load(C/'receipt.json');assert r['source_spec_sha256']==sha(sp)and r['source_receipt_sha256']==sha(V/'sources'/str(frame)/cam/'receipt.json');box=s['crop_xyxy'];x0,y0,x1,y1=box
 with np.load(V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz'))as z:A=(z['black']|z['grey'])[y0:y1,x0:x1];Hprior=z['qualified_actor_rgb'][y0:y1,x0:x1]if'qualified_actor_rgb'in z.files else np.zeros_like(A);veto=z['source_ineligible'][y0:y1,x0:x1]
 with np.load(row['guidance_archive'])as z:flow=cv2.resize(z['flowed_complete_union'].astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1]>0
 near=cv2.distanceTransform((~A).astype('uint8'),cv2.DIST_L2,5)<=32;semantic=[]
 from crew_guard_plan_v53 import candidates
 cp,_=candidates(V,frame,cam)
 with np.load(cp)as z:current_masks=z['masks']
 from crew_guard_plan_v53 import candidates as original_candidates
 _,current_meta=original_candidates(V,frame,cam);selected=[v for v in current_meta.get('selected',{}).values()if v is not None];raw_core=cv2.distanceTransform(current_masks[selected].any(0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>32;raw_core=raw_core[y0:y1,x0:x1]
 rawCrew,crew_proof=crew_core(frame,cam,current_masks);crew=rawCrew[y0:y1,x0:x1];allowed=np.zeros((1008,1008),bool)
 for proposal in row['components']:
  _,labels=cv2.connectedComponents(current_masks[proposal['instance']].astype('uint8'),8);allowed|=labels[y0:y1,x0:x1]==proposal['native_component']
 allowed=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64
 for family,prompts in FAMILIES.items():
  rows=[]
  for prompt in prompts:
   p=C/(prompt.replace(' ','_')+'.npz');record=next(x for x in r['records']if x['prompt']==prompt);assert sha(p)==record['archive_sha256']
   with np.load(p)as z:
    masks=z['masks'];scores=z['scores'];assert np.array_equal(z['crop_xyxy'],box)
    for i in np.flatnonzero(scores>=.7):rows.append(dict(prompt=prompt,index=int(i),score=float(scores[i]),mask=masks[int(i)].copy(),archive=str(p),archive_sha256=sha(p)))
  for a,b in itertools.combinations(rows,2):
   if a['prompt']==b['prompt']:continue
   H=a['mask']&b['mask'];iou=int(H.sum())/max(int((a['mask']|b['mask']).sum()),1)
   if iou<.85:continue
   n,lab,stats,_=cv2.connectedComponentsWithStats(H.astype('uint8'),8)
   for k in range(1,n):
    part=(lab==k)&allowed;new=part&~Hprior
    if(new&crew).any()or not crew_proof:continue
    if new.sum()<32:continue
    anchor=float((part&(flow|A)).sum()/max(int(part.sum()),1));newflow=float((new&flow).sum()/max(int(new.sum()),1));newnear=float((new&near).sum()/max(int(new.sum()),1))
    if anchor<.2 or(newflow<.25 and newnear<.8):continue
    if family=='head'and float((part&A).sum()/max(int(part.sum()),1))<.8 and(int((new&raw_core).sum())<128 or newflow<.6):continue
    semantic.append(dict(family=family,a={k:v for k,v in a.items()if k!='mask'},b={k:v for k,v in b.items()if k!='mask'},IoU=iou,NN_intersection_component=k,new_pixels=int(new.sum()),new_actor_union_pixels=int((part&~A).sum()),existing_actor_semantically_qualified_pixels=int((part&A).sum()),complete_union_anchor_fraction=anchor,new_exact_union_flow_fraction=newflow,new_current_actor32native_fraction=newnear,independent_current_CREW_new_overlap=0,independent_current_CREW_context_available=True,crew_proof=crew_proof,mask=part))
 return semantic

def main(frame,cameras,report_only=True):
 O,records=plans(frame,cameras);launch(frame,O,records);output=[]
 for row in load(O/'bindings.json')['records']if records else[]:
  for q in qualify_tile(frame,row):output.append(dict(camera=row['camera'],ROI=row['ROI'],NN_pair={k:v for k,v in q.items()if k!='mask'},spec=row['spec'],candidates=row['candidates'],proposal=row['proposal'],proposal_sha256=row['proposal_sha256']))
 write(O/'semantic_pair_REVIEW_READY.json',dict(frame=frame,records=output,current_NN_only=True,previous_union_ROI_and_identity_only=True,no_native_or_field_modified=True,report_only=report_only));return O
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--cameras',nargs='+');a=p.parse_args();main(a.frame,a.cameras or load(V/'source_inventory.json')['cameras'])
