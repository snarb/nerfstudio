"""Source-qualified native HEAD NN queue; previous images position ROIs only."""
from pathlib import Path
import argparse,sys,json,time,subprocess,shlex,uuid,os
import cv2,numpy as np
from PIL import Image
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write

def make(frame,previous):
 r=json.loads((V/'operations/sam3_assist/v5/full_release.json').read_text());SV=Path(r['remote_root']);O=V/'operations'/f'{frame}_head_ALL66_ROI_v73';O.mkdir(exist_ok=True);SO=SV/'operations'/O.name
 if(O/'launch.json').exists():return O
 cameras=json.loads((V/'source_inventory.json').read_text())['cameras'];prior=V/'frames'/f'{previous:06d}'/'data/admission_unknown_hair/manifest.json';m=json.loads(prior.read_text());assert m['frame']==previous and len(m['records'])==66;records=[];jobs=[]
 for cam in cameras:
  oldp=V/'sources'/str(previous)/cam/'receipt.json';newp=V/'sources'/str(frame)/cam/'receipt.json';old=json.loads(oldp.read_text());new=json.loads(newp.read_text());row=next(x for x in m['records']if x['camera']==cam);assert row['source_receipt_sha256']==sha(oldp)
  choices=row['proofs'];strong_choices=[p for p in choices if p['status']=='CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR'];chosen=strong_choices[0]if strong_choices else choices[-1]
  td=prior.parent/chosen['directory'];q=json.loads((td/'qualification_original.json').read_text());x0,y0,x1,y1=q['crop_xyxy']
  with np.load(td/'current_native_proposal.npz')as z:h=z['geometry_hair_proposal'].copy()
  strong=q['status']=='CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR';yy,xx=np.where(h);cx=float(np.median(xx))+x0 if strong and len(xx)else(x0+x1)/2;cy=float(np.median(yy))+y0 if strong and len(yy)else(y0+y1)/2
  if not strong:
   with np.load(V/'frames'/f'{previous:06d}'/'native_masks'/(cam+'.npz'))as nz:h=nz['black'][y0:y1,x0:x1]&~nz['actor_ownership_uncertain'][y0:y1,x0:x1]
   if not h.any():h=np.ones((y1-y0,x1-x0),bool)
  # Dense flow is an ROI proposal, never copied into any actor/UNKNOWN mask.
  pad=256;l=max(0,x0-pad);t=max(0,y0-pad);right=min(8192,x1+pad);bottom=min(4320,y1+pad);size=((right-l)//4,(bottom-t)//4)
  images=[]
  for s in [old,new]:
   assert sha(Path(s['raw_native_image']))==s['raw_native_image_sha256'];im=Image.open(s['raw_native_image']).convert('L').crop((l,t,right,bottom));images.append(np.asarray(im.resize(size,Image.Resampling.BOX)))
  flow=cv2.calcOpticalFlowFarneback(images[0],images[1],None,.5,4,25,5,7,1.5,0);sel=np.zeros((bottom-t,right-l),np.uint8);sel[y0-t:y1-t,x0-l:x1-l]=h;sel=cv2.resize(sel,size,interpolation=cv2.INTER_NEAREST).astype(bool);v=flow[sel];delta=np.median(v,axis=0)*4 if len(v)else np.zeros(2);delta=np.clip(delta,-256,256);cx+=float(delta[0]);cy+=float(delta[1]);nx=max(0,min(8192-1008,int(round(cx))-504));ny=max(0,min(4320-1008,int(round(cy))-504));roi=[nx,ny,nx+1008,ny+1008]
  local=O/cam;local.mkdir(exist_ok=True);remote=SO/cam;spec=dict(overlay=str(SV/'tools/sam3_overlay'),frozen_scripts=str(SV/'code/LookCloser/scripts'),raw_native_image=str(SV/'operations/sam3_assist/v5'/f'frame{frame:06d}'/'tasks'/cam/'raw_native.png'),raw_native_image_sha256=new['raw_native_image_sha256'],source_receipt_sha256=sha(newp),checkpoint_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',crop_xyxy=roi,output=str(remote/'candidates'),frame=frame,camera=cam,prompts=['dreadlocks','long dreadlocks','braids','blond dreadlocks','hair'],current_native_pixels_only=True,ROI_proposal=dict(previous_frame=previous,previous_source_receipt_sha256=sha(oldp),previous_head_manifest_sha256=sha(prior),previous_current_NN_proposal_sha256=sha(td/'current_native_proposal.npz'),current_source_receipt_sha256=sha(newp),flow_native_delta=delta.tolist(),previous_pixels_NEVER_final_FG_or_UNKNOWN=True));write(local/'spec.json',spec);records.append(dict(camera=cam,ROI=roi,source_spec_sha256=sha(local/'spec.json'),source_receipt_sha256=sha(newp)));jobs.append(dict(name=f'current{frame}_{cam}_head_native_v73',host='silver',command=[r['remote_python'],str(SV/'operations/native_text_actor_parts_v6.py'),'--spec',str(remote/'spec.json')],env={'CREW_PILOT_LEASE_RECEIPT':'{job_root}/receipt.json'},metrics_file=str(remote/'candidates/receipt.json')))
 host=dict(address=r['remote_host'],python=r['remote_python'],controller=str(SV/'code/LookCloser/scripts/marshawn_tuning_controller.py'),root=str(SO/'dispatch'),gpus=list(range(8)));plan=dict(schema_version=1,poll_seconds=5,hosts={'silver':host},jobs=jobs);write(O/'plan.json',plan);write(O/'source_ROI_proofs.json',dict(frame=frame,previous=previous,records=records,previous_frame_only_ROI_proposal=True))
 # Specs are tiny; source PNGs were verified by the independent native factory.
 subprocess.run(['ssh','-o','BatchMode=yes',r['remote_host'],shlex.join(['mkdir','-p',str(SO)])],check=True)
 subprocess.run(['rsync','-rt','--include=*/','--include=spec.json','--exclude=*',str(O)+'/',r['remote_host']+':'+str(SO)+'/'],check=True)
 request=V/'operations/native10_requests'/f'{frame}_head_ALL66_ROI_v73.json';write(request,dict(status='pending',host_gpus={'silver':list(range(8))},source_bound_plan=str(O/'plan.json'),source_bound_plan_sha256=sha(O/'plan.json'),worker_sha256=sha(V/'operations/native_text_actor_parts_v6.py'),frame=frame,nonce=uuid.uuid4().hex,created=time.time()))
 env=dict(os.environ,PYTHONPATH=str(V/'operations/sam3_assist/code')+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1');argv=[sys.executable,str(V/'operations/native_ROI_pool_controller_v6.py'),str(O/'controller'),str(O/'plan.json'),str(request)]
 with(O/'controller.log').open('ab')as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 write(O/'launch.json',dict(pid=p.pid,argv=argv,plan_sha256=sha(O/'plan.json'),source_ROI_proofs_sha256=sha(O/'source_ROI_proofs.json'),request=str(request),created=time.time()));print('HEAD_NATIVE_CONTROLLER',p.pid,'frame',frame,'cameras',len(cameras),flush=True);return O
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--previous',type=int,required=True);a=p.parse_args();make(a.frame,a.previous)
