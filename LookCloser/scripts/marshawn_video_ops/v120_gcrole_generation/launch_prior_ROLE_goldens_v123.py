from pathlib import Path
import os,sys,json,subprocess,shlex,time,uuid
import cv2,numpy as np
from PIL import Image,ImageDraw
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations/private_prior_ROLE_goldens_v123';O.mkdir(exist_ok=True);sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from crew_guard_plan_v53 import candidates
from marshawn_current_parts_v105 import FAMILIES
SV=Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107');SO=SV/'operations/private_prior_ROLE_goldens_v123';SGPY='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/python'
def load(p):return json.loads(Path(p).read_text())
def scope(frame,cam,index):
 qpath=V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json';row=load(qpath)['records'][index];assert row['camera']==cam;q=row['NN_pair'];x0,y0,x1,y1=row['ROI'];ms=[]
 for a in[q['a'],q['b']]:
  assert sha(a['archive'])==a['archive_sha256']
  with np.load(a['archive'])as z:ms.append(z['masks'][a['index']])
 _,lab=cv2.connectedComponents((ms[0]&ms[1]).astype('uint8'),8);part=lab==q['NN_intersection_component'];cp,_=candidates(V,frame,cam)
 with np.load(cp)as z:masks=z['masks']
 br=next(x for x in load(V/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'bindings.json')['records']if x['camera']==cam and x['ROI']==row['ROI']);allowed=np.zeros_like(part)
 for pr in br['components']:
  _,lab=cv2.connectedComponents(masks[pr['instance']].astype('uint8'),8);allowed|=lab[y0:y1,x0:x1]==pr['native_component']
 part&=cv2.distanceTransform((~allowed).astype('uint8'),cv2.DIST_L2,5)<=64
 p=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}'/'applications'/cam/('old_'+cam+'.npz');p=p if p.exists()else V/'frames'/f'{frame:06d}'/'native_masks'/(cam+'.npz')
 with np.load(p)as z:A=(z['black']|z['grey'])[y0:y1,x0:x1]
 R=np.zeros((4320,8192),bool);R[y0:y1,x0:x1]=part&~A;assert int(R.sum())==q['new_actor_union_pixels'];return R,dict(current_qualification=str(qpath),current_qualification_sha256=sha(qpath),current_record_index=index,current_preSEM_native=str(p),current_preSEM_native_sha256=sha(p))
def main():
 cv2.setNumThreads(1);cases=[(1695,'MD','foot',21),(1705,'FB','hand',29),(1700,'ED','hand',151)];jobs=[];records=[]
 for frame,cam,family,index in cases:
  previous=frame-5;key=f'{cam}_prior{previous}_{family}';D=O/key;D.mkdir(exist_ok=True);R,proof=scope(frame,cam,index);sources=[load(V/'sources'/str(f)/cam/'receipt.json')for f in[frame,previous]];ims=[Image.open(s['raw_native_image']).convert('RGB')for s in sources];gray=[np.array(im.convert('L').resize((1024,540),Image.Resampling.BILINEAR))for im in ims]
  flow=cv2.calcOpticalFlowFarneback(gray[0],gray[1],None,.5,3,21,3,7,1.5,0);back=cv2.calcOpticalFlowFarneback(gray[1],gray[0],None,.5,3,21,3,7,1.5,0);yy,xx=np.indices(gray[0].shape,dtype=np.float32);tx=xx+flow[...,0];ty=yy+flow[...,1];bx=cv2.remap(back[...,0],tx,ty,cv2.INTER_LINEAR);by=cv2.remap(back[...,1],tx,ty,cv2.INTER_LINEAR);fb=np.hypot(flow[...,0]+bx,flow[...,1]+by);photo=cv2.remap(gray[1],tx,ty,cv2.INTER_LINEAR);coarse=R.reshape(540,8,1024,8).any((1,3));valid=coarse&(fb<1.5)&(np.abs(photo.astype('float32')-gray[0].astype('float32'))<35)&(tx>=0)&(tx<1024)&(ty>=0)&(ty<540)
  if not valid.any():raise ValueError('No reliable backwards current part ROI proposal '+key)
  cx=float(np.median(tx[valid])*8);cy=float(np.median(ty[valid])*8);x0=max(0,min(8192-1008,int(round(cx))-504));y0=max(0,min(4320-1008,int(round(cy))-504));roi=[x0,y0,x0+1008,y0+1008];source=sources[1];remote=SO/key;spec=dict(overlay=str(SV/'tools/sam3_overlay'),frozen_scripts=str(SV/'code/LookCloser/scripts'),raw_native_image=str(SV/'operations/sam3_assist/v5'/f'frame{previous:06d}'/'tasks'/cam/'raw_native.png'),raw_native_image_sha256=source['raw_native_image_sha256'],source_receipt_sha256=sha(V/'sources'/str(previous)/cam/'receipt.json'),checkpoint_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',crop_xyxy=roi,output=str(remote/'candidates'),frame=previous,camera=cam,prompts=FAMILIES[family],current_native_pixels_only=True,purpose='PRIVATE_PRIOR_SAME_FAMILY_ROLE_GOLDEN_NOT_FG_APPLICATION',target_current_frame=frame)
  write(D/'spec.json',spec);current=ims[0].copy();ys,xs=np.where(R);cbox=[max(0,int(xs.min())-150),max(0,int(ys.min())-150),min(8192,int(xs.max())+151),min(4320,int(ys.max())+151)];panels=[]
  for im,title in[(current.crop(cbox),f'{frame} current qualified candidate'),(ims[1].crop(roi),f'{previous} actual PRIOR query1:1')]:
   im.thumbnail((700,700));tile=Image.new('RGB',(720,750));tile.paste(im,(0,35));ImageDraw.Draw(tile).text((5,8),title,fill='white');panels.append(tile)
  contact=Image.new('RGB',(1440,750));contact.paste(panels[0],(0,0));contact.paste(panels[1],(720,0));contact.save(D/'current_and_prior_ROI_source.jpg')
  records.append(dict(key=key,current_frame=frame,prior_frame=previous,camera=cam,family=family,crop_xyxy=roi,spec=str(D/'spec.json'),spec_sha256=sha(D/'spec.json'),current_part_native_pixels=int(R.sum()),backward_reliable_proposal_coarse_pixels=int(valid.sum()),ROI_prediction_only_NOT_actor_evidence=True,prior_native_image_sha256=source['raw_native_image_sha256'],current_source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),**proof));jobs.append(dict(name='PRIVATE_ROLE_'+key,host='silver',command=[SGPY,str(SV/'operations/native_text_actor_parts_v6.py'),'--spec',str(remote/'spec.json')],env={'CREW_PILOT_LEASE_RECEIPT':'{job_root}/receipt.json'},metrics_file=str(remote/'candidates/receipt.json')))
 host=dict(address='ubuntu@silver-sable',python=SGPY,controller=str(SV/'code/LookCloser/scripts/marshawn_tuning_controller.py'),root=str(SO/'dispatch'),gpus=list(range(8)));write(O/'plan.json',dict(schema_version=1,poll_seconds=5,hosts={'silver':host},jobs=jobs));write(O/'bindings.json',dict(kind='ROOT_AUTHORIZED_THREE_PRIVATE_PRIOR_ROLE_ROI_GOLDENS',records=records,maximum_ROIs=3,current_final_FG_NEVER_from_prior_pixels=True,created=time.time()));subprocess.run(['ssh','-o','BatchMode=yes','ubuntu@silver-sable',shlex.join(['mkdir','-p',str(SO)])],check=True);subprocess.run(['rsync','-rt','--include=*/','--include=spec.json','--exclude=*',str(O)+'/','ubuntu@silver-sable:'+str(SO)+'/'],check=True)
 request=V/'operations/native10_requests/private_prior_ROLE_goldens_v123.json';write(request,dict(status='pending',host_gpus={'silver':list(range(8))},frame=1690,actual_prior_frames=[1690,1695,1700],current_target_frames=[1695,1700,1705],nonce=uuid.uuid4().hex,source_bound_plan=str(O/'plan.json'),source_bound_plan_sha256=sha(O/'plan.json'),kind='FINITE_THREE_PRIOR_ROLE_NN_QUERY_GOLDENS',created=time.time()));argv=[sys.executable,str(V/'operations/native_ROI_pool_controller_v6.py'),str(O/'controller'),str(O/'plan.json'),str(request)]
 with(O/'controller.log').open('ab')as log:p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONPATH=str(V/'operations/sam3_assist/code')+':'+str(V/'code/LookCloser/scripts')))
 write(O/'launch.json',dict(pid=p.pid,argv=argv,plan_sha256=sha(O/'plan.json'),created=time.time()));print('PRIVATE_PRIOR_ROLE3_CONTROLLER',p.pid,flush=True)
if __name__=='__main__':main()
