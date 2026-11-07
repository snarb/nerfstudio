"""Fresh native crew segmentation guided by 2D identity references, not mask transfer."""
from pathlib import Path
import argparse,json,os,sys,time
import cv2,numpy as np
from PIL import Image,ImageDraw

def sha(p):
 import hashlib
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def small(m):return cv2.resize(m.astype('uint8'),(512,270),interpolation=cv2.INTER_AREA)>0

def points_inside(mask,count=4):
 from scipy.ndimage import distance_transform_edt
 score=distance_transform_edt(mask);points=[]
 for _ in range(count):
  y,x=np.unravel_index(score.argmax(),score.shape)
  if score[y,x]<8:break
  points.append([x,y]);yy,xx=np.ogrid[:score.shape[0],:score.shape[1]];score[(xx-x)**2+(yy-y)**2<80**2]=0
 return np.asarray(points,dtype=np.float32).reshape(-1,2)

def main(a):
 spec=json.loads(a.spec.read_text());overlay=Path(spec['overlay']);sys.path[:0]=[str(overlay/'src'),str(overlay/'deps'),spec['frozen_scripts']]
 import torch
 from scipy.ndimage import distance_transform_edt
 from marshawn_gpu_isolation import require_controller_lease,verify_cuda
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 gpu=require_controller_lease(os.environ['CREW_PILOT_LEASE_RECEIPT']);gpu.update(verify_cuda(gpu['expected_physical_uuid'],torch));os.fstat(int(os.environ['LOOKCLOSER_GPU_LEASE_FD']))
 torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(min(.9,16*1024**3/torch.cuda.get_device_properties(0).total_memory))
 for key in ['previous_image','current_image','previous_crew','current_actors']:
  if sha(spec[key])!=spec[key+'_sha256']:raise ValueError('Crew identity INPUT changed '+key)
 checkpoint=overlay/'weights/sam3.pt';assert sha(checkpoint)==spec['checkpoint_sha256']
 with np.load(spec['previous_crew'])as z:oldmask=z['mask'].astype(bool);oldbox=z['crop_xyxy'].astype(int)
 x0,y0,x1,y1=oldbox;prev=Image.open(spec['previous_image']).convert('RGB');image=Image.open(spec['current_image']).convert('RGB');pg=cv2.cvtColor(np.array(prev.crop(tuple(oldbox))),cv2.COLOR_RGB2GRAY);cg=cv2.cvtColor(np.array(image.crop(tuple(oldbox))),cv2.COLOR_RGB2GRAY)
 interior=cv2.erode(oldmask.astype('uint8'),np.ones((9,9),np.uint8));feature=cv2.goodFeaturesToTrack(pg,maxCorners=64,qualityLevel=.01,minDistance=12,mask=interior)
 if feature is None or len(feature)<8:raise ValueError('UNKNOWN: insufficient tracked crew identity features')
 tracked,forward,_=cv2.calcOpticalFlowPyrLK(pg,cg,feature,None,winSize=(31,31),maxLevel=3);back,backward,_=cv2.calcOpticalFlowPyrLK(cg,pg,tracked,None,winSize=(31,31),maxLevel=3);displacement=tracked[:,0]-feature[:,0];median=np.median(displacement,axis=0);fb=np.linalg.norm(back[:,0]-feature[:,0],axis=1);deviation=np.linalg.norm(displacement-median,axis=1);good=forward[:,0].astype(bool)&backward[:,0].astype(bool)&(fb<1.5)&(deviation<8)
 if good.sum()<8 or good.mean()<.4 or np.linalg.norm(median)>180:raise ValueError('UNKNOWN: unreliable current crew identity tracking')
 delta=np.median(displacement[good],axis=0);nx0=int(np.clip(round(x0+delta[0]),0,8192-1008));ny0=int(np.clip(round(y0+delta[1]),0,4320-1008));box=np.array([nx0,ny0,nx0+1008,ny0+1008]);offset=box[:2]-oldbox[:2];pred=cv2.warpAffine(oldmask.astype('uint8'),np.asarray([[1,0,delta[0]-offset[0]],[0,1,delta[1]-offset[1]]],np.float32),(1008,1008),flags=cv2.INTER_NEAREST)>0
 positive=tracked[good,0]-offset;positive=positive[(positive>=0).all(1)&(positive<1008).all(1)];positive=positive[::max(1,len(positive)//12)][:12]
 model=build_sam3_image_model(checkpoint_path=str(checkpoint),load_from_HF=False,enable_inst_interactivity=True);proc=Sam3Processor(model,confidence_threshold=.25)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  state=proc.set_image(image);global_result=proc.set_text_prompt(prompt='person',state=state);global_masks=global_result['masks'].detach().cpu().numpy().reshape(-1,4320,8192)
 pred_full=np.zeros((4320,8192),bool);pred_full[ny0:ny0+1008,nx0:nx0+1008]=pred;ps=small(pred_full);scores=np.asarray([(small(m)&ps).sum()/max((small(m)|ps).sum(),1)for m in global_masks]);order=np.argsort(scores)[::-1];eligible=[int(k)for k in order if int(k)not in spec['actor_global_instances']]
 if not eligible or scores[eligible[0]]<.10:raise ValueError('UNKNOWN: no fresh current-frame person matches crew identity')
 person=eligible[0];person_mask=global_masks[person];global_xy=np.rint(positive+box[:2]).astype(int);covered=person_mask[global_xy[:,1],global_xy[:,0]]
 if covered.mean()<.65:raise ValueError('UNKNOWN: current person does not retain tracked crew features')
 positive=positive[covered]
 with np.load(spec['current_actors'])as z:actors={k:z[k][ny0:ny0+1008,nx0:nx0+1008].copy()for k in ['black','grey']}
 safecrew=distance_transform_edt(~pred)<=96;negative=[]
 for actor,m in actors.items():
  safe=cv2.erode((m&~safecrew).astype('uint8'),np.ones((17,17),np.uint8))>0;negative.extend(points_inside(safe,4).tolist())
 if spec.get('exclude_current_black_core'):
  black_core=cv2.erode(actors['black'].astype('uint8'),np.ones((9,9),np.uint8))>0;xy=np.clip(np.rint(positive).astype(int),0,1007);keep=~black_core[xy[:,1],xy[:,0]];positive=positive[keep]
  if len(positive)<4:raise ValueError('UNKNOWN: crew feature prompts are occluded by current black actor')
 negative=np.asarray(negative,np.float32).reshape(-1,2)
 if len(negative)<2:raise ValueError('UNKNOWN: insufficient current actor negative prompts')
 head_reference=None;head_points=None;head_previous_pixels=None
 if spec.get('previous_head'):
  if sha(spec['previous_head'])!=spec['previous_head_sha256']:raise ValueError('Qualified crew HEAD identity source changed')
  with np.load(spec['previous_head'])as z:hm=z['mask'].astype('uint8');assert np.array_equal(z['crop_xyxy'],oldbox)
  head_previous_pixels=int(hm.sum());head_reference=cv2.warpAffine(hm,np.asarray([[1,0,delta[0]-offset[0]],[0,1,delta[1]-offset[1]]],np.float32),(1008,1008),flags=cv2.INTER_NEAREST)>0
  current_person_roi=person_mask[ny0:ny0+1008,nx0:nx0+1008];visible=head_reference&current_person_roi&~cv2.dilate(actors['black'].astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
  head_points=points_inside(visible,2)
  if len(head_points)<1 or visible.sum()<512:raise ValueError('UNKNOWN: qualified crew head has no distinct current visible positive proposal')
  positive=np.r_[head_points,positive]
 allowed=distance_transform_edt(~pred)<=128;out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True);accepted=[];allmasks={};quality=[]
 # Body and head references choose prompts only. Every final pixel below is
 # predicted from the current source image and is never a warped old bitmap.
 head=head_points if head_points is not None else positive[positive[:,1]<=np.percentile(positive[:,1],45)]
 if len(head)<2:head=positive[:2]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for kind,pts in [('body',positive),('head',head)]:
   state=proc.set_image(image.crop(tuple(box)));prompt=np.r_[pts,negative];labels=np.r_[np.ones(len(pts)),np.zeros(len(negative))].astype('int32');masks,nnscore,_=model.predict_inst(state,point_coords=prompt.astype('float32'),point_labels=labels,multimask_output=True);masks=np.asarray(masks)>0;allmasks[kind]=masks
   for k,m in enumerate(masks):
    pp=np.clip(np.rint(pts).astype(int),0,1007);nn=np.clip(np.rint(negative).astype(int),0,1007);coverage=float(m[pp[:,1],pp[:,0]].mean());negative_hits=int(m[nn[:,1],nn[:,0]].sum());area=int(m.sum());outside=int((m&~allowed).sum());iou=float((m&pred).sum()/max((m|pred).sum(),1));score=float(nnscore[k]);head_iou=float((m&head_reference).sum()/max((m|head_reference).sum(),1)) if head_reference is not None else iou;head_size_ok=(head_previous_pixels is None or area<=max(head_previous_pixels*2,4096))
    accept=coverage>=.8 and negative_hits==0 and score>=(.65 if kind=='body'else .45) and (head_iou if kind=='head'else iou)>=.15 and (head_size_ok if kind=='head'else True) and area<=max(int(oldmask.sum()*2),8192) and outside<=max(512,int(area*.02))
    quality.append(dict(kind=kind,index=k,score=score,positive_coverage=coverage,actor_negative_hits=negative_hits,predicted_reference_IoU=iou,head_reference_IoU=head_iou,head_size_ok=head_size_ok,outside128native_reference=outside,pixels=area,accepted=accept))
    if accept:accepted.append(m)
 input_qualified=any(q['accepted']and q['kind']=='body'for q in quality) and any(q['accepted']and q['kind']=='head'for q in quality)
 final=np.logical_or.reduce(accepted) if accepted else np.zeros_like(pred);qualified_head=np.logical_or.reduce([allmasks['head'][q['index']]for q in quality if q['kind']=='head'and q['accepted']]) if any(q['kind']=='head'and q['accepted']for q in quality)else np.zeros_like(pred)
 np.savez_compressed(out/'candidates.npz',qualified_head_mask=qualified_head,masks=final[None],scores=np.array([min([q['score']for q in quality if q['accepted']]or[0.])]),crop_xyxy=box,body_masks=allmasks['body'],head_masks=allmasks['head'],positive=positive,negative=negative)
 np.savez_compressed(out/'current_people.npz',masks=global_masks)
 base=np.array(image.crop(tuple(box)))
 point_image=Image.fromarray(base.copy());draw=ImageDraw.Draw(point_image)
 for pt in positive:
  x,y=pt;draw.ellipse([x-5,y-5,x+5,y+5],outline='lime',width=3)
 for pt in negative:
  x,y=pt;draw.ellipse([x-5,y-5,x+5,y+5],outline='red',width=3)
 point_image.save(out/'points_current_native.png')
 sheet=Image.new('RGB',(3024,1032));panels=[('SOURCE current',np.zeros_like(final)),('WARPED LABEL FOR ID ONLY',pred),('FRESH CURRENT CREW',final)]
 for j,(label,m)in enumerate(panels):
  p=base.copy();p[m]=(p[m]*.4+np.array([0,255,255])*.6).astype('uint8');sheet.paste(Image.fromarray(p),(j*1008,24));ImageDraw.Draw(sheet).text((j*1008+4,4),label,fill='white')
 sheet.save(out/'contact.jpg',quality=96)
 candidate_sheet=Image.new('RGB',(3072,2112))
 for j,(kind,k)in enumerate([(a,b)for a in ['body','head']for b in range(3)]):
  m=allmasks[kind][k];p=base.copy();p[m]=(p[m]*.4+np.array([0,255,255])*.6).astype('uint8');candidate_sheet.paste(Image.fromarray(p),(j%3*1024,j//3*1056+24));q=next(v for v in quality if v['kind']==kind and v['index']==k);ImageDraw.Draw(candidate_sheet).text((j%3*1024+4,j//3*1056+4),kind+str(k)+' score'+str(q['score'])+' accepted'+str(q['accepted']),fill='white')
 candidate_sheet.save(out/'all_candidates_native.jpg',quality=96)
 receipt=dict(spec_sha256=sha(a.spec),code_sha256=sha(__file__),frame=spec['frame'],camera=spec['camera'],source_receipt_sha256=spec['source_receipt_sha256'],current_image_sha256=spec['current_image_sha256'],previous_crew_identity_sha256=spec['previous_crew_sha256'],current_candidate_archive_sha256=sha(out/'candidates.npz'),current_people_sha256=sha(out/'current_people.npz'),current_person_instance=person,current_person_identity_IoU=float(scores[person]),tracked_features=int(good.sum()),median_flow=delta.tolist(),forward_backward_max=float(fb[good].max()),quality=quality,previous_labels_used_only_for_identity_and_prompt_proposals=True,no_previous_bitmap_final_pixels=True,no_NeRF_state=True,gpu=gpu,status='CURRENT_CREW_CANDIDATE_PENDING_INITIAL_ROOT_CANARY'if input_qualified else'UNKNOWN_BODY_HEAD_QUALITY_GUARD',input_qualified=input_qualified,created=time.time());(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);main(p.parse_args())
