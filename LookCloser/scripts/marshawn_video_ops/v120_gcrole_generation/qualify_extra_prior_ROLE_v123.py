from pathlib import Path
import sys,json,time,itertools
import cv2,numpy as np
from PIL import Image,ImageDraw
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');O=V/'operations/private_prior_ROLE_goldens_v123';sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts'),str(V/'operations/semantic_actor_role_v120')]
from prepare_marshawn_tuning import sha,write
from launch_prior_ROLE_goldens_v123 import scope
import marshawn_union_tracking_proposals_v1 as tracker

def run():
 cv2.setNumThreads(1);out=[]
 for row in json.loads((O/'bindings.json').read_text())['records']:
  key=row['key'];frame=row['current_frame'];prior=row['prior_frame'];cam=row['camera'];D=O/key;spec=json.loads((D/'spec.json').read_text());C=O/'remote_results'/key/'candidates';receipt=json.loads((C/'receipt.json').read_text());assert receipt['source_spec_sha256']==sha(D/'spec.json')and receipt['source_receipt_sha256']==sha(V/'sources'/str(prior)/cam/'receipt.json');x0,y0,x1,y1=spec['crop_xyxy'];pn=G/'frames'/f'{prior:06d}'/'native_masks'/(cam+'.npz')
  with np.load(pn)as z:A=(z['black']|z['grey'])&~z['source_ineligible']
  source=json.loads((V/'sources'/str(prior)/cam/'receipt.json').read_text());raw=Image.open(source['raw_native_image']).convert('RGB').crop((x0,y0,x1,y1));base=np.array(raw);candidates=[];panels=[];roles=np.zeros((4320,8192),bool);pairs=[]
  def panel(title,mask=None,col=(0,235,255)):
   a=base.copy()
   if mask is not None:a[mask]=(.45*a[mask]+.55*np.asarray(col)).astype('uint8')
   p=Image.fromarray(a);p.thumbnail((480,480));t=Image.new('RGB',(500,525));t.paste(p,(0,32));ImageDraw.Draw(t).text((5,6),title,fill='white');return t
  panels=[panel(f'{prior} {cam} actual prior RAW'),panel('PRIOR final qualified ACTOR',A[y0:y1,x0:x1],(255,0,255))]
  for rec in receipt['records']:
   ap=C/(rec['prompt'].replace(' ','_')+'.npz');assert sha(ap)==rec['archive_sha256']
   with np.load(ap)as z:
    order=np.argsort(z['scores'])[::-1]
    for rank,i in enumerate(order):
     m=z['masks'][i].copy();score=float(z['scores'][i]);frac=int((m&A[y0:y1,x0:x1]).sum())/max(int(m.sum()),1)
     if rank<2:panels.append(panel(f"{rec['prompt']} #{i} {score:.3f} actor{frac:.2f}",m))
     if score>=.7:candidates.append(dict(prompt=rec['prompt'],index=int(i),score=score,mask=m,archive=str(ap),archive_sha256=sha(ap)))
  for a,b in itertools.combinations(candidates,2):
   if a['prompt']==b['prompt']:continue
   m=a['mask']&b['mask'];iou=int(m.sum())/max(int((a['mask']|b['mask']).sum()),1)
   if iou<.85:continue
   n,lab=cv2.connectedComponents(m.astype('uint8'),8)
   for k in range(1,n):
    c=lab==k;area=int(c.sum());over=int((c&A[y0:y1,x0:x1]).sum());frac=over/max(area,1);accept=area>=32 and frac>=.8
    if accept:roles[y0:y1,x0:x1]|=c&A[y0:y1,x0:x1]
    pairs.append(dict(a={k:v for k,v in a.items()if k!='mask'},b={k:v for k,v in b.items()if k!='mask'},IoU=iou,component=k,pixels=area,prior_qualified_actor_pixels=over,prior_actor_fraction=frac,qualified_role=accept))
  imgs=[]
  for f in[prior,frame]:
   s=json.loads((V/'sources'/str(f)/cam/'receipt.json').read_text());imgs.append(np.array(Image.open(s['raw_native_image']).convert('L').resize(tracker.SIZE,Image.Resampling.BILINEAR)))
  flow=tracker.flow_channels(imgs[0],imgs[1],{'complete_union':tracker.pool(roles)})['flowed_complete_union'];native=cv2.resize(flow.astype('uint8'),(8192,4320),interpolation=cv2.INTER_NEAREST)>0;R,_=scope(frame,cam,row['current_record_index']);fraction=int((R&native).sum())/max(int(R.sum()),1)
  cols=4;canvas=Image.new('RGB',(500*cols,525*((len(panels)+cols-1)//cols)))
  for i,p in enumerate(panels):canvas.paste(p,((i%cols)*500,(i//cols)*525))
  canvas.save(D/'PRIOR_NATIVE_ROLE_CANDIDATES.jpg');np.savez_compressed(D/'private_prior_ROLE_support.npz',prior_current_family_role=roles,flowed_same_family_role_native=native)
  result=dict(camera=cam,current_frame=frame,prior_frame=prior,family=row['family'],prior_ROI=spec['crop_xyxy'],spec_sha256=sha(D/'spec.json'),NN_receipt_sha256=sha(C/'receipt.json'),prior_native_sha256=sha(pn),new_current_residual_pixels=int(R.sum()),qualified_prior_same_family_role_pixels=int(roles.sum()),current_residual_role_flow_pixels=int((R&native).sum()),current_residual_role_flow_fraction=fraction,pairs=pairs,manual_prior_identity_review_pending=True,no_NN_or_source_arrays_modified=True,role_threshold_private_not_released=.8);write(D/'private_role_assessment.json',result);out.append(result);print(key,'rolepixels',int(roles.sum()),'Rflow',fraction,flush=True)
 write(O/'three_extra_prior_ROLE_assessments.json',dict(records=out,private_only=True,manual_prior_identity_review_pending=True,created=time.time()))
if __name__=='__main__':run()
