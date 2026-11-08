"""Read-only, source-bound current native role decision contacts."""
from pathlib import Path
import json,hashlib,time,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(1)
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(frame):
 Q=G/'operations/current_native_parts_v105'/f'frame{frame:06d}'/'qualification.json';j=json.loads(Q.read_text());O=G/'operations/production_role_review_v139'/f'frame{frame:06d}';O.mkdir(parents=True,exist_ok=True)
 unknown=[i for i,r in enumerate(j['records'])if r['NN_pair'].get('ROLE_UNKNOWN_only')]
 feet=[i for i,r in enumerate(j['records'])if r['NN_pair']['family']=='foot'and not r['NN_pair'].get('ROLE_UNKNOWN_only')]
 selected=unknown[:]
 seen=set()
 for i in sorted(feet,key=lambda i:(not j['records'][i]['NN_pair']['target_human_role']['decision'].get('same_family_prior_role'),-j['records'][i]['NN_pair']['new_actor_union_pixels'])):
  cam=j['records'][i]['camera']
  if cam not in seen:selected.append(i);seen.add(cam)
  if len(seen)==4:break
 records=[];tiles=[]
 for i in selected:
  row=j['records'][i];q=row['NN_pair'];cam=row['camera'];x0,y0,x1,y1=row['ROI'];uncertain=bool(q.get('ROLE_UNKNOWN_only'))
  if uncertain:
   ap=Path(q['role_unknown_current_NN_archive']);assert sha(ap)==q['role_unknown_current_NN_archive_sha256']
   with np.load(ap)as z:m=z['role_unknown'].copy()
  else:
   for side in ['a','b']:assert sha(q[side]['archive'])==q[side]['archive_sha256']
   with np.load(q['a']['archive'])as z:a=z['masks'][q['a']['index']].copy()
   with np.load(q['b']['archive'])as z:b=z['masks'][q['b']['index']].copy()
   _,lab=cv2.connectedComponents((a&b).astype('uint8'),8);m=lab==q['NN_intersection_component']
   qualified=j['qualified_archives'][cam];assert sha(qualified['path'])==qualified['sha256']
   with np.load(qualified['path'])as z:m&=z['qualified_new_union'][y0:y1,x0:x1]
   ap=Path(qualified['path'])
  yy,xx=np.where(m)
  if len(xx):l=max(0,int(xx.min())-80);t=max(0,int(yy.min())-80);r=min(m.shape[1],int(xx.max())+81);b=min(m.shape[0],int(yy.max())+81)
  else:l=t=0;r=x1-x0;b=y1-y0
  source=G/'sources'/str(frame)/cam/'receipt.json';s=json.loads(source.read_text());raw=s['raw_native_image'];im=np.array(Image.open(raw).convert('RGB').crop((x0+l,y0+t,x0+r,y0+b)));mask=m[t:b,l:r];ov=im.copy();color=np.array([255,190,0]if uncertain else[0,230,255]);ov[mask]=(ov[mask]*.45+color*.55).astype('uint8');tile=Image.new('RGB',(960,530),'#171717');d=ImageDraw.Draw(tile);title=f'{frame} {cam} #{i} {q["family"]} '+('ROLE UNKNOWN'if uncertain else'RETAINED CURRENT NN');d.text((8,6),title,fill='white');d.text((8,24),f'native crop {x0+l},{y0+t},{x0+r},{y0+b}; highlighted {int(mask.sum())}px',fill='white')
  for n,pixels in enumerate([im,ov]):p=Image.fromarray(pixels);p.thumbnail((480,480));tile.paste(p,(n*480,50))
  out=O/(f'{i:03d}_{cam}_{q["family"]}_'+('UNKNOWN'if uncertain else'RETAINED'))
  # Path strings are composed explicitly, never used as model inputs.
  out=Path(str(out)+'.jpg');tile.save(out,quality=94);tiles.append(tile)
  records.append(dict(index=i,camera=cam,family=q['family'],role_unknown=uncertain,current_NN_scope_archive=str(ap),current_NN_scope_archive_sha256=sha(ap),source_receipt_sha256=sha(source),source_native_image_sha256=s['raw_native_image_sha256'],contact=str(out),contact_sha256=sha(out),native_crop=[x0+l,y0+t,x0+r,y0+b],highlighted_pixels=int(mask.sum()),target_human_role=q['target_human_role']))
 for start in range(0,len(tiles),4):
  page=Image.new('RGB',(1920,1060),'#171717')
  for k,tile in enumerate(tiles[start:start+4]):page.paste(tile,(k%2*960,k//2*530))
  page.save(O/f'contact_page_{start//4+1:02d}.jpg',quality=94)
 report=dict(kind='READONLY_PRODUCTION_ROLE_CURRENT_NATIVE_VISUALS_V139',frame=frame,qualification=str(Q),qualification_sha256=sha(Q),accepted_records=len(j['records'])-len(unknown),role_unknown_records=len(unknown),role_unknown_proposed_pixels_sum=sum(j['records'][i]['NN_pair']['originally_proposed_actor_union_pixels']for i in unknown),warning_and_representative_foot_records=records,NO_SOURCE_OR_POLICY_CHANGES=True,created=time.time());(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(str(O/'receipt.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);a=p.parse_args();run(a.frame)
