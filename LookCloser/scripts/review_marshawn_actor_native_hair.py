"""Candidate-only native hair review. Suggestion is never automatic acceptance."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt

def main(a):
 a.output.mkdir(parents=True,exist_ok=False);rows=json.loads(a.request.read_text())['records'];results=[];panels=[]
 for row in rows:
  p=a.candidates/row['camera']/'candidates.npz'
  if not p.exists():continue
  ref=np.load(row['coarse_reference']);black=ref['black'];union=ref['two_actor_union'];z=np.load(p);m=z['masks'];den=(m|black).sum((1,2));iou=(m&black).sum((1,2))/np.maximum(den,1);idx=int(iou.argmax())if len(m)else None;distance=distance_transform_edt(~union);add=m[idx]&~union&(distance<=32)if idx is not None else np.zeros_like(union);image=np.asarray(Image.open(row['image']).convert('RGB'));strip=Image.new('RGB',(1008,366));labels=['SOURCE','BLACK coarse','suggested TEXT'+str(idx)+' ADD'+str(int(add.sum()))]
  for j,mask in enumerate([np.zeros_like(union),black,m[idx]if idx is not None else np.zeros_like(union)]):
   ov=image.copy();ov[mask]=(ov[mask]*.5+np.array([0,255,255])*.5).astype('uint8')
   if j==2:ov[add]=[255,30,30]
   im=Image.fromarray(ov);im.thumbnail((336,336));strip.paste(im,(j*336,30));ImageDraw.Draw(strip).text((j*336+3,3),row['camera']+' '+labels[j],fill='white')
  strip.save(a.output/(row['camera']+'_source_coarse_proposal.jpg'));panels.append(strip);results.append({'camera':row['camera'],'selected_instance_SUGGESTION_ONLY':idx,'IoU_against_coarse_black':iou.tolist(),'add_only_within32_native_pixels_before_occluder_veto':int(add.sum()),'scope_measured_headpoint_count':row['scope_measured_headpoint_count'],'source_visibility':'PENDING_MANUAL','accept_add_only':False,'crop_xyxy':row['crop_xyxy'],'candidate_archive':str(p)})
 for n in range(0,len(panels),6):
  page=Image.new('RGB',(1008,366*len(panels[n:n+6])))
  for j,panel in enumerate(panels[n:n+6]):page.paste(panel,(0,j*366))
  page.save(a.output/('contact_%02d.jpg'%(n//6+1)))
 (a.output/'suggestions_NOT_APPROVAL.json').write_text(json.dumps({'rows':results,'never_subtract_gray_actor':True,'manual_required':True},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['request','candidates','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
