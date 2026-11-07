"""Apply a source-qualified current-frame visible-crew veto; preserve originals."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from PIL import Image,ImageDraw
from prepare_marshawn_tuning import sha,write

def apply(specpath):
 spec=json.loads(specpath.read_text());root=Path(spec['campaign']);frame=spec['frame'];camera=spec['camera'];froot=root/'frames'/f'{frame:06d}';native=froot/'native_masks';archive=native/(camera+'.npz');metadata=native/(camera+'.json');source=root/'sources'/str(frame)/camera/'receipt.json';candidate=Path(spec['candidate_archive'])
 if sha(source)!=spec['source_receipt_sha256'] or sha(candidate)!=spec['candidate_archive_sha256']:raise ValueError('Current-frame crew INPUT source changed')
 if not spec.get('approved') or not spec.get('approval_evidence'):raise ValueError('Qualified crew INPUT required')
 record=json.loads(metadata.read_text())
 if record.get('crew_guard',{}).get('spec_sha256')==sha(specpath):return
 if sha(archive)!=spec['input_native_archive_sha256'] or record['archive_sha256']!=sha(archive):raise ValueError('Original mask source changed')
 with np.load(archive,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
 with np.load(candidate,allow_pickle=False) as z:
  masks=z['masks'];indices=spec.get('candidate_indices',[spec.get('candidate_index')]);mask=(masks[indices]>0).any(0);box=z['crop_xyxy'].astype(int)
 if not np.isin(mask,[0,1]).all() or box.tolist()!=spec['crop_xyxy']:raise ValueError('Binary native crew ROI contract differs')
 x0,y0,x1,y1=box;crew=np.zeros((4320,8192),bool);crew[y0:y1,x0:x1]=mask.astype(bool);changed={k:v.copy() for k,v in original.items()};counts={}
 for actor in ['black','grey']:
  counts[actor]=int((original[actor]&crew).sum());changed[actor]&=~crew
 changed['source_ineligible']|=crew;changed['unknown']|=crew
 for actor in ['black','grey']:assert np.array_equal(changed[actor],original[actor]&~crew)
 backup=froot/'native_masks_unqualified_v1';backup.mkdir(exist_ok=True)
 for path in [archive,metadata]:
  dst=backup/path.name
  if not dst.exists():shutil.copy2(path,dst)
 crewroot=froot/'crew_inputs';crewroot.mkdir(exist_ok=True);np.savez_compressed(crewroot/(camera+'_bald_helper.npz'),mask=mask.astype(bool),crop_xyxy=box)
 temp=archive.with_suffix('.npz.guardtmp')
 with temp.open('wb') as stream:np.savez_compressed(stream,**changed)
 temp.replace(archive)
 receipt=dict(frame=frame,camera=camera,spec_sha256=sha(specpath),source_receipt_sha256=sha(source),original_native_archive_sha256=spec['input_native_archive_sha256'],corrected_native_archive_sha256=sha(archive),current_crew_candidate_archive_sha256=sha(candidate),candidate_indices=indices,crew_roi=box.tolist(),crew_pixels=int(crew.sum()),actor_pixels_removed=counts,all_pixels_outside_crew_mask_unchanged=True,field_recipe_changed=False,RGB_changed=False,previous_frame_mask_subtracted=False,approval_evidence=spec['approval_evidence'],created=time.time())
 receipt_name=camera+'_guard_'+sha(specpath)[:12]+'.json'
 write(crewroot/receipt_name,receipt);write(crewroot/(camera+'_guard_receipt.json'),receipt);record.update(archive_sha256=sha(archive),crew_guard=dict(status='qualified_current_frame_visible_crew_veto',spec_sha256=sha(specpath),receipt_path='crew_inputs/'+receipt_name,receipt_sha256=sha(crewroot/receipt_name)));write(metadata,record)
 row=json.loads(source.read_text());im=np.array(Image.open(row['raw_native_image']).convert('RGB'))[y0:y1,x0:x1];sheet=Image.new('RGB',((x1-x0)*4,y1-y0+24))
 for j,(label,m) in enumerate([('SOURCE',np.zeros_like(mask,bool)),('GREY BEFORE',original['grey'][y0:y1,x0:x1]),('VISIBLE CREW VETO',mask.astype(bool)),('GREY AFTER',changed['grey'][y0:y1,x0:x1])]):
  p=im.copy();p[m]=(p[m]*.4+np.array([0,255,255])*.6).astype('uint8');sheet.paste(Image.fromarray(p),(j*(x1-x0),24));ImageDraw.Draw(sheet).text((j*(x1-x0)+4,4),f'{frame} {camera} '+label,fill='white')
 sheet.save(crewroot/(camera+'_before_veto_after_native.jpg'),quality=96);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);apply(p.parse_args().spec)
