"""Add a qualified CURRENT same-person component; never delete actor pixels."""
from pathlib import Path
import argparse,json,shutil,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from prepare_marshawn_tuning import sha,write

def apply(path):
 s=json.loads(path.read_text());root=Path(s['campaign']);frame=s['frame'];camera=s['camera'];f=root/'frames'/f'{frame:06d}';p=f/'native_masks'/f'{camera}.npz';m=p.with_suffix('.json');source=root/'sources'/str(frame)/camera/'receipt.json';c=Path(s['candidate_archive'])
 if not s.get('approved')or not s.get('source_review_evidence'):raise ValueError('Current source body ownership qualification required')
 if sha(c)!=s['candidate_archive_sha256']or sha(source)!=s['source_receipt_sha256']:raise ValueError('Actor-fragment current INPUT changed')
 record=json.loads(m.read_text())
 if any(r.get('spec_sha256')==sha(path)for r in record.get('actor_fragments',[])):return
 if sha(p)!=s['input_native_sha256']or record['archive_sha256']!=sha(p):raise ValueError('Original current actor mask changed')
 with np.load(p)as z:before={k:z[k].copy()for k in z.files}
 with np.load(c)as z:mask=z['masks'][s['person_instance']]
 n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8);component=labels==s['component'];assert int(component.sum())==s['component_pixels']
 actor=s['actor'];other='grey'if actor=='black'else'black';addition=component&~before[actor]&~before[other]&~before['source_ineligible'];after={k:v.copy()for k,v in before.items()};after[actor]|=addition
 # Keep independent UNKNOWN unchanged. A positive visible actor label does not
 # provide a clear-depth vote, and UNKNOWN must never erase this real limb.
 assert np.array_equal(after[other],before[other]);assert np.array_equal(after['source_ineligible'],before['source_ineligible']);assert np.array_equal(after['unknown'],before['unknown']);assert not(before[actor]&~after[actor]).any()
 backup=f/'native_masks_before_fragment_recovery';backup.mkdir(exist_ok=True)
 for a in [p,m]:
  if not(backup/a.name).exists():shutil.copy2(a,backup/a.name)
 inputs=f/'actor_fragment_inputs';inputs.mkdir(exist_ok=True);out=inputs/f'{camera}_person{s["person_instance"]}_component{s["component"]}.npz';np.savez_compressed(out,qualified_component=component,addition=addition)
 temp=p.with_suffix('.npz.fragmenttmp')
 with temp.open('wb')as stream:np.savez_compressed(stream,**after)
 temp.replace(p)
 proof=dict(qualified_component_archive_sha256=sha(out),frame=frame,camera=camera,actor=actor,person_instance=s['person_instance'],component=s['component'],component_pixels=int(component.sum()),added_current_native_pixels=int(addition.sum()),existing_gray_pixels_preserved=int((component&before['grey']).sum()),crew_veto_pixels_untouched=int((component&before['source_ineligible']).sum()),UNKNOWN_unchanged=True,all_existing_actor_pixels_preserved=True,only_current_NN_pixels_added=True,no_previous_bitmap_copy=True,source_receipt_sha256=sha(source),candidate_archive_sha256=sha(c),original_native_sha256=s['input_native_sha256'],new_native_sha256=sha(p),spec_sha256=sha(path),source_review_evidence=s['source_review_evidence'],head_anchor_evidence=s['head_anchor_evidence'],occlusion_context=s['occlusion_context'],created=time.time())
 proofpath=inputs/f'{camera}_fragment_{sha(path)[:12]}.json';write(proofpath,proof);record.update(archive_sha256=sha(p));record.setdefault('actor_fragments',[]).append(dict(spec_sha256=sha(path),receipt_path=str(proofpath.relative_to(f)),receipt_sha256=sha(proofpath)));write(m,record)
 y,x=np.where(component);box=(max(0,x.min()-128),max(0,y.min()-128),min(8192,x.max()+129),min(4320,y.max()+129));x0,y0,x1,y1=box;src=json.loads(source.read_text());raw=np.array(Image.open(src['raw_native_image']).convert('RGB'))[y0:y1,x0:x1];sheet=Image.new('RGB',((x1-x0)*4,y1-y0+24))
 for j,(text,overlay)in enumerate([('SOURCE',np.zeros_like(addition)),('BLACK BEFORE',before[actor]),('QUALIFIED CURRENT ADDITION',addition),('BLACK AFTER',after[actor])]):
  image=raw.copy();v=overlay[y0:y1,x0:x1];image[v]=(image[v]*.4+np.array([0,255,255])*.6).astype('uint8');sheet.paste(Image.fromarray(image),(j*(x1-x0),24));ImageDraw.Draw(sheet).text((j*(x1-x0)+4,4),f'{frame} {camera} '+text,fill='white')
 sheet.save(inputs/f'{camera}_leg_before_add_after_native.jpg',quality=96);print(json.dumps(proof),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);apply(p.parse_args().spec)
