"""Qualified CURRENT native NN intersections; preserve all existing actors/geometry."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np,cv2
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha

def apply(specpath):
 s=json.loads(specpath.read_text());V=Path(s['root']);frame=s['frame'];cam=s['camera'];f=V/'frames'/f'{frame:06d}';source=V/'sources'/str(frame)/cam/'receipt.json';p=f/'native_masks'/(cam+'.npz');meta=p.with_suffix('.json');m=json.loads(meta.read_text())
 if not s.get('approved')or not s.get('source_review_evidence'):raise ValueError('Current source qualification required')
 if sha(source)!=s['source_receipt_sha256']or sha(p)!=s['input_native_sha256']or m['archive_sha256']!=sha(p):raise ValueError('Qualified current source changed')
 if(f/'native_semantic_ready.json').exists()or(f/'dispatch_ready.json').exists():raise ValueError('Never alter published source inputs')
 with np.load(p)as z:before={k:z[k].copy()for k in z.files}
 after={k:v.copy()for k,v in before.items()};H=np.zeros_like(before['black']);partrecords=[]
 for part in s['parts']:
  archive=Path(part['archive']);assert sha(archive)==part['archive_sha256'];NN=Path(part['NN_receipt']);r=json.loads(NN.read_text());assert sha(NN)==part['NN_receipt_sha256']and r['source_receipt_sha256']==sha(source)
  assert any(item['archive_sha256']==sha(archive)for item in r['records']), 'Candidate archive is not a hashed output of the current NN receipt'
  assert r['checkpoint_sha256']=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
  with np.load(archive)as z:
   mask=z[part.get('mask_key','masks')][part['index']].copy();box=z['crop_xyxy'];score=float(z['scores'][part['index']])
  assert mask.dtype==np.bool_ and score==part['score'];x0,y0,x1,y1=map(int,box);assert mask.shape==(y1-y0,x1-x0)
  if part.get('intersection'):
   other=part['intersection'];op=Path(other['archive']);orp=Path(other['NN_receipt']);orr=json.loads(orp.read_text());assert sha(op)==other['archive_sha256']and sha(orp)==other['NN_receipt_sha256'];assert orr['source_receipt_sha256']==sha(source)and orr['checkpoint_sha256']==r['checkpoint_sha256'];assert any(item['archive_sha256']==sha(op)for item in orr['records'])
   with np.load(op)as oz:
    assert np.array_equal(oz['crop_xyxy'],box);assert float(oz['scores'][other['index']])==other['score'];mask&=oz['masks'][other['index']]
  if part.get('components'):
   _,labels,_,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8);mask=np.isin(labels,part['components'])
  h=np.zeros_like(H);h[y0:y1,x0:x1]=mask;actor=part['actor'];assert actor in ['black','grey'];other='grey'if actor=='black'else'black';addition=h&~after[actor]&~after[other];after[actor]|=addition;H|=h;partrecords.append(dict(part,qualified_native_pixels=int(h.sum()),added_actor_pixels=int(addition.sum())))
 if'qualified_actor_rgb'in before:raise ValueError('Cumulative repair needs one fresh combined source spec; refuse hidden replacement')
 after['source_ineligible']&=~H;after['qualified_actor_rgb']=H
 assert np.array_equal(after['unknown'],before['unknown'])and np.array_equal(after['actor_ownership_uncertain'],before['actor_ownership_uncertain'])
 assert not(before['black']&~after['black']).any()and not(before['grey']&~after['grey']).any()and not(after['black']&after['grey']).any()
 assert not((after['source_ineligible']^before['source_ineligible'])&~H).any()
 backup=f/'native_masks_before_qualified_parts_v6';backup.mkdir(exist_ok=True)
 for old in [p,meta]:
  if(backup/old.name).exists():raise ValueError('Qualified actor part input already archived')
  shutil.copy2(old,backup/old.name)
 inputs=f/'qualified_hair_inputs'/cam;inputs.mkdir(parents=True,exist_ok=True);out=inputs/'qualified_hair_native.npz';np.savez_compressed(out,qualified_actor_rgb=H)
 temp=p.with_suffix('.npz.parttmp')
 with temp.open('wb')as stream:np.savez_compressed(stream,**after)
 temp.replace(p)
 hp=inputs/'application.json';proof=dict(frame=frame,camera=cam,kind='QUALIFIED_CURRENT_VISIBLE_ACTOR_PARTS',source_receipt_sha256=sha(source),native_hair_archive_sha256=sha(out),native_hair_mask_sha256=array_sha(H),old_native_sha256=m['archive_sha256'],new_native_sha256=sha(p),parts=partrecords,source_review_evidence=s['source_review_evidence'],application_spec_sha256=sha(specpath),added_actor_pixels=sum(x['added_actor_pixels']for x in partrecords),false_CREW_veto_trimmed_pixels=int((H&before['source_ineligible']).sum()),geometry_UNKNOWN_unchanged=True,source_RGB_UNKNOWN_override_ONLY_qualified_hair_pixels=int((H&before['unknown']).sum()),no_previous_bitmap_as_foreground=True,created=time.time());write(hp,proof)
 up=f/m['union_resolution']['receipt'];u=json.loads(up.read_text());shutil.copy2(up,up.with_name('application_before_current_parts_v6.json'));target=(after['black']|after['grey'])&~after['source_ineligible'];u.update(selected_pre_union_before_parts_sha256=u['eligible_pre_union_sha256'],eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),new_native_sha256=sha(p),qualified_current_actor_additions=[dict(kind='visible_actor_parts',receipt=str(hp.relative_to(f)),receipt_sha256=sha(hp))],qualified_actor_rgb_sha256=array_sha(H),source_target_definition='Current selected PRE union PLUS source-qualified fresh native NN actor parts; independent veto unchanged outside those exact masks');write(up,u);m.update(archive_sha256=sha(p));m['union_resolution'].update(receipt_sha256=sha(up),eligible_pre_union_sha256=array_sha(target));m['qualified_hair']=dict(receipt=str(hp.relative_to(f)),receipt_sha256=sha(hp),mask_sha256=array_sha(H));write(meta,m);return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();print(json.dumps(apply(a.spec)))
