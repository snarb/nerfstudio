"""Apply only ROOT-qualified current-image hair; never reuse warped pixels."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha

def main(root):
 frame=1675;camera='NA';f=root/'frames/001675';base=root/'operations/NA1675_native_hair';spec=json.loads((base/'hair_native1008_text_spec.json').read_text());archive=base/'fresh_text_candidates/hair.npz';nnreceipt=base/'fresh_text_candidates/receipt.json';r=json.loads(nnreceipt.read_text());source=root/'sources/1675/NA/receipt.json'
 assert r['source_receipt_sha256']==sha(source)and r['source_spec_sha256']==sha(base/'hair_native1008_text_spec.json')
 expected=next(row for row in r['records']if row['prompt']=='hair');assert expected['archive_sha256']==sha(archive)
 with np.load(archive)as z:hair=z['masks'][11].copy();score=float(z['scores'][11]);box=z['crop_xyxy']
 assert score>=.79 and np.array_equal(box,spec['crop_xyxy']);H=np.zeros((4320,8192),bool);x0,y0,x1,y1=box;H[y0:y1,x0:x1]=hair
 p=f/'native_masks/NA.npz';metadata=p.with_suffix('.json');m=json.loads(metadata.read_text());assert sha(p)==m['archive_sha256']
 with np.load(p)as z:before={k:z[k].copy()for k in z.files}
 after={k:v.copy()for k,v in before.items()};addition=H&~before['black']&~before['grey'];after['black']|=addition;after['source_ineligible']&=~H;after['qualified_actor_rgb']=H.copy();assert np.array_equal(after['unknown'],before['unknown']);assert np.array_equal(after['grey'],before['grey']);assert not(before['black']&~after['black']).any();assert not((after['source_ineligible']^before['source_ineligible'])&~H).any();assert int(addition.sum())==6910
 backup=f/'native_masks_before_hair_v5';backup.mkdir(exist_ok=True)
 for item in [p,metadata]:
  if(backup/item.name).exists():raise ValueError('Hair application already archived')
  shutil.copy2(item,backup/item.name)
 inputs=f/'qualified_hair_inputs/NA';inputs.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,inputs/'current_hair_candidates.npz');shutil.copy2(nnreceipt,inputs/'current_hair_NN_receipt.json');np.savez_compressed(inputs/'qualified_hair_native.npz',qualified_actor_rgb=H,crop_mask=hair,crop_xyxy=box)
 temp=p.with_suffix('.npz.hairtmp')
 with temp.open('wb')as stream:np.savez_compressed(stream,**after)
 temp.replace(p)
 oldproofpath=f/m['union_resolution']['receipt'];oldproof=json.loads(oldproofpath.read_text());shutil.copy2(oldproofpath,oldproofpath.with_name('application_before_qualified_hair.json'))
 hp=inputs/'application.json';hairproof=dict(frame=frame,camera=camera,kind='ROOT_QUALIFIED_CURRENT_VISIBLE_ACTOR_HAIR',actor='black',prompt='hair',instance=11,score=score,checkpoint_sha256=r['checkpoint_sha256'],source_receipt_sha256=sha(source),NN_candidate_archive_sha256=sha(archive),NN_receipt_sha256=sha(nnreceipt),native_hair_archive_sha256=sha(inputs/'qualified_hair_native.npz'),native_hair_mask_sha256=array_sha(H),old_native_sha256=m['archive_sha256'],new_native_sha256=sha(p),added_actor_pixels=int(addition.sum()),false_CREW_veto_trimmed_pixels=int((H&before['source_ineligible']).sum()),geometry_UNKNOWN_unchanged=True,source_RGB_UNKNOWN_override_ONLY_qualified_hair_pixels=int((H&before['unknown']).sum()),old_veto_sha256=array_sha(before['source_ineligible']),new_veto_sha256=array_sha(after['source_ineligible']),root_review='ROOT approved native hair mask11 score.793; bald cap21 rejected; face/leather negative prompts excluded; current RGB only',source_review_image=str(base/'fresh_text_candidates/TOP_NATIVE_candidates.jpg'),source_review_image_sha256=sha(base/'fresh_text_candidates/TOP_NATIVE_candidates.jpg'),no_previous_bitmap_as_foreground=True,created=time.time());write(hp,hairproof)
 target=(after['black']|after['grey'])&~after['source_ineligible'];oldproof.update(selected_pre_union_before_hair_sha256=oldproof['eligible_pre_union_sha256'],eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),new_native_sha256=sha(p),qualified_current_actor_additions=[dict(kind='visible_hair',receipt=str(hp.relative_to(f)),receipt_sha256=sha(hp))],source_target_definition='Current selected PRE union PLUS source-qualified current NN hair; independent veto excludes all else',qualified_actor_rgb_sha256=array_sha(H));write(oldproofpath,oldproof);m['archive_sha256']=sha(p);m['union_resolution']['receipt_sha256']=sha(oldproofpath);m['union_resolution']['eligible_pre_union_sha256']=array_sha(target);m['qualified_hair']=dict(receipt=str(hp.relative_to(f)),receipt_sha256=sha(hp),mask_sha256=array_sha(H));write(metadata,m);print(json.dumps(hairproof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
