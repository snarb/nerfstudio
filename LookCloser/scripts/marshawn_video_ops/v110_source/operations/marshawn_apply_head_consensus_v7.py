"""Apply only policy-qualified current native semantic intersections, cumulatively."""
from pathlib import Path
import json,shutil,time
import numpy as np
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_head_hair_qualify_v71 import qualify

def apply(root,frame,camera,tiles,policy_path):
 root=Path(root);f=root/'frames'/f'{frame:06d}';native=f/'native_masks'/(camera+'.npz');metadata=native.with_suffix('.json');m=json.loads(metadata.read_text());policy=json.loads(Path(policy_path).read_text());assert policy['authorized']and frame in policy['frames']
 if(f/'native_semantic_ready.json').exists()or(f/'dispatch_ready.json').exists():raise ValueError('Published source inputs cannot change')
 with np.load(native)as z:before={k:z[k].copy()for k in z.files}
 if sha(native)!=m['archive_sha256']:raise ValueError('Current native source changed')
 H=np.zeros_like(before['black']);evidence=[]
 for row in tiles:
  q,fg,_=qualify(root,frame,camera,row['spec'],row['candidates'],policy)
  if q['status']!='CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR':continue
  x0,y0,x1,y1=q['crop_xyxy'];H[y0:y1,x0:x1]|=fg;evidence.append(q)
 oldH=before.get('qualified_actor_rgb',np.zeros_like(H));newscope=H&~oldH
 if not newscope.any():return dict(camera=camera,changed=False,reason='No additional high-confidence current native hair beyond existing qualified scope')
 Htotal=oldH|H;after={k:v.copy()for k,v in before.items()};addition=H&~before['black']&~before['grey'];after['black']|=addition;after['source_ineligible']&=~H;after['qualified_actor_rgb']=Htotal
 for k in ['grey','unknown','actor_ownership_uncertain','crop_xyxy']:assert np.array_equal(before[k],after[k]),k
 assert not(before['black']&~after['black']).any()and not(after['black']&after['grey']).any()
 assert not((before['source_ineligible']^after['source_ineligible'])&~H).any()
 inputs=f/'qualified_hair_inputs'/camera;old_archive=f/'head_consensus_prior_sources_v7'/camera;old_archive.mkdir(parents=True,exist_ok=True)
 if(old_archive/'native_before.npz').exists():raise ValueError('One combined current-frame consensus application only; keep previous application immutable')
 shutil.copy2(native,old_archive/'native_before.npz');shutil.copy2(metadata,old_archive/'native_before.json');oldscope_receipt=None
 if inputs.exists():
  shutil.copytree(inputs,old_archive/'qualified_source_before',dirs_exist_ok=True);oldscope_receipt=dict(path=str((old_archive/'qualified_source_before').relative_to(f)),source_metadata_sha256=sha(old_archive/'native_before.json'))
 inputs.mkdir(parents=True,exist_ok=True);maskpath=inputs/'qualified_hair_native.npz';np.savez_compressed(maskpath,qualified_actor_rgb=Htotal,new_semantic_consensus=H)
 temp=native.with_suffix('.npz.consensustmp')
 with temp.open('wb')as stream:np.savez_compressed(stream,**after)
 temp.replace(native)
 source=root/'sources'/str(frame)/camera/'receipt.json';proofpath=inputs/'application.json';proof=dict(frame=frame,camera=camera,kind='CURRENT_NATIVE_DUAL_PROMPT_ACTOR_HAIR_CONSENSUS_V7',source_receipt_sha256=sha(source),native_hair_archive_sha256=sha(maskpath),native_hair_mask_sha256=array_sha(Htotal),old_native_sha256=m['archive_sha256'],new_native_sha256=sha(native),policy_sha256=sha(policy_path),current_native_semantic_evidence=evidence,prior_same_frame_qualified_scope=oldscope_receipt,added_actor_pixels=int(addition.sum()),false_CREW_veto_trimmed_pixels=int((H&before['source_ineligible']).sum()),geometry_UNKNOWN_unchanged=True,no_previous_frame_bitmap_as_foreground=True,human_review_claim=False,created=time.time());write(proofpath,proof)
 p=f/m['union_resolution']['receipt'];u=json.loads(p.read_text());shutil.copy2(p,old_archive/'union_application_before.json');target=(after['black']|after['grey'])&~after['source_ineligible'];u.update(new_native_sha256=sha(native),eligible_pre_union_sha256=array_sha(target),eligible_post_union_sha256=array_sha(target),pre_union_eligible_pixels=int(target.sum()),post_union_eligible_pixels=int(target.sum()),qualified_current_actor_additions=[dict(kind='visible_current_native_semantic_hair',receipt=str(proofpath.relative_to(f)),receipt_sha256=sha(proofpath))],qualified_actor_rgb_sha256=array_sha(Htotal),source_target_definition='Current selected PRE union PLUS cumulative qualified SAME-FRAME current native semantic hair; independent veto unchanged outside exact qualified masks');write(p,u);m.update(archive_sha256=sha(native));m['union_resolution'].update(receipt_sha256=sha(p),eligible_pre_union_sha256=array_sha(target));m['qualified_hair']=dict(receipt=str(proofpath.relative_to(f)),receipt_sha256=sha(proofpath),mask_sha256=array_sha(Htotal));write(metadata,m);return dict(camera=camera,changed=True,added_actor_pixels=int(addition.sum()),newscope_native_pixels=int(newscope.sum()),new_native_sha256=sha(native))
