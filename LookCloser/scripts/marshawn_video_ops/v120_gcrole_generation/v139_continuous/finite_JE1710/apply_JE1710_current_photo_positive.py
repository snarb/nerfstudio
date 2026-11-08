"""Finite ROOT-photo-qualified current NN actor-role correction; no policy edit."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,sys,time
import numpy as np
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(G/'operations'),str(G/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union
O=G/'operations/1710_JE12226_current_positive';F=G/'frames/001710';J=G/'operations/ROOT_1710_JE12226_photo_role_judgment.json';AUTH=O/'ROOT_application_authorization.json'
def load(p):return json.loads(Path(p).read_text())
def npz(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.npz.atomic_tmp')
 with t.open('wb')as f:np.savez_compressed(f,**a)
 t.replace(p)
def scope_inputs():
 sim=load(O/'simulation.json');j=load(J);p=Path(j['exact_current_NN_scope_archive']);assert sha(p)==j['exact_current_NN_scope_sha256']
 with np.load(p)as z:R=np.zeros((4320,8192),bool);x0,y0,x1,y1=z['crop_xyxy'];R[y0:y1,x0:x1]=z['role_unknown']
 assert R.sum()==12226 and array_sha(R)==sim['exact_scope_sha256']and sha(O/'exact_CURRENT_NN_positive.npz')==sim['proposal_archive_sha256']
 q=load(G/'operations/current_native_parts_v105/frame001710/qualification.json');assert sha(G/'operations/current_native_parts_v105/frame001710/qualification.json')==j['current_qualification_sha256']
 with np.load(F/'role_unknown_visibility_inputs_v135/JE/CURRENT_NN_role_UNKNOWN.npz')as z:U=z['role_unknown'].copy();soft=z['soft_eligible'].copy()
 left=U&~R
 other=q['records'][104]['NN_pair'];assert other['originally_proposed_actor_union_pixels']==74
 with np.load(other['role_unknown_current_NN_archive'])as z:
  P=np.zeros_like(R);l,t,r,b=z['crop_xyxy'];P[t:b,l:r]=z['role_unknown']
 assert np.array_equal(P,left)and not(P&R).any()
 # Record overlap truthfully; only approved positive scope is subtracted.
 return R,left,soft&left,sim,dict(original_UNKNOWN_pixels=int(U.sum()),approved_intersection=int((U&R).sum()),remaining_UNKNOWN_pixels=int(left.sum()),JE74_overlap_approved_pixels=int((P&R).sum()))
def plan():
 R,left,soft,sim,counts=scope_inputs();assert counts['remaining_UNKNOWN_pixels']==74
 print(json.dumps(dict(applier=__file__,applier_sha256=sha(__file__),scope_sha256=array_sha(R),counts=counts,remaining_array_sha256=array_sha(left),simulation_sha256=sha(O/'simulation.json'),judgment_sha256=sha(J),not_applied=True),indent=2))
def apply():
 a=load(AUTH);assert a['authorized'] is True and a['frame']==1710 and a['camera']=='JE'and a['applier_sha256']==sha(__file__)and a['photo_judgment_sha256']==sha(J)and a['simulation_sha256']==sha(O/'simulation.json')
 lock=open(G/'operations/source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(O/'application_complete.json').exists():
  d=load(O/'application_complete.json');assert sha(F/'native_masks/JE.npz')==d['new_native_sha256'];validate_union(G,1710,'JE');return
 assert not(F/'native_semantic_ready.json').exists()and not(F/'dispatch_ready.json').exists()
 R,left,soft,sim,counts=scope_inputs();assert counts['remaining_UNKNOWN_pixels']==74
 p=F/'native_masks/JE.npz';mp=p.with_suffix('.json');assert sha(p)==sim['native_before_sha256'];meta=load(mp);union=F/meta['union_resolution']['receipt'];stage=F/'ordered_current_semantic_component_stage.json';stage_complete=G/'operations/ordered_source_stage_v107/frame001710/complete.json';rq=F/'role_unknown_visibility_inputs_v135/JE/receipt.json';rc=rq.with_name('complete.json');ra=rq.with_name('CURRENT_NN_role_UNKNOWN.npz')
 # No active old source descendants may be writing; source marker is withheld.
 hold=load(G/'operations/1710_JE12226_current_actor_review_hold.json')
 for r in hold['records']:
  proc=Path(f'/proc/{r["pid"]}')
  if proc.exists():
   s=(proc/'stat').read_text();fields=s[s.rfind(')')+2:].split();assert fields[0]in['T','Z']and fields[19]==r['start']
 before=O/'before_application';before.mkdir(exist_ok=True)
 files=[p,mp,union,stage,stage_complete,rq,rc,ra,G/'operations/current_native_parts_v105/frame001710/qualification.json']
 inventory={str(x):sha(x)for x in files};write(before/'original_files_sha256.json',inventory)
 for index,x in enumerate(files):shutil.copy2(x,before/(str(index)+'_'+x.name))
 for folder,name in[(F/'qualified_hair_inputs/JE','qualified_RGB'),(F/'role_unknown_visibility_inputs_v135/JE','ROLE_UNKNOWN')]:
  if folder.exists():shutil.copytree(folder,before/name,dirs_exist_ok=True)
 unchanged={str(x):sha(x)for x in(F/'native_masks').glob('*.npz')if x.stem!='JE'};rgb={c:sha(load(G/'sources/1710'/c/'receipt.json')['image'])for c in load(G/'source_inventory.json')['cameras']};assert len(unchanged)==65 and len(rgb)==66
 with np.load(p)as z:old={k:z[k].copy()for k in z.files}
 assert not(R&(old['black']|old['grey']|old['qualified_actor_rgb'])).any()
 Q=O/'finite_apply_scope';Q.mkdir(exist_ok=True);write(Q/'qualification.json',dict(kind='ROOT_FINITE_CURRENT_PHOTO_NATIVE_NN_ACTOR_ROLE_BOOTSTRAP',frame=1710,input_native_sha256={'JE':sha(p)},qualified_archives={'JE':dict(path=str(O/'exact_CURRENT_NN_positive.npz'),sha256=sim['proposal_archive_sha256'],array_sha256=array_sha(R),pixels=12226)},ROOT_authorization=str(AUTH),ROOT_authorization_sha256=sha(AUTH)))
 import marshawn_current_parts_apply_v105 as application
 application.V=G;proof=application.apply_one(1710,'JE',Q)
 with np.load(p)as z:new={k:z[k].copy()for k in z.files}
 expected={k:v.copy()for k,v in old.items()};expected['black']|=R;expected['qualified_actor_rgb']|=R;expected['actor_ownership_uncertain']|=R;expected['source_ineligible']&=~R
 assert set(expected)==set(new)
 for k in new:assert np.array_equal(expected[k],new[k]),'Unexpected native array delta '+k
 # Supersede only this reviewed warning; keep the original NN decision immutable.
 original=load(rq);npz(ra,role_unknown=left,soft_eligible=soft);role=dict(original);role.update(new_native_sha256=sha(p),scope_array_sha256=array_sha(left),scope_pixels=74,scope_archive_sha256=sha(ra),soft_eligible_current_refined_PRE0_pixels=int(soft.sum()),NN_role_proofs=[x for x in original['NN_role_proofs']if x['archive']!=load(J)['exact_current_NN_scope_archive']],finite_current_positive_supersession=dict(ROOT_authorization=str(AUTH),ROOT_authorization_sha256=sha(AUTH),scope_array_sha256=array_sha(R),scope_pixels=12226,original_role_receipt_sha256=inventory[str(rq)]));write(rq,role);write(rc,role)
 ext=O/'finite_current_actor_extension.json';write(ext,dict(kind='CURRENT_NN_ROOT_PHOTO_QUALIFIED_ACTOR_ROLE_EXTENSION',frame=1710,camera='JE',current_NN_scope_sha256=array_sha(R),scope_pixels=12226,ROOT_authorization_sha256=sha(AUTH),original_qualification_sha256=inventory[str(files[-1])],original_qualification_decisions_preserved=True,original_ordered_stage_sha256=inventory[str(stage)],old_native_sha256=sim['native_before_sha256'],new_native_sha256=sha(p),application_receipt=str(Q/'applications/JE/complete.json'),application_receipt_sha256=sha(Q/'applications/JE/complete.json'),remaining_ROLE_UNKNOWN_pixels=74,remaining_ROLE_UNKNOWN_receipt_sha256=sha(rq),original_geometry_UNKNOWN_and_all_old_actor_pixels_preserved=True,created=time.time()))
 meta=load(mp);meta['ROLE_UNKNOWN_visibility']['receipt_sha256']=sha(rq);meta['finite_photo_role_extension']=dict(receipt=str(ext),sha256=sha(ext));write(mp,meta)
 current=load(stage);current['native_after_application_sha256']['JE']=sha(p);current['role_unknown_visibility']=[role if r['camera']=='JE'else r for r in current['role_unknown_visibility']];current['finite_current_photo_role_extensions']=current.get('finite_current_photo_role_extensions',[])+[dict(path=str(ext),sha256=sha(ext))];current['ROLEUNKNOWN_consistency_scope_after_finite_positive']=dict(JE=74,DE=126);write(stage,current);write(stage_complete,current)
 validate_union(G,1710,'JE')
 for path,digest in unchanged.items():assert sha(path)==digest
 for c,digest in rgb.items():assert sha(load(G/'sources/1710'/c/'receipt.json')['image'])==digest
 assert sha(G/'operations/current_native_parts_v105/frame001710/qualification.json')==inventory[str(files[-1])]
 write(O/'application_complete.json',dict(frame=1710,camera='JE',status='pass',new_native_sha256=sha(p),native_metadata_sha256=sha(mp),exact_new_actor_pixels=12226,all_native_arrays_exact_expected=True,all65_other_native_archives_unchanged=unchanged,all66_RGB_unchanged=rgb,remaining_JE_ROLE_UNKNOWN_pixels=74,physical_UNKNOWN_unchanged=True,original_NN_qualification_unchanged=True,ordered_finite_extension_sha256=sha(ext),ordered_current_stage_sha256=sha(stage),ROOT_authorization_sha256=sha(AUTH),fresh_hair_all66_guard_DATA_QA_still_required=True,source_marker_not_published=True,completed=time.time()));print(str(O/'application_complete.json'),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args();apply()if args.apply else plan()
