"""Finite reviewed CURRENT NN union recovery and exact283 RAW-proposal accounting.
No implementation, generic policy, physical UNKNOWN, RGB or other-camera edit.
"""
from pathlib import Path
import argparse,fcntl,json,shutil,sys,time
import numpy as np,cv2
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(G/'operations'),str(G/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union
O=G/'operations/1715_FD_components_current_positive';F=G/'frames/001715';J=G/'operations/ROOT_1715_FD8_photo_role_judgment.json';AUTH=O/'ROOT_application_authorization.json'
def load(p):return json.loads(Path(p).read_text())
def scope_inputs():
 from crew_guard_plan_v53 import candidates
 s=load(O/'scope_proof.json');j=load(J);assert j['positive_component_ids']==[1,2,4,7,9,10,11]and j['preserve_unchanged_component_ids']==[3,5,6,8]
 for path,digest in j['files_sha256'].items():assert sha(path)==digest
 cp,_=candidates(G,1715,'FD');assert sha(cp)==s['candidate_archive_sha256']and sha(F/'native_masks/FD.npz')==s['native_before_sha256']
 assert sha(s['proposal_archive'])==s['proposal_archive_sha256']and sha(s['photo_RAW_PRE283_archive'])==s['photo_RAW_PRE283_archive_sha256']
 with np.load(cp)as z:_,labels=cv2.connectedComponents(z['masks'][8].astype('uint8'),8)
 with np.load(F/'native_masks/FD.npz')as z:A=z['black']|z['grey']|z['qualified_actor_rgb'];unknown=z['unknown'];veto=z['source_ineligible']
 with np.load(s['proposal_archive'])as z:R=z['qualified_new_union'].copy()
 with np.load(s['photo_RAW_PRE283_archive'])as z:P=z['unresolved_RAW_proposal'].copy()
 assert np.array_equal(R,np.isin(labels,[1,2,4,7,9,10,11])&~A)and int(R.sum())==s['new_actor_union_pixels']and array_sha(R)==s['exact_scope_array_sha256']
 assert np.array_equal(P,np.isin(labels,[5,6]))and P.sum()==283 and array_sha(P)==s['photo_RAW_PRE283_array_sha256']and not(P&A).any()and unknown[P].all()and veto[P].all()
 assert s['independent_CURRENT_CREW_overlap_new']==0 and s['role_UNKNOWN_original_pixels']==0
 assert not(F/'role_unknown_visibility_inputs_v135/FD/receipt.json').exists(),'Unexpected FD ROLE_UNKNOWN state'
 assert sha(s['refined_PRE_archive'])==s['refined_PRE_archive_sha256']
 with np.load(s['refined_PRE_archive'])as z:assert not(P&(z['pre_black']|z['pre_grey'])).any()
 return R,P,labels,s

def plan():
 R,P,labels,s=scope_inputs();print(json.dumps(dict(applier=__file__,applier_sha256=sha(__file__),scope_proof_sha256=sha(O/'scope_proof.json'),photo_judgment_sha256=sha(J),new_union_pixels=int(R.sum()),independent_CURRENT_CREW_overlap=0,guard_accounting_only_RAW_PRE_pixels=int(P.sum()),no_arrays_changed=True),indent=2),flush=True)

def apply():
 a=load(AUTH);assert a['authorized']and a['frame']==1715 and a['camera']=='FD'and a['applier_sha256']==sha(__file__)and a['photo_judgment_sha256']==sha(J)and a['simulation_sha256']==sha(O/'scope_proof.json')
 lock=open(G/'operations/source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(O/'application_complete.json').exists():
  done=load(O/'application_complete.json');assert done['status']=='pass'and sha(F/'native_masks/FD.npz')==done['new_native_sha256'];validate_union(G,1715,'FD');return
 assert not(F/'native_semantic_ready.json').exists()and not(F/'dispatch_ready.json').exists()
 retired=load(O/'worker_retirement.json');assert retired['all_live_writing_descendants_absent']
 for r in retired['descendants']+[retired['worker']]:
  p=Path('/proc')/str(r['pid'])
  if p.exists():
   stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split();assert fields[19]!=r['start']or fields[0]=='Z','Old writer still alive'
 R,P,labels,s=scope_inputs();p=F/'native_masks/FD.npz';mp=p.with_suffix('.json');meta=load(mp);union=F/meta['union_resolution']['receipt'];stage=F/'ordered_current_semantic_component_stage.json';stage_complete=G/'operations/ordered_source_stage_v107/frame001715/complete.json';qualification=G/'operations/current_native_parts_v105/frame001715/qualification.json'
 before=O/'before_application';before.mkdir(exist_ok=True);files=[p,mp,union,stage,stage_complete,qualification];inventory={str(x):sha(x)for x in files};write(before/'original_files_sha256.json',inventory)
 for i,x in enumerate(files):shutil.copy2(x,before/(str(i)+'_'+x.name))
 if(F/'qualified_hair_inputs/FD').exists():shutil.copytree(F/'qualified_hair_inputs/FD',before/'qualified_RGB',dirs_exist_ok=True)
 unchanged={str(x):sha(x)for x in(F/'native_masks').glob('*.npz')if x.stem!='FD'};rgb={c:sha(load(G/'sources/1715'/c/'receipt.json')['image'])for c in load(G/'source_inventory.json')['cameras']};assert len(unchanged)==65 and len(rgb)==66
 with np.load(p)as z:old={k:z[k].copy()for k in z.files}
 Q=O/'finite_apply_scope';Q.mkdir(exist_ok=True);write(Q/'qualification.json',dict(kind='ROOT_FINITE_CURRENT_PHOTO_NATIVE_NN_ACTOR_ROLE_BOOTSTRAP',frame=1715,input_native_sha256={'FD':sha(p)},qualified_archives={'FD':dict(path=s['proposal_archive'],sha256=s['proposal_archive_sha256'],array_sha256=array_sha(R),pixels=int(R.sum()))},ROOT_authorization=str(AUTH),ROOT_authorization_sha256=sha(AUTH)))
 import marshawn_current_parts_apply_v105 as application
 application.V=G;application.apply_one(1715,'FD',Q)
 with np.load(p)as z:new={k:z[k].copy()for k in z.files}
 expected={k:v.copy()for k,v in old.items()};expected['black']|=R;expected['qualified_actor_rgb']|=R;expected['actor_ownership_uncertain']|=R;expected['source_ineligible']&=~R
 assert set(new)==set(expected)
 for k in new:
  assert np.array_equal(new[k],expected[k]),'Unexpected array delta '+k
  if new[k].shape==labels.shape:assert np.array_equal(new[k][np.isin(labels,[3,5,6,8])],old[k][np.isin(labels,[3,5,6,8])]),'Unapproved component changed '+k
 # Original137 validates this exact scope then subtracts it from RAW proposal
 # conflict accounting only. Physical UNKNOWN/veto remain1; no grid edit.
 guard=F/'guard_proposal_uncertainty_inputs/FD';guard.mkdir(parents=True,exist_ok=True);shutil.copy2(s['photo_RAW_PRE283_archive'],guard/'mask.npz')
 proof=dict(kind='ROOT_SCOPED_RAW_PROPOSAL_UNCERTAINTY_ONLY',frame=1715,camera='FD',instance=8,source_receipt_sha256=s['source_receipt_sha256'],candidate_archive_sha256=s['candidate_archive_sha256'],native_archive_sha256=sha(p),mask_archive_sha256=sha(guard/'mask.npz'),mask_array_sha256=array_sha(P),scope_pixels=283,refined_PRE_archive_sha256=s['refined_PRE_archive_sha256'],ROOT_approval_path=str(AUTH),ROOT_approval_sha256=sha(AUTH),ROOT_photo_judgment=str(J),ROOT_photo_judgment_sha256=sha(J),scope_components=[5,6],photo_resolution='Visible brown calibration cardboard; RAW-selected proposal false-positive, refined PRE0',original_RAW_PRE_preserved=True,no_native_DATA_RGB_confidence_or_physical_UNKNOWN_changed_by_accounting=True,NOT_physical_free_space_certificate=True,created=time.time());write(guard/'proof.json',proof)
 ext=O/'finite_current_actor_extension.json';write(ext,dict(kind='CURRENT_NN_ROOT_PHOTO_QUALIFIED_ACTOR_ROLE_EXTENSION',frame=1715,camera='FD',positive_components=[1,2,4,7,9,10,11],current_NN_scope_sha256=array_sha(R),scope_pixels=int(R.sum()),ROOT_authorization_sha256=sha(AUTH),original_qualification_sha256=inventory[str(qualification)],original_qualification_decisions_preserved=True,original_ordered_stage_sha256=inventory[str(stage)],old_native_sha256=s['native_before_sha256'],new_native_sha256=sha(p),application_receipt=str(Q/'applications/FD/complete.json'),application_receipt_sha256=sha(Q/'applications/FD/complete.json'),guard_accounting_proof_sha256=sha(guard/'proof.json'),original_geometry_UNKNOWN_and_all_old_actor_pixels_preserved=True,created=time.time()))
 meta=load(mp);meta['finite_photo_role_extension']=dict(receipt=str(ext),sha256=sha(ext));meta['finite_RAW_proposal_accounting']=dict(receipt=str((guard/'proof.json').relative_to(F)),sha256=sha(guard/'proof.json'));write(mp,meta)
 current=load(stage);current['native_after_application_sha256']['FD']=sha(p);current['finite_current_photo_role_extensions']=current.get('finite_current_photo_role_extensions',[])+[dict(path=str(ext),sha256=sha(ext))];write(stage,current);write(stage_complete,current)
 validate_union(G,1715,'FD')
 for path,digest in unchanged.items():assert sha(path)==digest
 for c,digest in rgb.items():assert sha(load(G/'sources/1715'/c/'receipt.json')['image'])==digest
 assert sha(qualification)==inventory[str(qualification)]
 write(O/'application_complete.json',dict(frame=1715,camera='FD',status='pass',new_native_sha256=sha(p),native_metadata_sha256=sha(mp),exact_new_actor_pixels=int(R.sum()),all_native_arrays_exact_expected=True,unchanged_components=[3,5,6,8],unchanged_component3_existing_actor_H_pixels=4155,RAW_PRE_photo283_accounting_proof_sha256=sha(guard/'proof.json'),all65_other_native_archives_unchanged=unchanged,all66_RGB_unchanged=rgb,physical_UNKNOWN_unchanged=True,original_NN_qualification_unchanged=True,ordered_finite_extension_sha256=sha(ext),ordered_current_stage_sha256=sha(stage),ROOT_authorization_sha256=sha(AUTH),fresh_hair_all66_guard_DATA_QA_still_required=True,source_marker_not_published=True,completed=time.time()));print(str(O/'application_complete.json'),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');a=parser.parse_args();apply()if a.apply else plan()
