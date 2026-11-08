from pathlib import Path
import os,sys,json,time,fcntl,traceback
import numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_video_producer_v3 import native_inventory
from marshawn_union_guard_v53 import validate_union

def load(p):return json.loads(Path(p).read_text())
def publish(frame):
 f=G/'frames'/f'{frame:06d}';origin=V/'frames'/f'{frame:06d}';data=f/'data';ready_review=f/'generation_prepared_REVIEW_READY.json';marker=f/'native_semantic_ready.json'
 if marker.exists():assert load(marker)['prepared_ready_sha256']==sha(data/'prepared_ready.json');return
 r=load(ready_review);assert r['status']=='pass'and r['prepared_ready_sha256']==sha(data/'prepared_ready.json')and r['ready_sha256']==sha(data/'ready.json');q=load(data/'final_v5_source_QA.json');assert q['status']=='pass'and q['prepared_ready_sha256']==sha(data/'prepared_ready.json')and q['ready_sha256']==sha(data/'ready.json');assert load(data/'visibility_unknown_QA_v74.json')['status']=='pass';guard=f/'crew_guard/plan_receipt.json';assert load(guard)['status']=='pass'
 cameras=load(G/'source_inventory.json')['cameras'];inventory=native_inventory(G,frame,cameras);ready=load(data/'ready.json');oldready=load(origin/'data/ready.json');unchanged={}
 for c in cameras:
  validate_union(G,frame,c);current=next(x for x in ready['records']if x['camera']==c);old=next(x for x in oldready['records']if x['camera']==c);assert current['rgb_sha256']==old['rgb_sha256']==sha(data/current['image']);assert sha(current['semantic'])==current['semantic_sha256']
  if c!='GC':
   assert inventory[c]['archive_sha256']==sha(origin/'native_masks'/(c+'.npz'))
   with np.load(current['semantic'])as a,np.load(old['semantic'])as b:
    assert set(a.files)==set(b.files),(c,a.files,b.files)
    for key in a.files:assert np.array_equal(a[key],b[key]),(c,key)
   unchanged[c]=dict(native_sha256=inventory[c]['archive_sha256'],new_semantic_sha256=current['semantic_sha256'],original_semantic_sha256=old['semantic_sha256'],ALL_semantic_arrays_bitwise_identical=True)
  row=load(f/'crew_guard'/(c+'_people.json'));assert row['current_native_archive_sha256']==inventory[c]['archive_sha256']and row['current_candidate_archive_sha256']==inventory[c]['person_candidates_sha256']
 hair=data/'admission_unknown_hair/manifest.json';assert load(data/'prepared_ready.json')['admission_unknown_hair_manifest_sha256']==sha(hair)
 for p,d in load(f/'generation_origin.json')['original_files_sha256'].items():assert sha(origin/p)==d
 invariant=f/'generation_corrections/65_camera_DATA_invariance.json';write(invariant,dict(frame=frame,cameras=65,records=unchanged,GC_retraction_only_current_native_source_change=True,same_frame_RGB_ALL66_byte_identical=True,all_original_published_sources_unchanged=True,original_NN_inference_metadata_not_restamped=True,created=time.time()))
 release=G/'operations/source_generation_release.json'
 write(marker,dict(kind='SOURCE_BOUND_NATIVE_ACTOR_SEMANTICS_READY_V1',frame=frame,cameras=66,source_ready_sha256=sha(G/'sources'/str(frame)/'source_ready.json'),prepared_ready_sha256=sha(data/'prepared_ready.json'),semantic_ready_sha256=sha(data/'ready.json'),crew_guard_receipt_sha256=sha(guard),native_masks=inventory,common_box_sha256=sha(G/'common_box.json'),no_cross_frame_NeRF_state=True,visibility_grid_dispatch_deferred=True,operations_release_sha256=sha(release),source_QA_sha256=sha(data/'final_v5_source_QA.json'),visibility_UNKNOWN_QA_sha256=sha(data/'visibility_unknown_QA_v74.json'),admission_unknown_hair_manifest_sha256=sha(hair),required_router='NEW_C_F10_140000_CAP_ONLY',source_revision='ROOT_SCOPED_GCROLE_G1_CURRENT_SOURCE_CORRECTION',source_generation='gcrole_g1',source_generation_root=str(G),origin_published_source_marker_sha256=sha(origin/'native_semantic_ready.json'),role_retraction_receipt_sha256=sha(f/'generation_corrections/GC_visible_CREW_hand/complete.json'),other65_DATA_invariance_sha256=sha(invariant),NO_FIELD_BEFORE_NEW_GRID_AND_DENSE=True,NO_FIELD_BEFORE_ROOT_CORRECTED_GENERATION_GRID_REVIEW=True,human_source_review_claim='ROOT personally identified wrong CREW hand; actual new native before/delta/after contact bound below',generation_native_contact=str(G/'operations/GC_corrected_native_visual'/f'GC{frame}_native_before_delta_after.jpg'),generation_native_contact_sha256=sha(G/'operations/GC_corrected_native_visual'/f'GC{frame}_native_before_delta_after.jpg'),routine_source_policy_authorized_by_ROOT=False,finite_source_repair_authorized_by_ROOT=True,created=time.time()));print(frame,'SOURCE_PUBLISHED',sha(marker),flush=True)
if __name__=='__main__':
 lock=open(G/'operations/source_generation_publisher.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for frame in[1695,1700,1705]:
  while not(G/'frames'/f'{frame:06d}'/'generation_prepared_REVIEW_READY.json').exists():time.sleep(5)
  try:publish(frame)
  except Exception as e:write(G/'operations/source_generation_publish_error.json',dict(frame=frame,error=str(e),traceback=traceback.format_exc(),created=time.time()));raise
 write(G/'operations/source_generation_published.json',dict(frames=[1695,1700,1705],status='pass',all_new_grid_dense_and_ROOT_grid_review_required=True,created=time.time()))
