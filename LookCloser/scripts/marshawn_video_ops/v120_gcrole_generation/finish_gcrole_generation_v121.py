from pathlib import Path
import sys,os,json,shutil,fcntl,time,subprocess,concurrent.futures,multiprocessing,traceback
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_guard_v53 import validate_union
from marshawn_cpu_worker_init_v108 import initialize

def load(p):return json.loads(Path(p).read_text())
def state(frame,stage,**kw):write(G/'operations/corrected_source_generation_state.json',dict(frame=frame,stage=stage,pid=os.getpid(),time=time.time(),**kw))
def guard(args):
 import marshawn_source_factory_v118 as factory
 factory.V=G
 return factory.camera_guard(args)
def call(frame,stage,argv):
 state(frame,stage,argv=argv);p=G/'operations'/f'{frame}_{stage}.log'
 with p.open('ab')as log:subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONPATH=str(V/'operations')+':'+str(V/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
def finish(frame):
 f=G/'frames'/f'{frame:06d}';done=f/'generation_prepared_REVIEW_READY.json'
 if done.exists():
  d=load(done);assert sha(f/'data/prepared_ready.json')==d['prepared_ready_sha256'];return
 cameras=load(G/'source_inventory.json')['cameras']
 for c in cameras:validate_union(G,frame,c)
 stage=f/'ordered_current_semantic_component_stage.json'
 if not stage.exists():
  origin_stage=V/'frames'/f'{frame:06d}'/'ordered_current_semantic_component_stage.json';original=load(origin_stage);assert original['status']=='pass'
  for c in cameras:
   if c!='GC':assert sha(f/'native_masks'/(c+'.npz'))==original['native_after_application_sha256'][c]
  write(stage,dict(kind='GENERATION_ORDERED_CURRENT_NN_STAGE_ADOPTION_AFTER_EXACT_ROOT_ROLE_RETRACTION',frame=frame,status='pass',generation='gcrole_g1',original_ordered_stage=str(origin_stage),original_ordered_stage_sha256=sha(origin_stage),GC_exact_role_correction_receipt=str(f/'generation_corrections/GC_visible_CREW_hand/complete.json'),GC_exact_role_correction_receipt_sha256=sha(f/'generation_corrections/GC_visible_CREW_hand/complete.json'),native_after_application_sha256={c:sha(f/'native_masks'/(c+'.npz'))for c in cameras},all_other65_current_NN_arrays_exact=True,new_NN_execution_NOT_claimed=True,no_foreground_added_by_adoption=True,created=time.time()))
 guarddone=f/'generation_guard_complete.json'
 if not guarddone.exists():
  state(frame,'fresh_current_source_guard_66')
  with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=initialize,mp_context=multiprocessing.get_context('spawn'))as pool:reports=list(pool.map(guard,[(frame,c)for c in cameras]))
  status='pass'if all(r['status']=='pass'for r in reports)else'unknown';p=f/'crew_guard/plan_receipt.json';write(p,dict(frame=frame,cameras=66,status=status,reports=[dict(camera=r['camera'],status=r['status'],unknown=r['unknown_person_instances'],actor_novelty=r['actor_novelty_trigger'])for r in reports],generation='gcrole_g1',current_H_novelty_qualified=True,conservative_PRE0_existing_visibility_UNKNOWN_policy_sha256=sha(G/'operations/visibility_unknown_policy_v107.json'),no_prior_bitmap_final_pixels=True))
  if status!='pass':raise RuntimeError('Corrected generation current guard unresolved '+str([(r['camera'],r['unknown_person_instances'])for r in reports if r['status']!='pass']))
  write(guarddone,dict(frame=frame,status='pass',guard_sha256=sha(p),current_native_sha256={c:sha(f/'native_masks'/(c+'.npz'))for c in cameras}))
 if not(f/'data/prepared_ready.json').exists():call(frame,'frozen_DATA',[sys.executable,str(V/'code/LookCloser/scripts/marshawn_video_prep_data.py'),'--root',str(G),'--frame',str(frame),'--box',str(G/'common_box.json')])
 if not(f/'data/semantic_ownership_v5.json').exists():call(frame,'ownership',[sys.executable,str(V/'operations/marshawn_semantic_ownership_v5.py'),'--root',str(G),'--frame',str(frame)])
 from marshawn_hair_admission_layer_v88 import build,bind
 from marshawn_visibility_unknown_v110 import validate_DATA
 manifest=f/'data/admission_unknown_hair/manifest.json'
 if not manifest.exists():
  state(frame,'current_head_UNKNOWN_recompute_owned_generation');old=V/'frames'/f'{frame:06d}'/'data/admission_unknown_hair';om=load(old/'manifest.json');observations={}
  for row in om['records']:
   observations[row['camera']]=[dict(proposal=str(old/p['directory']/'current_native_proposal.npz'),qualification=str(old/p['directory']/'qualification_original.json'))for p in row['proofs']]
  build(G,frame,observations,G/'operations/head_hair_policy_v74.json')
 bind(G,frame);validate_DATA(G,frame)
 call(frame,'source_QA',[sys.executable,str(V/'operations/marshawn_final_source_QA_v7.py'),'--root',str(G),'--frame',str(frame)])
 call(frame,'native_visual',[sys.executable,str(V/'operations/marshawn_native_review.py'),'--root',str(G),'--frame',str(frame)])
 original=V/'frames'/f'{frame:06d}';originals=load(f/'generation_origin.json')['original_files_sha256']
 for p,d in originals.items():assert sha(original/p)==d,'Original published generation changed '+p
 d=dict(kind='GCROLE_G1_CORRECTED_SOURCE_REVIEW_READY',frame=frame,cameras=66,source_root=str(G),status='pass',prepared_ready_sha256=sha(f/'data/prepared_ready.json'),ready_sha256=sha(f/'data/ready.json'),source_QA_sha256=sha(f/'data/final_v5_source_QA.json'),visibility_QA_sha256=sha(f/'data/visibility_unknown_QA_v74.json'),head_UNKNOWN_manifest_sha256=sha(manifest),GC_correction_proof_sha256=sha(f/'generation_corrections/GC_visible_CREW_hand/complete.json'),all_original_published_files_unchanged=True,source_marker_withheld_until_ROOT_generation_review=True,no_learned_state_reused=True,created=time.time());write(done,d);state(frame,'review_ready',receipt=str(done));print(frame,'REVIEW_READY',flush=True)
def main():
 G.joinpath('operations').mkdir(exist_ok=True)
 for n in['visibility_unknown_policy_v107.json','head_hair_policy_v74.json']:shutil.copy2(V/'operations'/n,G/'operations'/n)
 lock=open(G/'operations/source_generation_writer.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  assert load(G/'operations/finite_GC_role_corrections_complete.json')['status']=='pass'
  for frame in[1695,1700,1705]:finish(frame)
  write(G/'operations/corrected_source_generation_REVIEW_READY.json',dict(frames=[1695,1700,1705],status='pass',source_marker_withheld=True,created=time.time()))
 except Exception as e:
  write(G/'operations/corrected_source_generation_error.json',dict(error=str(e),traceback=traceback.format_exc(),created=time.time()));raise
if __name__=='__main__':main()
