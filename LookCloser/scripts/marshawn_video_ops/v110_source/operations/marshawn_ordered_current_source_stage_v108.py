"""Same immutable preapplication native snapshot for semantic and closure stages."""
from pathlib import Path
import sys,json,time,concurrent.futures
import numpy as np
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_union_resolution_v5 import array_sha
from marshawn_union_guard_v53 import validate_union
import marshawn_current_parts_v105 as semantic
import marshawn_current_parts_apply_v105 as semantic_scope
import marshawn_ordered_union_apply_v107 as application
from marshawn_current_component_closure_v106 import propose
from semantic_CPU8_v1.marshawn_semantic_qualification_cpu8 import qualify as parallel_semantic_qualify
from marshawn_cpu_worker_init_v108 import initialize as cpu_worker_initialize

def closure_one(args):
 frame,cam,archive,output=args;return propose(frame,cam,[archive]if archive else[],output)
def run(frame):
 f=V/'frames'/f'{frame:06d}';marker=f/'ordered_current_semantic_component_stage.json';O=V/'operations/ordered_source_stage_v107'/f'frame{frame:06d}';O.mkdir(parents=True,exist_ok=True)
 if marker.exists():
  r=json.loads(marker.read_text());assert r['status']=='pass'
  for cam,digest in r['native_after_application_sha256'].items():assert sha(f/'native_masks'/(cam+'.npz'))==digest
  return r
 golden=V/'operations/ordered_closure_golden_v106/PASS.json';assert json.loads(golden.read_text())['status']=='pass'
 cameras=json.loads((V/'source_inventory.json').read_text())['cameras'];Q=O/'qualification.json'
 if not Q.exists():
  SO,records=semantic.plans(frame,cameras);semantic.launch(frame,SO,records)
  if records:parallel_semantic_qualify(frame,SO/'bindings.json',root=V,workers=8);sq=json.loads((SO/'qualification.json').read_text())
  else:sq=dict(qualified_archives={})
  args=[(frame,c,sq['qualified_archives'].get(c,{}).get('path'),O/'closures')for c in cameras]
  with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=cpu_worker_initialize)as pool:closures=list(pool.map(closure_one,args))
  qualified={};inputs={};proofs=[]
  for r in closures:
   cam=r['camera'];p=f/'native_masks'/(cam+'.npz');inputs[cam]=sha(p);assert r['current_native_sha256']==sha(p)
   with np.load(r['qualified_archive'])as z:H=z['qualified_current_component_union'].copy()
   if cam in sq['qualified_archives']:
    source=sq['qualified_archives'][cam];assert sha(source['path'])==source['sha256']
    with np.load(source['path'])as z:H|=z['qualified_new_union']
   with np.load(p)as z:H&=~z['qualified_actor_rgb']if'qualified_actor_rgb'in z.files else True
   archive=O/(cam+'_qualified_current_scope.npz');np.savez_compressed(archive,qualified_new_union=H);qualified[cam]=dict(path=str(archive),sha256=sha(archive),array_sha256=array_sha(H),pixels=int(H.sum()));proofs.append(dict(camera=cam,path=str(O/'closures'/(cam+'_proof.json')),sha256=sha(O/'closures'/(cam+'_proof.json'))))
  write(Q,dict(kind='ORDERED_CURRENT_SEMANTIC_AND_COMPONENT_UNION_QUALIFICATION_V107',frame=frame,input_native_sha256=inputs,qualified_archives=qualified,current_semantic_qualification=str(SO/'qualification.json')if records else None,current_semantic_qualification_sha256=sha(SO/'qualification.json')if records else None,closure_proofs=proofs,golden_sha256=sha(golden),same_preapplication_native_for_both_stages=True,source_only_current_NN=True,previous_union_identity_and_ROI_only=True,created=time.time()))
 q=json.loads(Q.read_text());applied=[]
 for cam,r in q['qualified_archives'].items():
  if r['pixels']:applied.append(application.apply_one(frame,cam,O))
  validate_union(V,frame,cam)
 result=dict(frame=frame,status='pass',ordered_qualification_sha256=sha(Q),current_semantic_NN_stage=True,current_component_closure_stage=True,same_preapplication_native_for_both=True,all66_union_conservation_pass=True,native_after_application_sha256={cam:sha(f/'native_masks'/(cam+'.npz'))for cam in cameras},golden_sha256=sha(golden),qualification=str(Q),current_NN_only_no_prior_bitmap_as_FG=True,created=time.time());write(marker,result);write(O/'complete.json',result);return result
