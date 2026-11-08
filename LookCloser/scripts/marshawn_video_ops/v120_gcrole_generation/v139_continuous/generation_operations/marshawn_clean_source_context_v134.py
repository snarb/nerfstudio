"""Explicit source-context parameters for released read/processing functions.

Immutable original implementations remain untouched. Target/native/CREW/2D-flow
and proposal outputs use G1; old RAW, original own-frame NN, calibrated camera
source and GPU scheduler/cache are permitted only as explicit immutable/shared
inputs. This module does not start NN, train, install masks or publish DATA.
"""
from pathlib import Path
import sys,importlib,json,time
BASE=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
ROOT=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108')
sys.path[:0]=[str(BASE/'operations'),str(BASE/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
TARGET_MODULES=[
 'marshawn_head_queue_v90','marshawn_current_parts_v105',
 'marshawn_current_parts_v96','marshawn_current_component_closure_v118',
 'marshawn_ordered_current_source_stage_v118','marshawn_ordered_union_apply_v107',
 'marshawn_current_parts_apply_v105','marshawn_source_factory_v118',
]
def configure(root=ROOT):
 root=Path(root);assert root==ROOT
 modules={}
 for name in TARGET_MODULES:
  m=importlib.import_module(name);original=str(getattr(m,'V',None));m.V=root
  if hasattr(m,'CREW_CACHE'):m.CREW_CACHE.clear()
  assert m.V==root;modules[name]=dict(source=str(Path(m.__file__)),source_sha256=sha(m.__file__),V=str(m.V),original_default_V=original)
 # Explicit spawn initializer: parent module globals are never assumed to
 # survive a new interpreter. These actual call sites resolve this function.
 import marshawn_ordered_current_source_stage_v118 as ordered
 import marshawn_source_factory_v118 as factory
 ordered.cpu_worker_initialize=initialize_child
 factory.cpu_worker_initialize=initialize_child
 return modules

def initialize_child():
 from marshawn_cpu_worker_init_v108 import initialize
 initialize()
 configure()

def record():
 modules=configure();tracker=importlib.import_module('marshawn_union_tracking_proposals_v1')
 assert sha(tracker.__file__)=='065a3ef88a6a3b72a904b4d1764761b7f7b65e7e64f950c5b68c8484e75f6f6f'
 write(ROOT/'operations/1710_inference_adoption/context_audit_v134.json',dict(kind='EXPLICIT_G1_TARGET_CONTEXT_OVERLAY_NOT_PRODUCER_ACTIVATION',target_root=str(ROOT),target_modules=modules,explicit_root_helpers=['marshawn_union_tracking_proposals_v1','marshawn_identity_refresh_v79','crew_guard_plan_v53','marshawn_actor_conflict_v51','marshawn_hair_admission_layer_v88','marshawn_video_prep_data','marshawn_semantic_ownership_v5','marshawn_final_source_QA_v7'],allowed_original_contexts=dict(RAW=str(ROOT/'sources')+' -> original hash-bound RAW',original_same_frame_NN='Original spec/receipts preserved and input-equivalence bound; no inference restamp',calibration='Frozen physical66camera SOURCE transforms/masks, no actor/current-frame mask authority',remote_SV='Original own-frame PNG/NN transport and checkpoint namespace; new reference blobs supplied from G1',GPU_controls='Shared global leases/fleet pointer/native requests/cache, no actor/native read authority'),all_target_prior_native_CREW_complete_union_and_proposals_route_G1=True,spawn_initializers_explicitly_context_aware=True,NO_producer_or_SEM_filter_activation=True,created=time.time()))
if __name__=='__main__':record()
