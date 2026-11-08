"""G1 context and role policy hooks for the original source factory."""
from pathlib import Path
import sys,json
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');BASE=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(G/'operations'),str(BASE/'operations'),str(BASE/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha
from marshawn_clean_source_context_v134 import configure as configure_context

def configure():
 configure_context()
 import marshawn_ordered_current_source_stage_v118 as old_ordered
 import marshawn_ordered_source_role_stage_v136 as role_ordered
 import marshawn_source_factory_v118 as factory
 import marshawn_role_visibility_v137 as visibility
 old_ordered.run=role_ordered.run
 factory.camera_guard=camera_guard
 factory.validate_DATA=visibility.validate_DATA
 return factory

def camera_guard(args):
 frame,camera=args;configure()
 import crew_guard_plan_v53 as m
 import marshawn_current_parts_v105 as semantic
 from marshawn_role_visibility_v137 import resolve
 semantic.V=G;semantic.CREW_CACHE.clear();policy=json.loads((G/'operations/visibility_unknown_policy_v107.json').read_text());policy['policy_sha256']=sha(G/'operations/visibility_unknown_policy_v107.json');original=m.write
 def output(path,value):
  if Path(path).name in['plan_receipt.json','identity_novelty_pass.json']:return
  if Path(path).name==camera+'_people.json':value=resolve(G,frame,value,policy)
  original(path,value)
 code=Path(m.__file__).read_text().replace('qualified_actor=np.zeros((270,512),bool);qualified_instances={};fragment_proofs=[]','qualified_actor=small(qualified_actor_rgb);qualified_instances={};fragment_proofs=[]').replace("print(camera,report['status'],unknown,residual_n,flush=True)","print(camera,report['status'],report['unknown_person_instances'],residual_n,flush=True)");scope=dict(m.__dict__);exec(compile(code,str(G/'operations/crew_guard_G1_ROLE_runtime_v138.py'),'exec'),scope);scope['write']=output;scope['run'](G,frame,[camera],True);return json.loads((G/'frames'/f'{frame:06d}'/'crew_guard'/(camera+'_people.json')).read_text())
