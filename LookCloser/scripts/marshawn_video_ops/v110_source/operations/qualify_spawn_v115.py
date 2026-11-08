from pathlib import Path
import sys,json,ast,multiprocessing,concurrent.futures,time
import numpy as np,cv2
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write
from marshawn_ordered_current_source_stage_v115 import closure_one,cpu_worker_initialize

def normalize(tree):
 class N(ast.NodeTransformer):
  def visit_Import(self,n):n.names=[x for x in n.names if x.name!='multiprocessing'];return n
  def visit_ImportFrom(self,n):
   if n.module=='marshawn_ordered_current_source_stage_v115':n.module='marshawn_ordered_current_source_stage_v108'
   return n
  def visit_Call(self,n):self.generic_visit(n);n.keywords=[x for x in n.keywords if x.arg!='mp_context'];return n
 return ast.dump(N().visit(tree),include_attributes=False)

def main():
 for old,new in [('marshawn_ordered_current_source_stage_v108.py','marshawn_ordered_current_source_stage_v115.py'),('marshawn_source_factory_v110.py','marshawn_source_factory_v115.py')]:assert normalize(ast.parse((O/old).read_text()))==normalize(ast.parse((O/new).read_text()))
 # Reproduce the relevant threaded parent initialization before testing spawn.
 cv2.calcOpticalFlowFarneback(np.zeros((64,64),np.uint8),np.ones((64,64),np.uint8),None,.5,3,15,3,5,1.2,0)
 Q=O/'CPU_spawn_v115';before=sha(V/'frames/001700/native_masks/DE.npz');sq=json.loads((O/'current_native_parts_v105/frame001700/qualification.json').read_text());archive=sq['qualified_archives'].get('DE',{}).get('path');started=time.time();direct=closure_one((1700,'DE',archive,Q/'sequential'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_worker_initialize)as pool:spawned=pool.submit(closure_one,(1700,'DE',archive,Q/'spawn')).result(timeout=90)
 with np.load(direct['qualified_archive'])as a,np.load(spawned['qualified_archive'])as b:
  assert set(a.files)==set(b.files)
  for key in a.files:assert np.array_equal(a[key],b[key]),key
 assert sha(direct['qualified_archive'])==sha(spawned['qualified_archive'])
 def norm(x):
  if isinstance(x,dict):return{k:norm(v)for k,v in x.items()if k!='created'}
  if isinstance(x,list):return[norm(v)for v in x]
  if isinstance(x,str):return x.replace(str(Q/'sequential'),'<OUTPUT>').replace(str(Q/'spawn'),'<OUTPUT>')
  return x
 assert norm(direct)==norm(spawned)
 assert sha(V/'frames/001700/native_masks/DE.npz')==before
 write(Q/'PASS.json',dict(status='pass',threaded_parent_then_actual_DE_spawn_canary=True,closure_archive_byte_equal=True,all_decisions_equal=True,closure_and_guard_AST_only_start_method_changed=True,current_native_unchanged=True,seconds=time.time()-started));print('PASS threaded-parent spawn; actualDE archive/decisions exact; source unchanged',flush=True)
if __name__=='__main__':main()
