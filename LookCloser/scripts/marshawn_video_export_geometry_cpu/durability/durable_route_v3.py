"""OPS-only crash adoption around the unchanged qualified params-only router."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
O=Path(__file__).parent;C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');P=C/'operations/video_geometry_cpu_params_v2'
sys.path.insert(0,str(O))
from bridge_v3 import Bridge,AwaitCPUOwnership
import cpu_exec
from migrate_schema import migrate

def sha(p):return cpu_exec.sha(p)
def read(p):return cpu_exec.read(p)
def write(p,v):return cpu_exec.atomic(p,v)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(a):
 release=read(O/'root_release.json');release_digest=sha(O/'root_release.json')
 if not 1<=a.cpu_workers<=3:raise ValueError('Qualified CPU capacity is1..3')
 def guard():
  if release.get('approved')is not True or sha(O/'root_release.json')!=release_digest:raise ValueError('OPS durability release missing')
  for path,value in release['files_sha256'].items():
   if sha(path)!=value:raise ValueError('Durable export source/release changed '+path)
  base=read(a.export_plan)
  if sha(base['spec'])!=release['base_render_spec_sha256']or read(base['spec'])['source_count']!=16:raise ValueError('Frozen source-count/spec changed')
 guard();migration=migrate(P,O/'preserved_pre_GT_schema_qualification',sha(P/'one.py'));write(O/'schema_migration.json',migration);router=module(P/'route.py','qualified_params_router_unchanged');plan=read(a.fleet/'video_plan.json');hosts=plan['hosts']
 # Stage only the small read-only verifier, never models, YAMLs or RAWs.
 for name,host in hosts.items():
  root=Path(host['root']);dest=root/'operations/video_geometry_cpu_durable_v3';router.call(host,['mkdir','-p',str(dest)])
  if root!=C:subprocess.run(['rsync','-a',str(O/'verify_geometry.py'),host['address']+':'+str(dest)+'/'],check=True)
 bridge=Bridge(router,O/'CPU_exec',O/'cpu_exec.py',shader_sha=release['CPU_shader_sha256'],source_guard=guard,max_concurrent=a.cpu_workers);bridge.install()
 sys.path[:0]=[str(C/'code'),str(C/'code/LookCloser/scripts')]
 import marshawn_tuning_controller as fleet
 original_tick=fleet.tick
 def tick(plan,state,root):
  guard();result=original_tick(plan,state,root)
  for name,item in state['jobs'].items():
   if item.get('status')=='orphaned'and item.get('worker_alive'):
    item.update(status='unknown',adoption_wait='Exact old GPU worker still alive; no replacement/retry')
   elif item.get('status')in ['orphaned','failed']and item.get('worker_alive')is False and item.get('supervisor_alive')is False:
    # Honest completion is the renderer's own source-bound artifact, not an
    # invented training completion or an assumption based on process absence.
    index=int(name.rsplit('_',1)[1]);request=read(P/'canonical_path.json');host=hosts[item['host']];campaign=Path(host['root']);expected=dict(campaign=str(campaign),record=request['records'][index],path_sha256=sha(P/'canonical_path.json'),GPU_worker_sha256=release['GPU_shader_sha256'],code_seal_sha256=sha(C/'code_seal.json'));directory=campaign/'video/geometry_cpu_float_v1/frames'/f'{index:04d}';verifier=campaign/'operations/video_geometry_cpu_durable_v3/verify_geometry.py'
    try:
     command=[host['python'],str(verifier),'--directory',str(directory),'--expected-json',json.dumps(expected,separators=(',',':'))];proof=json.loads(router.call(host,command,timeout=60).stdout)
    except(subprocess.SubprocessError,OSError,json.JSONDecodeError)as exc:
     item.update(status='unknown',adoption_wait='CPU artifact proof unavailable; no replacement',adoption_transport_error=str(exc));continue
    if proof.get('verified')is True:
     if proof.get('verifier_sha256')!=sha(O/'verify_geometry.py'):raise ValueError('Remote artifact verifier changed')
     item.update(status='complete',completion_source='verified_own_render_artifact_after_supervisor_loss',render_artifact_proof=proof)
     with(O/'GPU_artifact_adoptions.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),job=name,proof=proof))+'\n')
  # This is the same file that frozen tick writes BEFORE the SSH side effect.
  fleet.atomic(Path(root)/'state.json',state)
  return result
 fleet.tick=tick
 try:
  while True:
   try:router.main(a);return
   except AwaitCPUOwnership as exc:
    write(O/'CPU_ownership_wait.json',dict(reason=str(exc),time=time.time(),retry_not_consumed=True));time.sleep(5)
 finally:fleet.tick=original_tick;bridge.restore()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fleet',type=Path,required=True);p.add_argument('--export-plan',type=Path,required=True);p.add_argument('--cpu-workers',type=int,default=3);a=p.parse_args();main(a)
