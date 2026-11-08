"""OPS-only DEV3 UUID entry adapter around the released final render route."""
from pathlib import Path
import argparse,copy,importlib.util,json,sys
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');O=Path(__file__).parent;E=C/'operations/video_geometry_cpu_encoding_v4';sys.path.insert(0,str(C/'code/LookCloser/scripts'))
import marshawn_tuning_controller as fleet

def sha(p):return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def adapt(plan):
 result=copy.deepcopy(plan)
 for job in result['jobs']:
  if job.get('host')!='dev3':continue
  command=job['command']
  if len(command)<3 or Path(command[1]).name!='worker.py'or command[2]!='gpu':raise ValueError('DEV3 adapter accepts frozen geometry shader only')
  expected='6603aa06dc8ef2df2cd89cada087729eaa40bd81be5b5c981e2e3e2369539eba'
  if sha(command[1])!=expected:raise ValueError('DEV3 GPU shader changed')
  job['command']=[command[0],str(O/'dev3_entry.py'),'--worker',command[1],'--expected-sha256',expected,'--',*command[2:]]
 return result

def main(a):
 release=read(O/'root_release.json');frozen_release_sha=sha(O/'root_release.json')
 def guard():
  if release.get('approved')is not True or sha(O/'root_release.json')!=frozen_release_sha:raise ValueError('ROOT DEV3 operational release missing/changed')
  for p,v in release['files_sha256'].items():
   if sha(p)!=v:raise ValueError('DEV3 operational source/qualification changed '+p)
 guard();original=fleet.tick
 def tick(plan,state,root):
  guard();return original(adapt(plan),state,root)
 fleet.tick=tick
 spec=importlib.util.spec_from_file_location('qualified_explicit_encoder_route_unchanged',E/'route.py');router=importlib.util.module_from_spec(spec);spec.loader.exec_module(router)
 try:return router.main(a)
 finally:fleet.tick=original
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fleet',type=Path,required=True);p.add_argument('--export-plan',type=Path,required=True);p.add_argument('--cpu-workers',type=int,default=3);main(p.parse_args())
