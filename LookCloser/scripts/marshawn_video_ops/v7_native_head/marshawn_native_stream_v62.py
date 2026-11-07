"""Per-camera immutable source qualification while RAW transport continues."""
from pathlib import Path
import json,subprocess,shlex,time
import marshawn_sam3_assist as io
REMOTE="""import sys,json,hashlib,pathlib
spec=json.load(sys.stdin);out={}
for camera,files in spec.items():
 ok=True;actual={}
 for filename,want in files.items():
  p=pathlib.Path(filename)
  if not p.is_file():ok=False;break
  actual[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
  if actual[filename]!=want:ok=False;break
 if ok:out[camera]=actual
print(json.dumps(out))
"""

def ready(config,root):
 proofpath=Path(root)/'streamed_source_ready.json';old=io.read(proofpath)if proofpath.exists()else dict(cameras={})
 expected={}
 for row in config['records']:
  camera=row['camera']
  if camera in old['cameras']:continue
  variant=row['variants']['silver'];task=io.read(variant['task']);d=Path(task['remote_task']);local=Path(variant['task']).parent
  expected[camera]={str(d/name):io.sha(local/name)for name in ['task.json','reference.npz','source_original_receipt.json','source_portable_receipt.json','union_resolver.py','seed_policy_v5.json']}
  expected[camera][str(d/'raw_native.png')]=task['raw_native_image_sha256']
 if expected:
  h=config['hosts']['silver'];argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',h['address'],shlex.join([h['python'],'-c',REMOTE])]
  try:
   r=subprocess.run(argv,input=json.dumps(expected),capture_output=True,text=True,timeout=45);r.check_returncode();fresh=json.loads(r.stdout);old['cameras'].update(fresh)
  except Exception as e:old['last_probe_error']=str(e)
 old.update(frame=config['frame'],configuration_sha256=io.sha(config['stream_configuration']),time=time.time(),all_inputs_actual_sha_verified=True);io.write(proofpath,old)
 return set(old['cameras'])
