"""Publish source-qualified INPUTS into NEW C/SGC; never copy a model."""
from pathlib import Path
import json,subprocess,shlex,sys

def publish(c,frame,gate,stage,read,write,sha):
 c=dict(c);plan=read(Path(c['fleet10_root'])/'video_plan.json');route=plan['frame_generations'][str(frame)]
 if Path(c['root'])!=Path(route['source_campaign']):raise ValueError('Generation source root differs')
 c['production_root']=route['model_campaign_by_host']['olive'];c['route_hosts']={h:dict(v,host_root=route['model_campaign_by_host'][h])for h,v in c['route_hosts'].items()}
 source=Path(c['root']);sf=source/'frames'/f'{frame:06d}';data=sf/'data';ops=Path(c['operations'])/f'{frame:06d}';C=Path(c['production_root']);fleet=Path(c['fleet10_root'])
 result=subprocess.run([sys.executable,str(fleet/'code/marshawn_fleet10_generations.py'),'reserve','--root',str(fleet),'--frame',str(frame)],capture_output=True,text=True,timeout=180)
 if result.returncode:raise RuntimeError('140k host reservation failed: '+result.stderr[-800:])
 response=json.loads(result.stdout)
 if response['status']in {'awaiting_idle_training_slot','awaiting_assignment_lock','held_source_generation_not_released','awaiting_current_generation_source_grid_dense'}:return None
 if response['status']!='reserved':raise ValueError('Unknown cap fleet reservation')
 assignment_path=Path(response['assignment_path']);expected=Path(plan['assignments'])/f'{frame:06d}.json';assignment=read(assignment_path)
 if assignment_path.resolve()!=expected.resolve()or assignment['frame']!=frame or assignment['kind']!='MARSAWN_FLEET10_FRAME_ASSIGNMENT_V1'or assignment.get('hard_max_iterations')!=140000:raise ValueError('Wrong capped assignment')
 if any(assignment.get(k)!=route[k]for k in ['model_generation','source_generation']):raise ValueError('Assignment generation differs')
 host=assignment['host'];allowed=c['route_hosts'][host]
 for key in ['host_root','python','address']:
  if assignment.get(key)!=allowed.get(key):raise ValueError('Training host differs from fixed C/SGC contract')
 def bindings():return dict(input_manifest_sha256=stage.digest(stage.inventory(data,frame)),data_ready_sha256=sha(data/'ready.json'),native_semantic_ready_sha256=sha(sf/'native_semantic_ready.json'),safe_gate_before_dispatch_sha256=sha(ops/'safe_gate_before_dispatch.json'))
 bound=bindings()
 if any(assignment.get(k)!=v for k,v in bound.items())or assignment['fleet_plan_sha256']!=sha(fleet/'video_plan.json'):raise ValueError('Assignment belongs to other source inputs/plan')
 target=C/'frames'/f'{frame:06d}';marker=target/'dispatch_ready.json'
 if marker.exists():
  r=read(marker)
  expected_marker=dict(frame=frame,host=host,host_root=assignment['host_root'],remote_root=assignment['host_root'],model_generation=route['model_generation'],source_generation=route['source_generation'],source_root=str(source),input_manifest_sha256=bound['input_manifest_sha256'],assignment_sha256=sha(assignment_path),assignment_nonce=assignment['assignment_nonce'],source_ready_sha256=bound['data_ready_sha256'],source_admission_manifest_sha256=gate['manifest_sha256'],hard_max_iterations=140000)
  if any(r.get(k)!=v for k,v in expected_marker.items()):raise ValueError('Never reuse published marker from another generation/root/assignment/source')
  return r
 for n in ['spec.json','train/runtime.json','train/cold_provenance.json']:
  if(target/n).exists():raise ValueError('Never overwrite a bound cap model')
 dest=target/'data';dest.mkdir(parents=True,exist_ok=True)
 # Regular owned same-frame INPUT copies only. No CP/TRAIN directory is in
 # this source subtree; no absolute operational frequency specs are copied.
 exclude=['--exclude=admission/','--exclude=frequency_specs/','--exclude=*.fit.lock','--exclude=*.partial*']
 subprocess.run(['rsync','-rLt',*exclude,str(data)+'/',str(dest)+'/'],check=True,timeout=1200)
 admission=c['target']+':'+str(Path(c['target_root'])/'frames'/f'{frame:06d}'/'data/admission')+'/'
 (dest/'admission').mkdir(exist_ok=True);subprocess.run(['rsync','-aL',admission,str(dest/'admission')+'/'],check=True,timeout=1200)
 from marshawn_video_support import load_support
 from prepare_marshawn_tuning import verify_inventory
 from marshawn_video_frequency_frame import specs
 import marshawn_video_frequency as frequency
 import numpy as np
 if sha(dest/'ready.json')!=bound['data_ready_sha256']or sha(dest/'admission/manifest.json')!=gate['manifest_sha256']:raise ValueError('Production source/grid bytes changed')
 _,support=load_support(dest/'admission/manifest.json',gate['manifest_sha256'],dest,np.asarray(read(dest/'ready.json')['aabb']),(128,)*3)
 if support!=gate['verified']:raise ValueError('Production frozen support proof differs')
 verify_inventory(dest)
 for p in specs(C,frame):
  s=read(p);frequency.validate_spec(s)
  if not frequency.validate_existing(s):raise ValueError('All66 source-bound maps must already exist; no target fitting allowed')
 stage.atomic(dest/'staged_input_manifest.json',stage.inventory(dest,frame));local=stage.verify(dest,frame)
 if host=='silver':
  remote=Path(assignment['host_root'])/'frames'/f'{frame:06d}';code='from pathlib import Path;import sys,json;f=Path(sys.argv[1]);assert not any((f/n).exists()for n in ["spec.json","dispatch_ready.json","train/runtime.json"]);f.mkdir(parents=True,exist_ok=True)'
  subprocess.run(['ssh',assignment['address'],shlex.join([assignment['python'],'-c',code,str(remote)])],check=True)
  # SG already owns the verified canonical source bytes. Copy them locally
  # on SG instead of sending the same gigabyte across the network twice.
  clone=['rsync','-rLt','--exclude=frequency_specs/','--exclude=*.fit.lock','--exclude=*.partial*',str(Path(c['target_root'])/'frames'/f'{frame:06d}'/'data')+'/',str(remote/'data')+'/']
  subprocess.run(['ssh',assignment['address'],shlex.join(clone)],check=True,timeout=1200)
  verification='''from pathlib import Path;import sys,json,numpy as np
r=Path(sys.argv[1]);f=int(sys.argv[2]);sys.path[:0]=[str(r/'code'),str(r/'code/LookCloser/scripts')]
from marshawn_video_frequency_frame import specs
import marshawn_video_frequency as frequency
from marshawn_video_stage import verify
from marshawn_video_support import load_support
d=r/'frames'/('%06d'%f)/'data';read=lambda p:json.loads(p.read_text())
for p in specs(r,f):
 s=read(p);frequency.validate_spec(s);assert frequency.validate_existing(s)
m=d/'admission/manifest.json';ready=read(d/'ready.json');_,proof=load_support(m,__import__('hashlib').sha256(m.read_bytes()).hexdigest(),d,np.asarray(ready['aabb']),(128,)*3)
print(json.dumps(dict(verification=verify(d,f),support=proof)))'''
  reply=subprocess.run(['ssh',assignment['address'],shlex.join([assignment['python'],'-c',verification,assignment['host_root'],str(frame)])],check=True,capture_output=True,text=True,timeout=300);checked=json.loads(reply.stdout)
  if checked['support']!=support:raise ValueError('SGC admission differs')
  verified=checked['verification']
 elif host=='dev3':
  argv=[assignment['python'],str(C/'code/LookCloser/scripts/marshawn_video_stage.py'),'verify','--data',str(dest),'--frame',str(frame)]
  reply=subprocess.run(['ssh',assignment['address'],shlex.join(argv)],check=True,capture_output=True,text=True,timeout=180);verified=json.loads(reply.stdout)
 else:verified=local
 if verified['manifest_sha256']!=sha(dest/'staged_input_manifest.json')or bindings()!=bound:raise ValueError('Source/target inputs changed before publication')
 if any(p.is_symlink()for p in dest.rglob('*')):raise ValueError('Production DATA contains foreign symlinks')
 receipt=dict(schema=1,frame=frame,host=host,remote_root=assignment['host_root'],host_root=assignment['host_root'],model_generation=route['model_generation'],source_generation=route['source_generation'],source_root=str(source),generation_route=route,input_manifest_sha256=bound['input_manifest_sha256'],remote_verification=verified,no_model_or_learned_state_staged=True,
  assignment_path=str(expected),assignment_sha256=sha(expected),assignment_nonce=assignment['assignment_nonce'],physical_GPU_index=assignment['gpu'],hard_max_iterations=140000,source_ready_sha256=bound['data_ready_sha256'],source_admission_manifest_sha256=gate['manifest_sha256'],frozen_source_support_verified=support,regular_owned_sameframe_DATA_materialized=True)
 write(ops/'host_route_publication.json',receipt);stage.atomic(marker,receipt);return receipt
