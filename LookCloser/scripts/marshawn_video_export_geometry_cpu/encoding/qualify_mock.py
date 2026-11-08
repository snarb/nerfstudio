"""CPU control-plane tests. Mock120-frame metadata never represents a real movie."""
from pathlib import Path
import ast,json,hashlib,shutil,subprocess,sys,types,fcntl,time
O=Path(__file__).parent;sys.path.insert(0,str(O));import encoder as E
C=O.parents[1];A=C/'operations/encoder_color_audit_v1';Q=O/'qualification_mock_v2';Q.mkdir(exist_ok=True)
policy=dict(ffmpeg='/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffmpeg',ffprobe='/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffprobe',profile_sha256='2bd9bca54f43bb5da2920a5f95124677ee2754762bb8ad5551163d9dbf06eb81',kind='CPU_MOCK_ONLY_NOT_PRODUCTION')
request_path=Q/'path.json';actual_path=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/video/path_4K_v4/path.json');request_path.write_bytes(actual_path.read_bytes());request=E.read(request_path);fixture=Q/'MOCK120';fixture.mkdir(exist_ok=True);specpath=Q/'spec.json';E.atomic(specpath,dict(output=str(fixture)));spec=dict(output=str(fixture),_spec_path=str(specpath));hashes={i:E.sha(A/'PNG_masters'/f'{i:04d}.png')for i in range(3)}
for i in range(120):
 d=fixture/'frames'/f'{i:04d}';d.mkdir(parents=True,exist_ok=True)
 for variant in ['lookcloser','hybrid']:
  p=d/(variant+'.png')
  if not p.exists():p.symlink_to(A/'PNG_masters'/f'{i%3:04d}.png')
 E.atomic(d/'complete.json',dict(MOCK_QUALIFICATION_ONLY=True,binding=dict(path_sha256=E.sha(request_path),spec_sha256=E.sha(specpath),record=request['records'][i]),images={v+'.png':hashes[i%3]for v in ['lookcloser','hybrid']}))
class Mock(E.Encoder):
 def inspect(self,path,count,frame_root,variant):
  assert Path(path).read_bytes()==(A/'explicit709.mp4').read_bytes()
  return dict(MOCK_ONLY=True,frame_count=count,seconds=count/24,source_variant=variant)
original_run=E.subprocess.run;calls=[]
def run(command,**kwargs):
 calls.append(command);shutil.copy2(A/'explicit709.mp4',Path(command[-1]));return types.SimpleNamespace(returncode=0)
# Actual encoded-control proof, not a mock: exactcommand, tags,duration and decode.
real_command=E.command(policy['ffmpeg'],A/'PNG_masters/%04d.png',3,A/'explicit709.mp4')
assert real_command==E.read(A/'receipt.json')['commands']['explicit709'];E.probe(policy['ffprobe'],A/'explicit709.mp4',3)
rejected=False
try:E.probe(policy['ffprobe'],A/'baseline.mp4',3)
except ValueError:rejected=True
assert rejected
# Execute the actual qualified route's encode call AST against the installed adapter.
route=C/'operations/video_geometry_cpu_params_v2/route.py';tree=ast.parse(route.read_text());calls_ast=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='renderer'and n.func.attr=='encode'];assert len(calls_ast)==1
call=calls_ast[0];unit=ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)],type_ignores=[]));renderer=types.SimpleNamespace(encode=Mock(policy,lambda:None));E.subprocess.run=run
try:
 exec(compile(unit,str(route),'exec'),dict(renderer=renderer,local_spec=spec,canonical=request_path));assert len(calls)==2
 complete=E.read(fixture/'complete.json');assert complete['frame_count']==120 and complete['seconds']==5 and complete['visual_review_required']is True
 # Idempotent exactbinding reuse does not re-encode. Realmetadataalreadytestedabove.
 old_probe=E.probe;E.probe=lambda *a,**kw:dict(MOCK_ONLY=True)
 renderer.encode(spec,request_path);assert len(calls)==2;E.probe=old_probe
 # Private corrupted source image is rejected without touching original symlinktarget.
 p=fixture/'frames/0000/lookcloser.png';target=p.resolve();p.unlink();p.write_bytes(b'private_corrupt_test')
 try:renderer.encode(spec,request_path);raise AssertionError('changedPNG accepted')
 except ValueError:pass
 p.unlink();p.symlink_to(target)
 # Foreign finalMP4 cannot be overwritten, even with otherwisecorrectsourcebinding.
 foreign=Q/'foreign';foreign.mkdir(exist_ok=True);movie=foreign/'Marshawn1660_3s_24fps_hybrid_native_GT2s.mp4';movie.write_bytes(b'foreign_movie');before=E.sha(movie)
 try:renderer.encode.one(foreign,'hybrid',120,dict(test='source'),0);raise AssertionError('foreignMP4accepted')
 except ValueError:pass
 assert E.sha(movie)==before
 # Four coordinator crash points; owncompleteartifactadoption avoids duplicateencoding.
 crashes=[]
 for point in ['after_PREintent','after_FFmpeg','before_publish','after_publish']:
  d=Q/point;d.mkdir(exist_ok=True);fired=[]
  def event(name,value):
   if name==point and not fired:fired.append(name);raise RuntimeError('injected '+point)
  binding=dict(test='same-source',PNG_masters_sha256={},source_completion_sha256={});first=Mock(policy,lambda:None,event=event);start=len(calls)
  try:first.one(d,'hybrid',120,binding,0);raise AssertionError('missingcrash')
  except RuntimeError:pass
  second=Mock(policy,lambda:None);done=second.one(d,'hybrid',120,binding,0)
  assert done['binding']==dict(binding,variant='hybrid')and len(calls)-start==1
  try:second.one(d,'hybrid',120,dict(binding,test='foreign-source'),0);raise AssertionError('mixedbindingaccepted')
  except ValueError:pass
  crashes.append(dict(point=point,actual_mock_encoder_calls=1,exact_owned_adoption=True))
finally:E.subprocess.run=original_run
# Real inheritedFD holds lock after coordinator closes itsown descriptor; no fakeGPU.
lease_root=Q/'real_inherited_lease';lease_root.mkdir(exist_ok=True);lock=(lease_root/'encoding.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'],pass_fds=(lock.fileno(),));lock.close()
try:
 Mock(policy,lambda:None)(dict(output=str(lease_root)),request_path);raise AssertionError('inheritedleaseignored')
except E.AwaitEncodingOwnership:pass
child.wait(timeout=5);lock=(lease_root/'encoding.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
assert E.sha(actual_path)==E.sha(request_path)
result=dict(kind='CPU_ENCODER_CONTROL_PLANE_AND_FROZEN_CALL_QUALIFICATION_V1',MOCK120_is_not_real_video=True,actual_frozen_route_sha256=E.sha(route),actual_route_call_AST=ast.dump(call),adapter_sha256=E.sha(O/'encoder.py'),actual_threePNG_control_command_exact=True,actual_tagged_control_probe_pass=True,actual_untagged_baseline_probe_rejected=True,mock_actual_route_call120frames5sec_pass=True,exactbinding_idempotent_no_reencode=True,changedPNG_rejected=True,foreign_existingMP4_preserved=True,crosssource_rejected=True,crash_cases=crashes,real_inherited_lock_pass=True,busylock_retry_not_consumed=True,production_paths_mutated=False,additional_real_encodes=0)
E.atomic(Q/'complete.json',result);print(json.dumps(result,indent=2))
