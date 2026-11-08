from pathlib import Path
import hashlib,json,os,subprocess,time,importlib.util
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');O=Path(__file__).parent;F=Path('/fsx/tmp/marshawn_fleet10_140k_20261107');p=F/'postprocess_command.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def immutable(p,data):
 blob=json.dumps(data,indent=2)+'\n'
 if p.exists():assert p.read_text()==blob;return
 p.write_text(blob)
assert sha(O/'root_release.json')=='f8f36759299f7c311ecdd403f8267bddafbf78d0f6f362c6ba579222124dd586';release=read(O/'root_release.json');assert release['approved']
for path,value in release['files_sha256'].items():assert sha(path)==value
assert sha(p)==release['prior_active_hook_sha256'];assert not(F/'postprocess_complete.json').exists()and not(C/'operations/video_geometry_cpu_params_v2/state.json').exists()
cmd=Path('/proc/2270475/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'marshawn_fleet10_140k.py'in cmd and str(F)in cmd
# Actual already-running campaign launcher's environment, never printed or changed.
env=dict(x.decode().split('=',1)for x in Path('/proc/2270475/environ').read_bytes().split(b'\0')if x)
code="from pathlib import Path;import importlib.util,json,hashlib;p=Path(%r);s=importlib.util.spec_from_file_location('entry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);r=m.read(p.parent/'root_release.json');assert r['approved'];exec(\"for q,v in r['files_sha256'].items():\\n assert m.sha(q)==v,q\");print(json.dumps(dict(passed=True,entry_sha256=m.sha(p),checked_files=len(r['files_sha256']),no_model_or_GPU_or_export_started=True)))"%str(O/'route.py')
PY='/home/pavlo/.conda/envs/nerfstudio/bin/python';before=json.loads(subprocess.check_output([PY,'-c',code],env=env,cwd=F,text=True));immutable(O/'actual_F_environment_before_activation.json',before)
old=p.read_bytes();backup=O/'previous_active_hook_06f2541b.json'
if backup.exists():assert backup.read_bytes()==old
else:backup.write_bytes(old)
hook=read(O/'postprocess_command.candidate.json');hook.update(activation_status='ACTIVE_ROOT_RELEASED',ROOT_DEV3_UUID_release_path=str(O/'root_release.json'),ROOT_DEV3_UUID_release_sha256=sha(O/'root_release.json'),activated_at=time.time());hook['source_hashes'][str(O/'root_release.json')]=sha(O/'root_release.json');temporary=p.with_name('postprocess_command.UUID_v5.pending')
with temporary.open('w')as stream:json.dump(hook,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
temporary.replace(p)
for path,value in hook['source_hashes'].items():assert sha(path)==value,path
actual=json.loads(subprocess.check_output([PY,'-c',code],env=env,cwd=F,text=True));immutable(O/'actual_F_environment_after_activation.json',actual)
result=dict(activated=True,active_hook_sha256=sha(p),previous_hook_sha256=sha(backup),ROOT_release_sha256=sha(O/'root_release.json'),entry_actual_F_environment_before=before,entry_actual_F_environment_after=actual,all_active_source_hashes_verified=len(hook['source_hashes']),final72_render_not_started=True,profile_sha256=hook['profile_sha256'],frozen_GPU_shader_unchanged='6603aa06dc8ef2df2cd89cada087729eaa40bd81be5b5c981e2e3e2369539eba',frozen_CPU_shader_unchanged='51239b1e3c2d969388bec12b7af907a2822b35454f08d0a7476ecb626bedc4bf',time=time.time());immutable(O/'activation.json',result);print(json.dumps(result))
