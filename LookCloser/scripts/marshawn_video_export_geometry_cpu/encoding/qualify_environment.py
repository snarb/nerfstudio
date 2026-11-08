from pathlib import Path
import os,sys,json,subprocess,hashlib,re,time
O=Path(__file__).parent;sys.path.insert(0,str(O));from encoder import sha,command,probe,decode_check
C=O.parents[1];F=Path('/fsx/tmp/marshawn_fleet10_140k_20261107');A=C/'operations/encoder_color_audit_v1';PID=2270475
proc=Path('/proc')/str(PID);argv=(proc/'cmdline').read_bytes().decode().split('\0');assert '_dispatch'in argv and str(F/'code/marshawn_fleet10_140k.py')in argv
raw=(proc/'environ').read_bytes();env={k.decode():v.decode()for x in raw.split(b'\0')if x for k,v in [x.split(b'=',1)]};identity=dict(PID=PID,start=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=argv)
tools={};libraries={};provenance={};B=Path('/home/pavlo/.conda/envs/lookcloser-jpeg/bin')
for name in ['ffmpeg','ffprobe']:
 p=B/name;version=subprocess.check_output([str(p),'-version'],env=env,text=True);linked=subprocess.check_output(['/usr/bin/ldd',str(p)],env=env,text=True);paths=[]
 for line in linked.splitlines():
  match=re.search(r'=>\s+(/\S+)',line)or re.match(r'\s*(/\S+)\s+\(',line)
  if match:paths.append(str(Path(match.group(1)).resolve()))
 for path in paths:libraries[path]=sha(path)
 tools[name]=dict(path=str(p),sha256=sha(p),version_sha256=hashlib.sha256(version.encode()).hexdigest());provenance[name]=dict(version=version,ldd=linked,resolved_paths=paths)
control=O/'control_actual_launcher.mp4';cmd=command(B/'ffmpeg',A/'PNG_masters/%04d.png',3,control)
with(O/'control_actual_launcher.log').open('w')as log:subprocess.run(cmd,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
meta=probe(B/'ffprobe',control,3)
# Keep actualPNGselection layout for the normal decodercheck, withoutcopiedmasters.
fixture=O/'fixture_frames';fixture.mkdir(exist_ok=True)
for i in range(3):
 d=fixture/f'{i:04d}';d.mkdir(exist_ok=True);p=d/'hybrid.png'
 if not p.exists():p.symlink_to(A/'PNG_masters'/f'{i:04d}.png')
decoded=decode_check(B/'ffmpeg',control,3,fixture,'hybrid');same=sha(control)==sha(A/'explicit709.mp4')
assert (proc/'cmdline').read_bytes().decode().split('\0')==argv
old=subprocess.run(['/home/pavlo/.conda/envs/nerfstudio/bin/ffmpeg','-version'],env=env,capture_output=True,text=True)
r=dict(kind='ACTUAL_FLEET_LAUNCHER_ENVIRONMENT_ENCODER_QUALIFICATION_V1',time=time.time(),identity=identity,environment_safe_fields={k:env.get(k)for k in ['PATH','LD_LIBRARY_PATH','CONDA_PREFIX','CONDA_DEFAULT_ENV']},tools=tools,resolved_library_sha256=libraries,provenance=provenance,existing_broken_default_tool_returncode=old.returncode,existing_broken_default_tool_stderr=old.stderr,command=cmd,ffprobe=meta,decode=decoded,repeated_same_single_control_byte_identical=same,control_sha256=sha(control),production_posthook_sha256=sha(F/'postprocess_command.json'),no_global_environment_or_production_change=True)
(O/'environment_qualification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'tools':tools,'resolved_libraries':len(libraries),'same_single_control_byte_identical':same,'olddefaultreturncode':old.returncode,'posthooksha':r['production_posthook_sha256']},indent=2))
