from pathlib import Path
import hashlib,json,sys,subprocess,time,importlib.util
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');Q=V/'operations/frequency13/qualification'
sys.path[:0]=[str(V/'operations/frequency13/code'),str(V/'code/LookCloser/scripts')]
import marshawn_clever_frequency_backend as b
import marshawn_tuning_controller as f
c=b.read(Q/'config.json');R=Path(c['hosts']['silver']['campaign']);B=Path(c['clever_backend']['private_root']);remote=B/'operations/frequency13_qualification/maps/001690_HC';tx=Path(c['clever_backend']['transactions'])/'001690_HC'
# Isolated destination only. The real canonical1690HC map is never modified.
sg=c['hosts']['silver'];spec=json.loads(b.host_call(sg,['cat',str(R/'operations/future_rgb_sg/001690/specs/HC.json')]).stdout);spec['output']=str(Path(c['clever_backend']['archive_root'])/'qualification_cache/frames/001690/data/lookcloser_frequencies/train_HC.pt');p=Q/'HC_SG_isolated_spec.json';b.write(p,spec);b.host_call(sg,['mkdir','-p',str(Path(c['clever_backend']['qualification_SG_spec']).parent)]);b.rsync([p,sg['address']+':'+c['clever_backend']['qualification_SG_spec']])
original={}
for host,paths in [('LOCAL',[V/'frames/001690/data/lookcloser_frequencies/train_HC.pt',V/'frames/001690/data/lookcloser_frequencies/train_HC.json']),('SG',[R/'frames/001690/data/lookcloser_frequencies/train_HC.pt',R/'frames/001690/data/lookcloser_frequencies/train_HC.json'])]:
 for x in paths:
  original[str(x)]=b.sha(x)if host=='LOCAL'else b.host_call(sg,['sha256sum',str(x)]).stdout.split()[0]
b.write(Q/'original_main_maps.json',original)
# Reuse the already-qualified SAME-image fit, with byte-bound provenance. No extra numerical fit.
code="from pathlib import Path;import shutil,hashlib; r=Path(%r);o=r/'output';o.mkdir(exist_ok=True);\nfor s in ['.pt','.json']:\n a=Path(%r)/('canary_HC'+s);t=o/('train_HC'+s);\n if t.exists():assert t.read_bytes()==a.read_bytes()\n else:shutil.copyfile(a,t)\n"%(str(remote),str(B/'output'))
b.host_call(c['hosts']['clever'],[c['hosts']['clever']['python'],'-c',code])
item=dict(frame=1690,camera='HC',name='clever_backend_canary001690_HC_a00',spec=str(remote/'source/spec.json'),host='clever');job=b.map_job(c,item);plan=dict(schema_version=1,poll_seconds=1,hosts={'clever':c['hosts']['clever']},jobs=[job]);root=Q/'isolated_dispatch';root.mkdir(exist_ok=True)
f.atomic(root/'manifest.json',plan);plan,state=f.prepare(root,root/'manifest.json');real=f.host_call;injected=[]
def lost_ack(host,args,**kwargs):
 out=real(host,args,**kwargs)
 if args[0]=='_start'and not injected:
  injected.append('SSH_ack_lost_AFTER_actual_detached_start');raise subprocess.TimeoutExpired(args,1)
 return out
f.host_call=lost_ack
if state['jobs'][item['name']]['status']=='pending':f.tick(plan,state,root)
f.host_call=real
# Simulate controller restart: discard local in-memory state and adopt durable receipt.
state=f.read(root/'state.json');dispatch_spec=state['jobs'][item['name']].get('spec_sha256');deadline=time.time()+100
while time.time()<deadline:
 done=f.tick(plan,state,root)
 if done:break
 time.sleep(2)
actual=state['jobs'][item['name']];assert actual['status']=='complete',actual
assert actual['spec_sha256']==dispatch_spec
log=json.loads(b.host_call(c['hosts']['clever'],['cat',str(Path(c['hosts']['clever']['root'])/'jobs'/item['name']/'receipt.json')]).stdout)
assert log['returncode']==0 and log['gpu_uuid']==c['clever_backend']['GPU_uuid']
workerlog=b.host_call(c['hosts']['clever'],['cat',str(Path(c['hosts']['clever']['root'])/'jobs'/item['name']/'worker.log')]).stdout
assert '"reused_same_image": true'in workerlog,workerlog
# Deliberate failures AFTER archive and AFTER SGpublication; retry must adopt exact bytes.
real_rpc=b.rpc;crashes=[]
for phase in ['archive','publish']:
 def crash(c0,h,a,q,phase=phase):
  result=real_rpc(c0,h,a,q)
  if h=='silver'and a==phase:
   crashes.append(phase+'_ack_lost_AFTER_success');raise subprocess.TimeoutExpired(a,1)
  return result
 b.rpc=crash
 try:b.finish(c,item,actual)
 except subprocess.TimeoutExpired:pass
 else:raise AssertionError('Expected crash injection '+phase)
 b.rpc=real_rpc
published=b.finish(c,item,actual);again=b.finish(c,item,actual);assert again==published
for x,digest in original.items():
 current=b.sha(x)if x.startswith(str(V))else b.host_call(sg,['sha256sum',x]).stdout.split()[0]
 assert current==digest,('MAIN canonical mutated',x)
result=dict(kind='ISOLATED_CLEVER_V13_BACKEND_END_TO_END_QUALIFICATION',passed=True,frame=1690,camera='HC',no_new_numerical_fit=True,reused_same_image=True,no_main_map_claim_or_mutation=True,SSH_ack_loss_adoption=injected,publication_ack_loss_adoption=crashes,supervisor=actual['supervisor'],worker=actual['worker'],gpu_uuid=actual['gpu_uuid'],job_spec_sha256=actual['spec_sha256'],published=published,source_config_sha256=b.sha(Q/'config.json'),backend_sha256=b.sha(V/'operations/frequency13/code/marshawn_clever_frequency_backend.py'))
b.write(Q/'backend_canary_result.json',result);print(json.dumps(result))
