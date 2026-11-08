from pathlib import Path
import json,sys,tempfile,types,shutil,ast,subprocess,hashlib
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');O=V/'operations';sys.path[:0]=[str(O),str(V/'code/LookCloser/scripts')]
import marshawn_native_cache_consumer_v114 as c
from prepare_marshawn_tuning import sha
base=ast.parse((O/'sam3_assist/code/marshawn_mask_assist_full_v113.py').read_text());new=ast.parse((O/'sam3_assist/code/marshawn_mask_assist_full_v114.py').read_text());old=next(n for n in base.body if isinstance(n,ast.FunctionDef)and n.name=='stage');fresh=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='stage');wrapped=next(n for n in fresh.body if isinstance(n,ast.Try));assert [ast.dump(n)for n in old.body[6:]]==[ast.dump(n)for n in wrapped.body]
original_run=c.subprocess.run;original_module=sys.modules.get('marshawn_native_png_prefetch');events=[]
try:
 with tempfile.TemporaryDirectory()as td:
  t=Path(td);K=t/'operations/native_png_prefetch_v1';K.mkdir(parents=True);shutil.copyfile(O/'native_png_prefetch_v1/release.json',K/'release.json');S=t/'sources/1705';S.mkdir(parents=True);(S/'source_ready.json').write_text(json.dumps(dict(cameras=['AA'])));root=t/'private_stage';root.mkdir();f=types.SimpleNamespace(identity=lambda:dict(pid=1,start='test',boot='test'))
  sys.modules['marshawn_native_png_prefetch']=types.SimpleNamespace(source_binding=lambda *a,**kw:dict(frame=1705,camera='AA'))
  def absent(argv,**kwargs):
   events.append(argv)
   if argv[0]=='ssh':return types.SimpleNamespace(stdout=json.dumps(dict(kind='CREATE_ONLY_NATIVE_PNG_CACHE_CONSUMER_V1',records=[dict(camera='AA',linked=False,reason='cache_not_ready')])))
   return types.SimpleNamespace(stdout='')
  c.subprocess.run=absent;release=dict(remote_root='/cache/test',remote_host='unused',remote_python='unused');token=c.begin(t,1705,release,root,f);assert token is not None;c.finish(token,'complete');assert json.loads(Path(token['marker']).read_text())['status']=='complete'
  def fail(*a,**kw):raise subprocess.TimeoutExpired('optional transport',1)
  c.subprocess.run=fail;token=c.begin(t,1705,release,root,f);assert json.loads((root/'optional_native_cache_result.json').read_text())['fall_through_original_transport'];c.finish(token,'failed');assert json.loads(Path(token['marker']).read_text())['status']=='failed'
  v=json.loads(Path(token['marker']).read_text());v['nonce']='newer-owner';Path(token['marker']).write_text(json.dumps(v));c.finish(token,'complete');assert json.loads(Path(token['marker']).read_text())['nonce']=='newer-owner'
finally:
 c.subprocess.run=original_run
 if original_module is None:sys.modules.pop('marshawn_native_png_prefetch',None)
 else:sys.modules['marshawn_native_png_prefetch']=original_module
proof=dict(status='pass',original_rsync_and_stream_statements_AST_exact=True,missing_cache_falls_through=True,optional_timeout_falls_through=True,marker_owner_scope_preserved=True,original_remote_and_source_inputs_untouched=True,adapter_sha256=sha(O/'marshawn_native_cache_consumer_v114.py'),factory_sha256=sha(O/'sam3_assist/code/marshawn_mask_assist_full_v114.py'))
(O/'native_png_prefetch_v1/consumer_v114_CPU_PASS.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
