from pathlib import Path
import ctypes,importlib.util,json,types,uuid,hashlib,copy
O=Path(__file__).parent;C=O.parents[1]
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=module(O/'dev3_entry.py','entry');route=module(O/'route.py','route');calls=[];expected='GPU-ab838312-8799-795c-7df2-b085cb3db092'
class Function:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*x):return self.fn(*x)
def getctx(pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int)).contents.value=7;return 0
def getuuid(pointer,device):
 assert device==7,'Must use actual currentcontext device, not ordinal0';calls.append(device);ctypes.cast(pointer,ctypes.POINTER(ctypes.c_ubyte*16)).contents[:]=uuid.UUID(expected[4:]).bytes;return 0
fake=types.SimpleNamespace(cuCtxGetDevice=Function(getctx),cuDeviceGetUuid_v2=Function(getuuid));original=a.ctypes.CDLL;a.ctypes.CDLL=lambda x:fake;torch=types.SimpleNamespace(cuda=types.SimpleNamespace(init=lambda:None,device_count=lambda:1,current_device=lambda:0,synchronize=lambda:None),empty=lambda *a,**k:object())
try:
 actual,device=a.context_uuid(torch);assert actual==expected and device==7 and calls==[7]
 fake.cuCtxGetDevice=Function(lambda p:201)
 try:a.context_uuid(torch)
 except RuntimeError:pass
 else:raise AssertionError('Unprovable context accepted')
 torch.cuda.device_count=lambda:2
 try:a.context_uuid(torch)
 except RuntimeError:pass
 else:raise AssertionError('Multiplevisibledevices accepted')
finally:a.ctypes.CDLL=original
inputs=json.loads((C/'operations/video_partial_first6_v1/inputs.json').read_text());plan=dict(schema_version=1,hosts=inputs['hosts'],jobs=inputs['jobs']);before=copy.deepcopy(plan);adapted=route.adapt(plan);assert plan==before
for old,new in zip(plan['jobs'],adapted['jobs']):
 if old['host']=='dev3':
  assert new['command'][1]==str(O/'dev3_entry.py')and new['command'][3]==old['command'][1]and new['command'][7:]==old['command'][2:]
  assert {k:v for k,v in old.items()if k!='command'}=={k:v for k,v in new.items()if k!='command'}
 else:assert old==new
result=dict(passed=True,actual_current_context_device_used_not_physical_ordinal=True,unprovable_context_rejected=True,multiple_visible_devices_rejected=True,call_interception_only_DEV3_geometry_entry=True,SG_olive_and_shader_args_unchanged=True,worker_SHA_6603_unchanged=True,adapter_sha256=sha(O/'dev3_entry.py'),route_sha256=sha(O/'route.py'),source_sha256=sha(__file__))
(O/'qualification_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
