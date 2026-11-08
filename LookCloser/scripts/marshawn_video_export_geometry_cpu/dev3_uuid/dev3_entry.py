"""OPS entry adapter: prove Torch's current CUDA context UUID on older Torch."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,runpy,sys,uuid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def context_uuid(torch):
 torch.cuda.init()
 if torch.cuda.device_count()!=1 or torch.cuda.current_device()!=0:raise RuntimeError('Exactly logical CUDA0 required')
 # Force the logical Torch device's current primary context without RNG/model work.
 marker=torch.empty(1,device='cuda:0');torch.cuda.synchronize()
 driver=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();getctx=driver.cuCtxGetDevice;getctx.argtypes=[ctypes.POINTER(ctypes.c_int)];getctx.restype=ctypes.c_int
 code=getctx(ctypes.byref(device))
 if code:raise RuntimeError('Cannot prove Torch current CUDA context device: '+str(code))
 class UUID(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
 value=UUID();getuuid=getattr(driver,'cuDeviceGetUuid_v2',None)or driver.cuDeviceGetUuid;getuuid.argtypes=[ctypes.POINTER(UUID),ctypes.c_int];getuuid.restype=ctypes.c_int;code=getuuid(ctypes.byref(value),device.value)
 if code:raise RuntimeError('Current-context physical UUID unavailable: '+str(code))
 actual='GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)));del marker
 return actual,device.value

def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=Path,required=True);p.add_argument('--expected-sha256',required=True);a,rest=p.parse_known_args();rest=rest[1:]if rest[:1]==['--']else rest
 if sha(a.worker)!=a.expected_sha256 or a.expected_sha256!='6603aa06dc8ef2df2cd89cada087729eaa40bd81be5b5c981e2e3e2369539eba':raise ValueError('Only unchanged released GPU shader accepted')
 code=Path(rest[rest.index('--code')+1]);sys.path[:0]=[str(code),str(code/'LookCloser/scripts'),str(a.worker.parent)]
 import marshawn_gpu_isolation as isolation
 original=isolation.verify_cuda
 def verify(expected,torch):
  environment=isolation.require_environment(expected)
  dependencies=Path(os.environ['TUNING_DEPENDENCIES_FILE'])
  if sha(dependencies)!=os.environ['TUNING_DEPENDENCIES_SHA256']:raise RuntimeError('Supervisor dependency receipt changed')
  receipt_path=dependencies.with_name('receipt.json');lease=isolation.require_controller_lease(receipt_path);receipt=json.loads(receipt_path.read_text())
  fd=int(os.environ['LOOKCLOSER_GPU_LEASE_FD']);held=os.fstat(fd);target=Path('/tmp')/('lookcloser_tuning_gpu_%s_%s.lock'%(os.getuid(),receipt['gpu']));owned=target.stat()
  if(held.st_dev,held.st_ino)!=(owned.st_dev,owned.st_ino):raise RuntimeError('Inherited FD is not supervisor physical GPU lock')
  if hasattr(torch.cuda.get_device_properties(0),'uuid'):return dict(original(expected,torch),lease=lease,adapter_sha256=sha(__file__))
  actual,device=context_uuid(torch)
  if actual.lower()!=expected.lower()or actual.lower()!=receipt['gpu_uuid'].lower():raise RuntimeError('Torch current-context UUID differs from leased supervisor UUID')
  return dict(**environment,visible_device_count=1,logical_current_device=0,actual_physical_uuid=actual,name=torch.cuda.get_device_properties(0).name,UUID_proof='cuCtxGetDevice_of_Torch_current_context_then_cuDeviceGetUuid',driver_context_device=device,lease=lease,adapter_sha256=sha(__file__),shader_sha256=a.expected_sha256)
 isolation.verify_cuda=verify;sys.argv=[str(a.worker),*rest];runpy.run_path(str(a.worker),run_name='__main__')
if __name__=='__main__':main()
