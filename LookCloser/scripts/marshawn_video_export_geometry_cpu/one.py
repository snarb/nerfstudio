"""Host-native frozen renderer with a source-visibility input adapter only."""
from pathlib import Path
import argparse,os,sys,json,hashlib

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def source_eligibility(arrays,rgb_valid):
 import numpy as np
 required={'W64','valid','source_ineligible','unknown'}
 if not required<=set(arrays):raise ValueError('Qualified source semantic flags missing')
 unknown=arrays['unknown']
 if 'source_unknown'in arrays:
  unknown=arrays['source_unknown']
  if 'qualified_actor_rgb'not in arrays:raise ValueError('Source UNKNOWN override has no visible-actor qualification')
  qualified=arrays['qualified_actor_rgb']
  if unknown.dtype!=np.bool_ or qualified.dtype!=np.bool_ or unknown.shape!=rgb_valid.shape or qualified.shape!=rgb_valid.shape:raise ValueError('Wrong source-visibility raster/schema')
  if np.any(arrays['unknown']&~unknown&~qualified):raise ValueError('Unqualified geometry UNKNOWN removed from RGB source veto')
 eligible=(arrays['W64']>0)&arrays['valid']&~arrays['source_ineligible']&~unknown&rgb_valid
 if eligible.shape!=rgb_valid.shape or np.any(eligible&(arrays['source_ineligible']|unknown)):raise ValueError('Unsafe source eligibility')
 # actor_ownership_uncertain is an identity/geometric flag. It is deliberately
 # not a veto on source RGB or a replacement for the frozen W64 weights.
 return eligible

def main():
 p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--path',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--cpu',action='store_true');a=p.parse_args()
 # Python3.8 compatibility, identical Path.relative_to semantics; no geometry
 # or radiometry code is changed.
 if not hasattr(Path,'is_relative_to'):
  def is_relative_to(self,other):
   try:self.relative_to(other);return True
   except ValueError:return False
  Path.is_relative_to=is_relative_to
 sys.path[:0]=[str(a.code),str(a.code/'LookCloser/scripts'),str(a.spec.parent),str(Path(__file__).parent)]
 spec=json.loads(a.spec.read_text())
 if spec['source_visibility_adapter_sha256']!=file_sha(__file__):raise ValueError('Renderer input adapter changed')
 if not a.cpu:
  device=os.environ.get('CUDA_VISIBLE_DEVICES','');fd=os.environ.get('LOOKCLOSER_GPU_LEASE_FD')
  if not device.startswith('GPU-')or','in device or fd is None:raise ValueError('Host renderer requires physical UUID and inherited fleet lease')
  os.fstat(int(fd));os.environ['MARSHAWN_RENDER_GPU_UUID']=device
 import marshawn_video_render as renderer
 renderer.source_eligibility=source_eligibility
 original_source_get=renderer.Sources.get
 def source_get(self,name):
  result=original_source_get(self,name)
  self.inventory[name]['eligibility']='W64>0 AND valid AND NOTsource_ineligible AND NOT(source_unknown if present else geometry_unknown) AND calibratedRGBvalid'
  self.inventory[name]['source_visibility_adapter_sha256']=file_sha(__file__)
  return result
 renderer.Sources.get=source_get
 if not a.cpu:
  import importlib
  if spec.get('hard_iteration_cap')!=140000:raise ValueError('Production render must bind current hard140k release')
  loader=importlib.import_module(spec['pipeline_module'])
  if file_sha(loader.__file__)!=spec['production_loader_sha256']:raise ValueError('Production revision loader changed')
  original=loader.load_pipeline
  def capped_loader(*args,**kwargs):
   pipe,binding=original(*args,**kwargs)
   if not 0<int(binding['step'])<=140000:raise ValueError('Overcap or invalid production checkpoint rejected')
   return pipe,binding
  loader.load_pipeline=capped_loader
 sys.argv=[str(renderer.__file__),'--spec',str(a.spec),'--path',str(a.path),'--index',str(a.index)];renderer.main()

if __name__=='__main__':main()
