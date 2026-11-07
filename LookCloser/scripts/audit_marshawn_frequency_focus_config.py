"""CPU actual-config parity for the isolated frequency sampling distribution arm."""
import argparse,json,sys,hashlib,dataclasses
from pathlib import Path

def canonical(v):
 if isinstance(v,type):return v.__module__+'.'+v.__qualname__
 if dataclasses.is_dataclass(v):return {f.name:canonical(getattr(v,f.name))for f in dataclasses.fields(v)}
 if isinstance(v,dict):return {str(k):canonical(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [canonical(x)for x in v]
 if isinstance(v,Path):return str(v)
 if v is None or isinstance(v,(str,int,float,bool)):return v
 if callable(v):return v.__module__+'.'+v.__qualname__
 return repr(v)
def main(root,parent):
 sys.path[:0]=[str(root/'code'),str(root/'code/LookCloser/scripts')]
 import torch
 import train_marshawn_semantic_admission as worker
 spec=json.loads((root/'spec.json').read_text());before=json.loads((parent/'spec.json').read_text());worker.validate_spec(spec)
 oldroot=str(Path(before['output']).parent);newroot=str(Path(spec['output']).parent)
 normalized=json.loads(json.dumps(spec).replace(newroot,oldroot));focus=normalized.pop('frequency_focus')
 if normalized!=before:raise ValueError('Scientific spec changed beyond frequency proposal')
 old=canonical(worker.build_config(before));new=json.loads(json.dumps(canonical(worker.build_config(spec))).replace(newroot,oldroot))
 if new!=old:raise ValueError('Actual model/pipeline/optimizer config differs')
 bad=[]
 for key,value in [('coarse_step',.003125),('distortion_final',.1),('width',32),('radius',1)]:
  try:worker.validate_spec(dict(spec,**{key:value}))
  except ValueError:bad.append(key)
  else:raise ValueError('Bundled change accepted')
 if torch.cuda.is_initialized():raise ValueError('CPU config proof initialized CUDA')
 out=dict(passed=True,actual_config_difference={},only_new_scientific_spec={'frequency_focus':focus},rejected_bundles=bad,field_sampler_pipeline_optimizer_all_identical=True,CUDA_initialized=False)
 (root/'config_parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();main(a.root,a.parent)
