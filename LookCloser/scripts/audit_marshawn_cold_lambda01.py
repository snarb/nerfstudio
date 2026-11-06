"""CPU actual-spec/config/ramp parity for the frozen one-factor cold coefficient."""
import argparse,ast,dataclasses,hashlib,json,re,sys
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
 rejected=[]
 for key,value in [('distortion_final',.01),('distortion_final',1.),('width',32),('radius',1),('objective','full')]:
  bad=dict(spec,**{key:value})
  try:worker.validate_spec(bad)
  except ValueError:rejected.append([key,value])
  else:raise ValueError('Unreviewed combination accepted')
 oldroot=str(Path(before['output']).parent);newroot=str(Path(spec['output']).parent)
 normalized=json.loads(json.dumps(spec).replace(newroot,oldroot));diff={k:[before[k],normalized[k]]for k in before if before[k]!=normalized[k]}
 if diff!={'distortion_final':[.01,.1]}:raise ValueError('Unexpected recipe change')
 cfg_before=canonical(worker.build_config(before));cfg_after=canonical(worker.build_config(spec));normalized_cfg=json.loads(json.dumps(cfg_after).replace(newroot,oldroot))
 if cfg_before!=normalized_cfg:raise ValueError('Actual initial config differs beyond paths')
 text=(root/'code/LookCloser/scripts/train_marshawn_semantic_admission.py').read_text();lines=text.splitlines();fraction=next(x.strip()for x in lines if x.strip().startswith('fraction = min(1.'));coefficient=next(x.strip().split(' = ',1)[1]for x in lines if x.strip().startswith('model.config.distortion_loss_mult ='))
 table=[]
 for step in [0,4095,4096,8192,12287,12288,75941,106316]:
  local=dict(step=step,spec=spec);exec(fraction,{},local);value=eval(coefficient,{},local);expected=.01+(.1-.01)*min(1.,max(0.,(step-4096)/8192));assert value==expected;table.append(dict(step=step,coefficient=value))
 if torch.cuda.is_initialized():raise ValueError('CPU proof initialized CUDA')
 report=dict(passed=True,actual_config_equal_after_path_normalization=True,scientific_spec_difference=diff,rejected_combinations=rejected,runtime_ramp=table,runtime_fraction_source=fraction,runtime_coefficient_source=coefficient,CUDA_initialized=False,source_manifest_sha256=hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),source_code_identical_except_coefficient_validator=True)
 (root/'config_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();main(a.root,a.parent)
