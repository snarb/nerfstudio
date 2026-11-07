"""CPU proof that the paired quarter continuation differs only in its future FR value."""
import argparse,dataclasses,hashlib,json,os,sys
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 code=a.root/'code';sys.path[:0]=[str(code),str(code/'LookCloser/scripts')]
 import torch,yaml
 import marshawn_quarter_FR_factory as factory
 if Path(factory.__file__).resolve()!=(code/'LookCloser/scripts/marshawn_quarter_FR_factory.py').resolve():raise ValueError('Factory import escaped frozen snapshot')
 inventory=json.loads((a.root/'source_inventory.json').read_text())
 for name,digest in inventory['files_sha256'].items():
  if sha(code/name)!=digest:raise ValueError('Frozen candidate drift')
 anchor=json.loads((a.root/'anchor_audit.json').read_text());parent=json.loads((a.archive/'spec.json').read_text());configs={}
 for name,strength in [('original_FR03',.3),('hold_FR1',1.)]:
  cfg=factory.build_config(parent,anchor,a.archive/'checkpoint.ckpt',a.root/'normalized_output',strength)
  configs[name]=json.loads(json.dumps(dataclasses.asdict(cfg),default=str,sort_keys=True));(a.root/(name+'_normalized_config.yml')).write_text(yaml.dump(cfg,sort_keys=True))
  assert cfg.pipeline.grid_update_interval==1024 and cfg.pipeline.grid_update_batch_size==2048
  assert cfg.pipeline.model.feature_reweighting_strength==1. and cfg.pipeline.model.distortion_loss_mult==.01
 differences=[]
 def walk(x,y,path=''):
  if isinstance(x,dict)and isinstance(y,dict):
   if set(x)!=set(y):raise ValueError('Configuration keys differ')
   for key in x:walk(x[key],y[key],path+'.'+key if path else key)
  elif isinstance(x,list)and isinstance(y,list):
   if len(x)!=len(y):raise ValueError('Configuration list differs')
   for i,(u,v)in enumerate(zip(x,y)):walk(u,v,path+f'[{i}]')
  elif x!=y:differences.append(dict(path=path,original=x,alternative=y))
 walk(configs['original_FR03'],configs['hold_FR1'])
 if differences!=[dict(path='pipeline.feature_reweighting_after_switch',original=.3,alternative=1.)]:raise ValueError(differences)
 if torch.cuda.is_initialized():raise ValueError('CPU proof initialized CUDA')
 result=dict(passed=True,differences=differences,source_inventory_sha256=sha(a.root/'source_inventory.json'),anchor_sha256=sha(a.root/'anchor_audit.json'),actual_factory_path=str(Path(factory.__file__).resolve()),factory_sha256=sha(factory.__file__),CUDA_initialized=False,model_instantiated=False,optimizer_instantiated=False,original1495_unchanged=inventory['parent_files_unchanged'],future_original_seed_reset=dict(step=75941,seed=42,both_arms=True),expected_first_frequency_refresh=76800,technical_steps=[75941,75972,76004],diagnostic_gates=[76440,76940,77940,79940])
 (a.root/'config_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(a.root/'config_parity.json')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);main(p.parse_args())
