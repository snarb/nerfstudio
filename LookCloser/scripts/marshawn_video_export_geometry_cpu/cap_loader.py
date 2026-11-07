"""Read original host-native NEW140k artifacts; reject retired V/SV models."""
from pathlib import Path
import json,hashlib
CAP=140000
ROOTS={Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107'),Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_140k_20261107')}
SEAL='d873cd61ad5c00f9501f2f6c477cb42f962d4d15591f5785830e0aeae381ec14'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def owned(root,p):
 p=Path(p).resolve();p.relative_to(root.resolve());return p
def load_pipeline(frame_root,checkpoint_kind='actor_lpips',device='cuda:0'):
 root=Path(frame_root).resolve();campaign=root.parents[1]
 if campaign not in ROOTS:raise ValueError('Retired or unknown production model root')
 if sha(campaign/'code_seal.json')!=SEAL:raise ValueError('Unqualified hard140k code release')
 receipt=json.loads((root/'frame_completion.json').read_text());result=json.loads((root/'train/result.json').read_text())
 if receipt['technical']is not False or result['technical_smoke']is not False or result['hard_max_iterations']!=CAP or not 0<int(result['completed_iterations'])<=CAP:raise ValueError('Overcap/incomplete/technical production revision rejected')
 if result['completed_iterations']!=result['final_step']+1 or int(receipt['final_step'])!=int(result['final_step']):raise ValueError('Training iteration accounting mismatch')
 choice=receipt['selectors'][checkpoint_kind];checkpoint=owned(root/'train/checkpoints',choice['checkpoint']);config=owned(root,receipt['config'])
 if not 0<int(choice['step'])<=CAP or sha(checkpoint)!=choice['checkpoint_sha256']:raise ValueError('Selected checkpoint outside current cap/source')
 if Path(receipt['code']).resolve()!=campaign/'code':raise ValueError('Model code points outside its training campaign')
 from marshawn_video_train import load_pipeline as canonical
 pipe,binding=canonical(root,checkpoint_kind=checkpoint_kind,device=device)
 if Path(binding['checkpoint']).resolve()!=checkpoint or Path(binding['config']).resolve()!=config:raise ValueError('Canonical loader chose other artifacts')
 if Path(pipe.datamanager.config.dataparser.data).resolve()!=root/'data':raise ValueError('Production model still references foreign dataset')
 binding.update(hard_iteration_cap=CAP,completed_iterations=result['completed_iterations'],active_hard140k_revision=True,retired_overcap_family=False,
  production_campaign=str(campaign),source_code_seal_sha256=SEAL,original_checkpoint_and_YAML_paths_preserved=True)
 return pipe,binding
