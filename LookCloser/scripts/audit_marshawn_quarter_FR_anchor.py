"""CPU-only seven-state/schema2 anchor for the isolated quarter75940 FR fork."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main(a):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 spec=importlib.util.spec_from_file_location('frworker',a.worker);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 p=a.archive;checkpoint=p/'checkpoint.ckpt';state=torch.load(checkpoint,map_location='cpu',weights_only=False);grid=json.loads((p/'grid.json').read_text());parent=json.loads((p/'spec.json').read_text())
 if state['step']!=75940 or grid['checkpoint_sha256']!=sha(checkpoint)or grid['schema']!=2:raise ValueError('Wrong boundary/schema/checkpoint')
 if state['tuning_runtime']['boundary_rng_reset']is not False or 'frequency_focus_state'in state['tuning_runtime']:raise ValueError('Original pre-reset global-frequency quarter required')
 capture=state['actual_occupancy_capture'];pipe=state['pipeline'];find=lambda suffix:next(v for k,v in pipe.items()if k.endswith(suffix));array=lambda x:x.detach().cpu().numpy()
 with np.load(p/'grid.npz',allow_pickle=False)as z:
  checks=dict(core_capture=np.array_equal(z['core'],array(capture['core'])),active_capture=np.array_equal(z['active'],array(capture['active'])),active_pipeline=np.array_equal(z['active'],array(find('occupancy_grid.binaries'))),EMA_pipeline=np.array_equal(z['ema'],array(find('occupancy_grid.occs')).reshape(z['ema'].shape)),learned_core_pipeline=np.array_equal(z['core'].reshape(-1),array(find('semantic_learned_core')).reshape(-1)))
  if z['core'].shape!=z['admission'].shape:raise ValueError('Saved support/core shape differs')
  checks['active_is_core_and_admission']=np.array_equal(z['active'],z['core']&z['admission'])
  checks['mapped_admission_sha']=hashlib.sha256(z['admission'].tobytes()).hexdigest()==capture['semantic_receipt']['mapped_bool_sha256']
  checks['receipt_matches_capture']=grid['semantic_receipt']==capture['semantic_receipt']
 if not all(checks.values()):raise ValueError('Actual saved semantic grid mismatch')
 adam=m.adam_steps(state['optimizers'])
 if not all(v and min(v)>0 for v in adam.values()):raise ValueError('NonemptyAdam source required')
 source_files=[p/name for name in ['checkpoint.ckpt','grid.json','grid.npz','config.yml','spec.json','source_manifest.json','release.json']]+sorted((p/'renders').glob('*'))
 source_hashes={str(f):sha(f)for f in source_files if f.is_file()}
 keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture']
 result=dict(CPU_builder_sha256=sha(Path(__file__)),digest_worker_sha256=sha(a.worker),anchor_status='EVALUATED_INTERMEDIATE',producer_continues=True,producer_continues_means='At immutable anchor capture; original full trajectory is not altered by this package',step=75940,source_hashes=source_hashes,fullstate_hashes={k:m.tree_digest(state[k])for k in keys},actual_Adam=adam,scheduler=state['schedulers'],scaler=state['scalers'],optimizer_hyperparameters={k:v['param_groups']for k,v in state['optimizers'].items()},sampling_state=state['tuning_runtime']['sampling_state'],historical_cache_order_observed=False,source_eval=json.loads((p/'renders/metrics.json').read_text()),semantic_grid_checks={k:bool(v)for k,v in checks.items()},semantic_capture_metadata={k:v for k,v in capture.items()if not torch.is_tensor(v)},semantic_pipeline_keys=[k for k in pipe if 'semantic'in k],CUDA_initialized=torch.cuda.is_initialized(),model_instantiated=False,optimizer_instantiated=False,ordinary_resume_changed=False,technical_or_production_launched=False)
 if result['CUDA_initialized']:raise RuntimeError('CPU audit initialized CUDA')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
