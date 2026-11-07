"""Verify actual nonempty private side-channel state across ordinary Adam checkpoints."""
import argparse,hashlib,json
from pathlib import Path
import torch

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main(root):
 records={}
 for label,start,boundary in [('warmup',1023,1024),('ARM',4096,4096)]:
  p=root/('technical_'+label);before=torch.load(p/'technical_before.ckpt',map_location='cpu',weights_only=False);after=torch.load(p/'checkpoints'/f'step-{start+63:09d}.ckpt',map_location='cpu',weights_only=False)
  b=before['tuning_runtime']['frequency_focus_state'];a=after['tuning_runtime']['frequency_focus_state'];r=json.loads((p/'resume_frequency_verification.json').read_text())
  if before['step']!=start+31 or after['step']!=start+63 or not r['passed']or r['state_sha256']!=digest(b)or r['restored_checkpoint_step']!=before['step']:raise ValueError('Private state restoration mismatch')
  if b['proposal']['updates']!=1 or a['proposal']['updates']!=1 or b!=a or b['last_step']!=boundary:raise ValueError('Actual frequency boundary/nonempty private RNG missing')
  from marshawn_frequency_sampling_proposal import FrequencyProposal
  import numpy as np
  # Private RNG state is not the seed/constructor state; restoring it is essential.
  initial=np.random.default_rng(4201024).bit_generator.state
  if b['proposal']['rng']==initial:raise ValueError('Private generator never advanced')
  result=json.loads((root/('verification_'+label+'.json')).read_text())
  if not result['passed']:raise ValueError('Actual Adam proof failed')
  records[label]=dict(actual_frequency_boundary=boundary,private_updates=1,private_state_sha256=digest(b),resume_receipt=r,successful_adam_before=result['before']['adam_steps'],successful_adam_after=result['after']['adam_steps'])
  del before,after
 out=dict(passed=True,ordinary_checkpoint_runtime_state=True,nonempty_private_rng_restored=True,counterfactual_technical_steps_not_quality=True,records=records)
 (root/'verification_frequency_private.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
