"""After natural completion, bind protected selector/raw-min snapshots to finite read-only audits."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)
def main(root):
 contract=read(root/'contract.json');template=read(root/'policy.template.json')
 if contract['policy_template_sha256']!=sha(root/'policy.template.json')or contract['watcher_sha256']!=sha(Path(__file__)):raise ValueError('Final-audit contract drift')
 choices=Path(contract['final_choices'])
 while not(choices/'complete.json').exists():write(root/'watcher_state.json',dict(phase='waiting_natural_completion',time=time.time(),pid=os.getpid()));time.sleep(30)
 complete=read(choices/'complete.json');seen={};arms=[];bindings={}
 if complete.get('selector_and_plateau_unchanged')is not True:raise ValueError('Unexpected final selection protocol')
 for role in ['established_selector','raw_lpips_min']:
  row=complete['choices'][role];path=Path(row['path']);checkpoint=path/'checkpoint.ckpt'
  if sha(checkpoint)!=row['checkpoint_sha256']or sha(path/'grid.npz')!=row['grid_sha256']:raise ValueError('Protected final snapshot changed')
  if row['checkpoint_sha256']in seen:bindings[role]=seen[row['checkpoint_sha256']];continue
  arm=dict(template['models'][0]);name=contract['name_prefix']+('_selected'if role=='established_selector'else'_rawmin');arm.update(name=name,gate_step=row['step'],expected_step=row['step'],checkpoint_override=str(checkpoint));arms.append(arm);bindings[role]=name;seen[row['checkpoint_sha256']]=name
 policy=dict(template,models=arms);write(root/'policy.json',policy);write(root/'release.json',dict(approved_readonly=True,policy_sha256=sha(root/'policy.json'),authorization=contract['authorization'],final_choices_sha256=sha(choices/'complete.json'),choice_roles=bindings,quality_promotion=False))
 with(root/'audit.log').open('w')as f:subprocess.run(['python3',str(root/'helpers/run_marshawn_semantic_gate_audits.py'),'--root',str(root)],stdout=f,stderr=subprocess.STDOUT,check=True)
 caps={}
 for arm in arms:
  out=root/arm['name'];target=out/'sampling'
  with(out/'sampling.log').open('w')as f:subprocess.run([policy['python'],str(root/'helpers/audit_marshawn_saved_sampling.py'),'--source',str(out),'--output',str(target)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),check=True)
  stats=read(target/'report.json')['profiles'];caps[arm['name']]={k:dict(native_cap_hits=v['summary']['native']['at_cap'],native_max_samples=v['summary']['native']['samples']['max'],dense_tail_max=v['summary']['dense_weight_after_native_last_end']['max'],allcells_cap_hits=v['summary']['all_cells']['at_cap'])for k,v in stats.items()}
 write(root/'final_review_ready.json',dict(complete=True,choice_roles=bindings,sampling=caps,native_cap_counterfactual_required=any(v['native_cap_hits']for a in caps.values()for v in a.values()),production_model_changed=False,quality_promotion=False,contract_sha256=sha(root/'contract.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
