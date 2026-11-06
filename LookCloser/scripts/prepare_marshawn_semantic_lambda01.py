"""Freeze lambda0.1 semantic30376 continuation; only extend v2 coefficient guards."""
import argparse,hashlib,json,os,shutil,difflib
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(c,out):
 parent=c/'semantic_late_distortion_candidate_v2';out.mkdir(exist_ok=False);shutil.copytree(parent/'code',out/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 sources=['LookCloser/scripts/train_marshawn_semantic_late_distortion.py','LookCloser/scripts/marshawn_semantic_late_factory.py'];diff=[]
 for name in sources:
  p=out/'code'/name;before=p.read_text();after=before.replace('not in (.01, 1.)','not in (.01, .1, 1.)').replace('not in (.01,1.)','not in (.01,.1,1.)').replace('reviewed .01/1 diagnostic pair','reviewed .01/.1/1 diagnostic branches').replace('reviewed .01/1 semantic pair','reviewed .01/.1/1 semantic branches')
  assert before!=after and before.count('not in (.01,')==1,name
  diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=name+' v2',tofile=name+' lambda0p1'));p.unlink();p.write_text(after)
 (out/'source_guard_delta.diff').write_text(''.join(diff))
 inv=json.loads((parent/'source_inventory.json').read_text());files={name:sha(out/'code'/name)for name in inv['files']};changed=sorted(name for name,digest in files.items()if digest!=inv['files'][name]);assert changed==sorted(sources)
 original=json.loads((c/'semantic_actor_candidate_v2_20261106/source_manifest.json').read_text())['files_sha256'];assert set(changed).isdisjoint(original)
 write(out/'source_inventory.json',dict(parent_files=inv['parent_files'],parent_byte_identical=True,validated_v2_source_inventory_sha256=sha(parent/'source_inventory.json'),changed_from_v2=changed,only_coefficient_guard_extension=True,files=files))
 remote='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/'+out.name
 for mode in ['zero','diagnostic']:
  spec=json.loads((parent/'specs/diagnostic_treatment1/spec.json').read_text());spec.update(mode=mode,distortion_coefficient=.1,output=remote+'/runs/'+mode+'_lambda0p1',release=remote+'/specs/'+mode+'_lambda0p1/root_release.json',worker_sha256=files[sources[0]],frozen_source_hashes={remote+'/code/'+name:digest for name,digest in files.items()});dest=out/'specs'/f'{mode}_lambda0p1';dest.mkdir(parents=True)
  write(dest/'spec.json',spec);write(dest/'root_release.template.json',dict(reviewer='root',technical_authorized=False,diagnostic_authorized=False,spec_sha256=sha(dest/'spec.json'),pending_review=True))
 inherited=out/'inherited_v2';inherited.mkdir()
 for name in ['anchor_audit.json','technical_policy.json','technical_complete.json','source_inventory.json']:shutil.copy2(parent/name,inherited/name)
 shutil.copy2(parent/'specs/diagnostic_control001/spec.json',inherited/'control_spec.json')
 write(out/'candidate_policy.json',dict(prepared_only=True,production_started=False,coefficient=.1,source_checkpoint_sha256=json.loads((out/'specs/zero_lambda0p1/spec.json').read_text())['parent_checkpoint_sha256'],fork_source='Exact evaluated semantic30376 fullstate, never zero/technical continuation',reuse_control='Unchanged v2 .01 diagnostic, only after zero/state/cache/ray binding equality',gates=[30876,31376,32376,34376],training_code_math_byte_identical=True,guard_only_delta=True,additional64Adam_not_required_by_root=True,zero_gate_pending=True))
 print(json.dumps(dict(root=str(out),files=len(files),source_inventory_sha256=sha(out/'source_inventory.json'),zero_spec_sha256=sha(out/'specs/zero_lambda0p1/spec.json'),diagnostic_spec_sha256=sha(out/'specs/diagnostic_lambda0p1/spec.json'))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
