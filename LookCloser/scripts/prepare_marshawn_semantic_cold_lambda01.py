"""Freeze an exact cold semanticR0/W64 final-distortion .1 control, never launch."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(c,out):
 parent=c/'semantic_actor_candidate_v2_20261106';out.mkdir(exist_ok=False)
 for name in ['code','source']:shutil.copytree(parent/name,out/name,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 worker=out/'code/LookCloser/scripts/train_marshawn_semantic_admission.py';old=worker.read_text();anchor='coarse_step=.00625, distortion_final=.01, seed=42,'
 if old.count(anchor)!=1:raise ValueError('Original coefficient validator changed')
 worker.unlink();worker.write_text(old.replace(anchor,'coarse_step=.00625, distortion_final=.1, seed=42,'))
 files={str(p.relative_to(out/'code')):sha(p)for p in(out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'};prior=json.loads((parent/'source_manifest.json').read_text())['files_sha256'];changed=[k for k in prior if prior[k]!=files[k]]
 if len(files)!=1495 or set(files)!=set(prior)or changed!=['LookCloser/scripts/train_marshawn_semantic_admission.py']:raise ValueError('Unexpected source delta')
 write(out/'source_manifest.json',dict(parent_source_manifest_sha256=sha(parent/'source_manifest.json'),files_sha256=files,changed_from_R0=changed,change='Only exact spec validator final distortion .01 to .1; all model/optimizer/sampler/ramp/plateau code byte-identical'))
 spec=json.loads((parent/'spec.json').read_text());oldroot=str(Path(spec['output']).parent);remote=str(Path(oldroot).with_name(out.name));spec=json.loads(json.dumps(spec).replace(oldroot,remote));spec['distortion_final']=.1;write(out/'spec.json',spec)
 plan=json.loads((parent/'controller_plan.json').read_text());plan=json.loads(json.dumps(plan).replace(oldroot,remote).replace('exp_actor_semantic_W64pad1_r0','exp_actor_semantic_W64pad1_dist01_r0'));write(out/'controller_plan.json',plan)
 for label,start in [('warmup',0),('ARM',4096)]:
  technical=dict(spec,output=remote+'/technical_'+label,technical_start_step=start,technical_pause_after=start+15,technical_memory_cap_gib=10.);write(out/('technical_'+label+'.json'),technical)
 proof=out/'inherited_proofs';proof.mkdir()
 for name in ['technical_complete.json','native_probe/probe.json','verification_warmup.json','verification_ARM.json']:
  dst=proof/name.replace('/','_');shutil.copy2(parent/name,dst)
 write(out/'candidate_policy.json',dict(parent='semanticR0/W64',only_scientific_change={'distortion_final':[.01,.1]},ramp_start=4096,ramp_end=12288,initial_coefficient=.01,minimum_trajectory_step=106316,stopping='unchanged validation plateau',geometry_quality_approved=False,sourcepoint=False,bag_correction=False,source4K=False,source_files=len(files),native_identity_proof_inherited=True,new_Adam_smoke_pending=True,authorized_by='root delegated autonomous technical and production after concrete checks'))
 print(json.dumps(dict(root=str(out),source_manifest_sha256=sha(out/'source_manifest.json'),spec_sha256=sha(out/'spec.json'))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
