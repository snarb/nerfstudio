"""Freeze a separate focused-frequency-only semanticR0 generation, never launch."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare(c,out):
 parent=c/'semantic_actor_candidate_v2_20261106';out.mkdir(exist_ok=False)
 for name in ['code','source']:shutil.copytree(parent/name,out/name,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 scripts=out/'code/LookCloser/scripts';worker=scripts/'train_marshawn_semantic_admission.py';text=worker.read_text()
 def replace(before,after):
  nonlocal text
  if text.count(before)!=1:raise ValueError('Worker anchor changed '+before[:60])
  text=text.replace(before,after)
 replace("'technical_memory_cap_gib'}", "'technical_memory_cap_gib', 'frequency_focus'}")
 replace("    gate = spec.get('semantic_admission', {})", "    if spec.get('frequency_focus') != dict(budget=2048,core_count=1229,band_count=410,uniform_count=409,private_seed=4201024):raise ValueError('Exact focused2048 specification required')\n    gate = spec.get('semantic_admission', {})")
 replace("'marshawn_tuning_metrics.py', 'marshawn_tuning_occupancy.py')]", "'marshawn_tuning_metrics.py', 'marshawn_tuning_occupancy.py', 'marshawn_frequency_sampling_proposal.py', 'marshawn_frequency_focus_runtime.py')]")
 replace("    install_capture_hook(model)", """    from marshawn_frequency_focus_runtime import FocusRuntime
    focus = FocusRuntime(pipe, sampler, spec['frequency_focus'])
    if resume is not None:
        if 'frequency_focus_state' not in state: raise ValueError('Missing private frequency state on resume')
        focus.load_state_dict(state['frequency_focus_state'])
        if focus.state_dict() != state['frequency_focus_state']: raise ValueError('Private frequency restoration differs')
        atomic_json(output / 'resume_frequency_verification.json', dict(passed=True, state_sha256=focus.digest(), restored_checkpoint_step=latest['step'], updates=focus.proposal.updates))
    install_capture_hook(model)""")
 replace("        state['sampling_state'] = sampler.state_dict()", "        state['sampling_state'] = sampler.state_dict()\n        state['frequency_focus_state'] = focus.state_dict()")
 replace("if step not in (0, 4096):", "if step not in (0, 1023, 4096):")
 worker.unlink();worker.write_text(text)
 for name in ['marshawn_frequency_sampling_proposal.py','marshawn_frequency_focus_runtime.py']:shutil.copy2(Path(__file__).parent/name,scripts/name)
 files={str(p.relative_to(out/'code')):sha(p)for p in(out/'code').rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'};prior=json.loads((parent/'source_manifest.json').read_text())['files_sha256'];changed=[k for k in prior if prior[k]!=files[k]];added=sorted(set(files)-set(prior))
 if len(files)!=1497 or changed!=['LookCloser/scripts/train_marshawn_semantic_admission.py']or len(added)!=2:raise ValueError('Unexpected source delta')
 write(out/'source_manifest.json',dict(parent_source_manifest_sha256=sha(parent/'source_manifest.json'),files_sha256=files,changed_from_R0=changed,added=added,change='Only focused frequency proposal/runtime worker integration; all original field/grid/sampler/pipeline/optimizer math identical'))
 spec=json.loads((parent/'spec.json').read_text());oldroot=str(Path(spec['output']).parent);remote=str(Path(oldroot).with_name(out.name));spec=json.loads(json.dumps(spec).replace(oldroot,remote));spec['frequency_focus']=dict(budget=2048,core_count=1229,band_count=410,uniform_count=409,private_seed=4201024);write(out/'spec.json',spec)
 plan=json.loads((parent/'controller_plan.json').read_text());plan=json.loads(json.dumps(plan).replace(oldroot,remote).replace('exp_actor_semantic_W64pad1_r0','exp_actor_semantic_frequencyfocus_r0'));write(out/'controller_plan.json',plan)
 for label,start in [('warmup',1023),('ARM',4096)]:write(out/('technical_'+label+'.json'),dict(spec,output=remote+'/technical_'+label,technical_start_step=start,technical_pause_after=start+31,technical_memory_cap_gib=10.))
 proof=out/'inherited_proofs';proof.mkdir()
 for name in ['technical_complete.json','native_probe/probe.json','verification_warmup.json','verification_ARM.json']:shutil.copy2(parent/name,proof/name.replace('/','_'))
 write(out/'candidate_policy.json',dict(only_scientific_change='frequency pixel proposal1229core/410band/409original uniform at same2048 globalcameraIDs',minimum_trajectory_step=106316,stopping='unchanged validation plateau',geometry_quality_approved=False,cold_parent='originalsemanticR0/W64/.00625/.01',main_body80_unchanged=True,private_rng_75941='continues saved independent state; historical global seedreset unchanged',source_files=1497,authorization='Root delegated technical then production after proof'))
 print(json.dumps(dict(root=str(out),source_sha256=sha(out/'source_manifest.json'),spec_sha256=sha(out/'spec.json'))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.campaign,a.output)
