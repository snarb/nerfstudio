"""Clone qualified MG runtime unchanged; bind W32 point support, never release."""
import argparse,ast,copy,hashlib,json,os,shutil
from pathlib import Path
ARMS=('original_mapped','source_margin002')
GEN='external_guard_margin_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def replace(x,old,new):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,dict):return {replace(k,old,new):replace(v,old,new)for k,v in x.items()}
 if isinstance(x,list):return [replace(v,old,new)for v in x]
 return x

def prepare(c,root,remote):
 remote=Path(remote);old=c/'semantic_margin_continuation_candidate_v1';oldremote=remote.parent/old.name;actual=c/'independent_occupancy/source_mask_W32_counterfactual_v1';proof=read(actual/'quarter_distortion_raw126316/complete.json');assert proof['passed']and proof['seven_states_unchanged']and proof['optimizer_updates']==0
 root.mkdir(parents=True,exist_ok=False);shutil.copytree(old/'code',root/'code',copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(old/'source_manifest.json',root/'source_manifest.json');manifest=read(root/'source_manifest.json');assert len(manifest['files_sha256'])==1506
 for p,h in manifest['files_sha256'].items():assert sha(root/'code'/p)==h,p
 shutil.copytree(old/'anchor',root/'anchor',ignore=shutil.ignore_patterns('checkpoint.ckpt'));anchor=replace(read(old/'anchor_audit.json'),str(oldremote),str(remote));anchor['inherited_anchor_audit_sha256']=sha(old/'anchor_audit.json');write(root/'anchor_audit.json',anchor)
 protocol=read(old/'protocol.json');protocol.update(status='CPU_EXECUTABLE_PREPARATION_NO_GPU_RELEASE',single_scientific_factor='original mapped versus W32SourcePad1+.002 sharedfield density admission; W64 RGB confidence/pools remain unchanged',source_geometry_width_native_pixels=32,source_key='semantic_gate_pad1',source_grid_sha256=sha(actual/'sources/W32.npz'),original_RGB_confidence_width=64,source_actual_Adam=126262,goal='Measure whether genuine gray contour targets recover with adaptation while cleanup remains; no alpha1 or target deletion',source_review='All109 sparsepositions retained;91 proxy misses root-reviewed distalbagtip/floor. No labels or whole-ray negatives changed.',external_guard='Same qualified priorMG external guardian safety core; separate generation/jobprefix; all release templates false')
 protocol['fresh_setup'][3]='Own branchzero: control originalA, treatment actual auditedW32SourcePad1+.002 at126316; never compare treatmentzero to A.';write(root/'protocol.json',protocol)
 (root/'margin').mkdir();shutil.copy2(actual/'sources/W32.npz',root/'margin/source_grid.npz');shutil.copy2(old/'margin/probe_positions.npz',root/'margin/probe_positions.npz');m=replace(read(old/'margin/manifest.json'),str(oldremote),str(remote));m.update(source_grid_sha256=sha(root/'margin/source_grid.npz'),source_geometry_support_width_native_pixels=32,original_RGB_loss_width=64,source_key='semantic_gate_pad1',source_support_proof_sha256=sha(actual/'fourway_CPU.json'),source_manual_review_sha256=sha(actual/'root_manual_review.json'),inherited_W32_readonly_proof_sha256=sha(actual/'quarter_distortion_raw126316/complete.json'));write(root/'margin/manifest.json',m)
 (root/'source_evidence').mkdir()
 for src,name in [(actual/'fourway_CPU.json','fourway_CPU.json'),(actual/'root_manual_review.json','root_manual_review.json'),(actual/'quarter_distortion_raw126316/complete.json','W32_readonly_complete.json'),(actual/'policy.json','W32_readonly_policy.json'),(actual/'release.json','W32_readonly_release.json')]:shutil.copy2(src,root/'source_evidence'/name)
 for arm,phase in [('original_mapped','native_A'),('source_margin002','source_W32')]:
  src=actual/'quarter_distortion_raw126316'/phase/'renders/step-000126316';dest=root/'zero_references'/arm;shutil.copytree(src,dest/'renders');ref=dict(arm=arm,source_checkpoint_sha256=proof['source_checkpoint_sha256'],source_geometry_width=32 if arm=='source_margin002'else None,render_directory=str(remote/'zero_references'/arm/'renders'),render_sha256={p.name:sha(p)for p in(dest/'renders').glob('*.png')},metrics=read(dest/'renders/metrics.json'),artifacts_sha256={str(remote/'zero_references'/arm/'renders'/p.name):sha(p)for p in(dest/'renders').iterdir()if p.is_file()},inherited_readonly_complete_sha256=sha(actual/'quarter_distortion_raw126316/complete.json'));write(dest/'reference.json',ref)
 for phase in ['zero','technical','production']:
  for arm in ARMS:
   dest=root/'specs'/f'{phase}_{arm}';s=replace(read(old/'specs'/f'{phase}_{arm}/spec.json'),str(oldremote),str(remote));s.update(anchor_audit_sha256=sha(root/'anchor_audit.json'),margin_manifest_sha256=sha(root/'margin/manifest.json'),zero_reference_sha256=sha(root/'zero_references'/arm/'reference.json'));write(dest/'spec.json',s);write(dest/'root_release.template.json',dict(reviewer='root',technical_authorized=False,production_authorized=False,spec_sha256=sha(dest/'spec.json'),margin_manifest_sha256=s['margin_manifest_sha256'],zero_reference_sha256=s['zero_reference_sha256']))
  plan=replace(read(old/GEN/(phase+'_plan.json')),str(oldremote),str(remote));plan=replace(plan,'margin126316_','sourceW32margin126316_');write(root/(phase+'_plan.json'),plan)
 shutil.copy2(old/'audit_cpu.py',root/'audit_cpu.py');write(root/'preparation.json',dict(CPU_only=True,GPU_started=False,releases_created=False,science_files_byte_identical_to_qualified_MG=1506,original1495_unchanged=True,source_manifest_sha256=sha(root/'source_manifest.json'),worker_sha256=sha(root/'code/LookCloser/scripts/train_marshawn_margin_continuation.py'),source_geometry_width=32,RGB_confidence_width=64,parent_fresh126316=True,source_checkpoint_sha256=proof['source_checkpoint_sha256'],source_actual_Adam=126262,phase_gates='ZERO then32+32 then separatelyreleasedfreshproduction; no inheritedW64 actualGPUproof mislabel'))
 print(root)

def guardian(c,root,remote):
 remote=Path(remote);old=c/'semantic_margin_continuation_candidate_v1';oldremote=remote.parent/old.name;source=(old/GEN/'guardian.py').read_text();text=source.replace('margin126316_production_','sourceW32margin126316_production_');out=root/GEN;out.mkdir(exist_ok=False);(out/'guardian.py').write_text(text)
 policy=replace(replace(read(old/GEN/'policy.json'),str(oldremote),str(remote)),str(old),str(root));policy['guardian_sha256']=sha(out/'guardian.py');policy['root_receipts']={p:sha(root/p)for p in ['protocol.json','CPU_readiness.json']}
 for phase in ['zero','technical','production']:shutil.copy2(root/(phase+'_plan.json'),out/(phase+'_plan.json'))
 bindings=[root/p for p in ['source_manifest.json','margin/manifest.json','anchor_audit.json','protocol.json','CPU_readiness.json']]+list((root/'specs').rglob('spec.json'))+list((root/'zero_references').rglob('reference.json'))+list((root/'source_evidence').glob('*.json'))+[out/(p+'_plan.json')for p in ['zero','technical','production']]+[out/'guardian.py'];policy['prepared_bindings']={str(p.relative_to(root)):sha(p)for p in bindings};write(out/'policy.json',policy)
 for phase in ['zero','technical','production']:
  release=dict(reviewer='root',external_guard_authorized=False,policy_sha256=sha(out/'policy.json'),authorized_phases=[phase],protocol_sha256=sha(root/'protocol.json'),CPU_readiness_sha256=sha(root/'CPU_readiness.json'))
  if phase!='zero':release[('zero'if phase=='technical'else'technical')+'_verification_sha256']='ROOT_BIND_AFTER_PREDECESSOR_PASS'
  write(out/(phase+'_guard_release.template.json'),release)
 a={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)};b={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)};keys=['identity','same','authorized_worker','event','fail','bounded_error','clear_errors','phase_status','signal_due','final_record','ssh_script','check_limits','require_phase_proof']
 for k in keys:assert a[k]==b[k],k
 write(out/'CPU_orchestration.json',dict(passed=True,unchanged_guard_core_functions=keys,inherited_guardian_sha256=sha(old/GEN/'guardian.py'),policy_sha256=sha(out/'policy.json'),science1506byteidentical=True,launch_origin='LOCAL_OLIVE_ONLY',production_deadline=None,GPU_started=False));print(sha(out/'policy.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','guardian']);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--remote',required=True);a=p.parse_args();globals()[a.phase](a.campaign,a.root,a.remote)
