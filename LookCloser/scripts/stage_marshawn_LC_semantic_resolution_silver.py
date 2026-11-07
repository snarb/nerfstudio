"""Byte-exact CPU staging of the false-release256 package; never launch a job."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

from prepare_marshawn_frozen_semantic_cpu import sha, write
from prepare_marshawn_LC_semantic_resolution_cpu import C, SC
from stage_marshawn_lc_head_silver import INVENTORY, UPLOAD

HOST = 'ubuntu@silver-sable'


def ssh(code, args=(), stdin=None):
    command = shlex.join(['python3', '-c', code, *map(str, args)])
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, command],
                          stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


REMOTE_CPU_PROOF = r'''
import contextlib,hashlib,importlib,importlib.util,io,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert not(root/'root_release.json').exists()and not(root/'runs').exists()and not(root/'queue').exists()
policy=json.loads((root/'root_review_TEMPLATE.json').read_text())
permissions={k:v for k,v in policy.items()if k.endswith('_authorized')or k=='production_after_visualgate'}
assert permissions and all(v is False for v in permissions.values())
for p,digest in policy['bound_files'].items():assert sha(p)==digest,p
for key in ['spec_sha256','controller_plan_sha256']:
 for p,digest in policy[key].items():assert sha(p)==digest,p
sys.path.insert(0,str(root/'code'))
imports={}
for p in sorted((root/'code').glob('*.py')):
 m=importlib.import_module(p.stem);imports[p.stem]=dict(path=m.__file__,sha256=sha(m.__file__))
import torch,numpy as np
assert not torch.cuda.is_initialized()
from marshawn_LC_semantic_resolution_contract import validate_qualification,checkpoint_metadata
from probe_marshawn_LC_semantic_resolution_head import run
dry=[];leaves={};histories=[]
for p in sorted((root/'specs').glob('*.json')):
 s=json.loads(p.read_text());assert s['grid_resolution']==256 and s['qualified_UNKNOWN_count']==4
 assert torch.__version__==s['torch_version']and torch.version.cuda==s['cuda_version']
 labels=json.loads(Path(s['label_qualification']).read_text());validate_qualification(labels)
 assert sha(s['label_qualification'])==policy['label_qualification_sha256']
 assert labels['capture_receipt_sha256']==sha(s['capture_receipt'])
 for r in labels['unknown_queries']:
  assert sha(r['qualification'])==r['qualification_sha256'];leaves[r['qualification']]=r['qualification_sha256']
 for path_key,hash_key in [('root_original_qualification','root_original_qualification_sha256'),('root_additional_LA_qualification','root_additional_LA_qualification_sha256')]:
  if path_key in labels:assert sha(labels[path_key])==labels[hash_key];leaves[labels[path_key]]=labels[hash_key]
 manifest=json.loads(Path(s['manifest']).read_text());assert manifest['native_world_to_box']==s['world_to_box']
 assert sha(manifest['native_source']['checkpoint'])==manifest['native_source']['checkpoint_sha256']
 assert sha(manifest['LC_capture_receipt'])==manifest['LC_capture_receipt_sha256']
 capture=json.loads(Path(s['capture_receipt']).read_text());assert capture['manifest_sha256']==sha(s['manifest'])
 pool=[r['camera']for r in capture['records']if r['rays']];rng=np.random.default_rng(42)
 actual=[pool[int(rng.integers(len(pool)))]for _ in range(2064)]
 assert actual==json.loads(Path(s['packet_history_reference']).read_text())['camera_history']
 histories.append(dict(spec=p.name,packet_reference_sha256=sha(s['packet_history_reference']),exact2064=True))
 text=io.StringIO()
 with contextlib.redirect_stdout(text):run(p,False)
 value=json.loads(text.getvalue());assert not value['CUDA_started']and not value['optimizer_created'];dry.append(value)
 checkpoint_metadata(s)
controllers=[]
for p in sorted((root/'controllers').glob('*.json')):
 plan=json.loads(p.read_text())
 for name,host in plan['hosts'].items():
  controller=Path(host['controller']);assert sha(controller)=='2e7a1be53794882ffc3c0a6a82fff3cc24070b36fbf7593e65771772e0dadf39'
  spec=importlib.util.spec_from_file_location('cpu_controller_import_only',controller);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  controllers.append(dict(plan=p.name,sha256=sha(p),controller_sha256=sha(controller),python=str(Path(host['python']).resolve()),GPU_indices=host['gpus'],jobs=[j['name']for j in plan['jobs']]))
assert not torch.cuda.is_initialized()
assert not(root/'root_release.json').exists()and not(root/'runs').exists()and not(root/'queue').exists()
assert not list(root.rglob('__pycache__'))
print(json.dumps(dict(passed=True,CPU_only=True,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=False,torch_version=torch.__version__,cuda_library_version=torch.version.cuda,python_executable=sys.executable,python_realpath=str(Path(sys.executable).resolve()),imports=imports,qualified_runtime_leaves=leaves,packet_history=histories,dry_runs=dry,controller_imports=controllers,permissions=permissions,head_nativefit_render_export_started=False,controller_or_worker_started=False,root_actual_release_created=False,new_runs_or_queue_or_outputs=False,all_bound_source_inputs_verified=len(policy['bound_files']))))
'''


def stage(root, output, campaign=C):
    if output.exists(): raise FileExistsError('Use a new staging receipt namespace')
    remote = SC/root.relative_to(campaign)
    policy = json.loads((root/'root_review_TEMPLATE.json').read_text())
    rows = {}
    for target, digest in policy['bound_files'].items():
        source = campaign/Path(target).relative_to(SC)
        if not source.is_file() and target == str(SC/'frozen_LC_semantic_head_silver_alias_v1/runs/prior001_fresh2064/head.pt'):
            source = campaign/'frozen_LC_semantic_head_archives/recovery1_20261107_v1/runs/prior001_fresh2064/head.pt'
        if sha(source) != digest: raise ValueError('Local input differs: '+str(source))
        rows[target] = dict(local=str(source), remote=target, sha256=digest, bytes=source.stat().st_size, new_package=False)
    for source in root.rglob('*'):
        if source.is_file():
            if '__pycache__' in source.parts: raise ValueError('Never stage generated bytecode')
            target = str(remote/source.relative_to(root))
            rows[target] = dict(local=str(source), remote=target, sha256=sha(source), bytes=source.stat().st_size, new_package=True)
    output.mkdir(parents=True)
    inventory = output/'source_inventory.json'; write(inventory, list(rows.values()))
    with inventory.open('rb') as f: before = json.loads(ssh(INVENTORY, stdin=f).stdout)
    write(output/'remote_before.json', before)
    if any(r['state']=='CONFLICT' for r in before): raise ValueError('Existing remote bytes conflict; never replace them')
    missing = [r for r in before if r['state']=='missing']
    if any(not r['new_package'] for r in missing): raise ValueError('Inherited source/runtime leaf is missing remotely; no source recopy authorized')

    def upload(row):
        with Path(row['local']).open('rb') as f: return json.loads(ssh(UPLOAD, [row['remote'], row['sha256']], stdin=f).stdout)

    with ThreadPoolExecutor(max_workers=4) as pool: uploaded = list(pool.map(upload, missing))
    write(output/'uploaded.json', uploaded)
    with inventory.open('rb') as f: after = json.loads(ssh(INVENTORY, stdin=f).stdout)
    if any(r['state']!='verified' for r in after): raise ValueError('Incomplete remote byte verification')
    write(output/'remote_after.json', after)
    # This invokes only CPU imports and run(...,False), with empty CUDA visibility.
    command = shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', str(SC/'python'),
                          '-c', REMOTE_CPU_PROOF, str(remote)])
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, command],
                            capture_output=True, text=True, check=True)
    proof = json.loads(result.stdout); write(output/'remote_CPU_proof.json', proof)
    receipt = dict(passed=True,CPU_only=True,GPU_or_controller_started=False,host=HOST,remote_root=str(remote),
        files_verified=len(after),source_files_reused=len(after)-len(missing),new_files_uploaded=len(missing),
        bytes_uploaded=sum(r['bytes'] for r in missing),root_template_sha256=sha(root/'root_review_TEMPLATE.json'),
        worker_sha256=sha(root/'code/probe_marshawn_LC_semantic_resolution_head.py'),
        helper_sha256=sha(root/'code/marshawn_LC_semantic_resolution_contract.py'),
        technical_DAG_sha256=sha(root/'controllers/technical.json'),
        first2064_DAG_sha256=sha(root/'controllers/first2064_after_root_technical_gate.json'),
        specs_sha256=policy['spec_sha256'],runtime_leaves_verified=proof['qualified_runtime_leaves'],
        remote_CPU_proof_sha256=sha(output/'remote_CPU_proof.json'),all_permissions_false=True,
        active_FOUR_UNKNOWN128_job_preempted_or_modified=False,new_outputs_created=False)
    write(output/'stage_receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=C/'frozen_LC_semantic_resolution256_fourUNK_plan_v1')
    parser.add_argument('--output', type=Path, default=C/'frozen_LC_semantic_resolution256_fourUNK_stage_v1')
    args=parser.parse_args(); stage(args.root,args.output)
