"""Dispatch from local olive to silver; require exact ROOT-released bindings.

The dispatcher runs under local Conda and the local v4 controller. Its host
manifest launches silver workers over SSH. Running this launcher ON silver
would create unsupported self-SSH and is deliberately rejected.
"""
import argparse,json,shlex,subprocess
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True,help='LOCAL /fsx experiment root on olive')
p.add_argument('--remote-root',type=Path,help='SC experiment root on silver; defaults to CPU_readiness.root')
p.add_argument('--stage',choices=['technical','first2064_gate'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
local_campaign=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
if a.root.parent!=local_campaign:
    raise ValueError('Launcher must dispatch locally from the /fsx olive campaign root, never from SC on silver')
readiness=json.loads((a.root/'CPU_readiness.json').read_text())
remote_root=str(a.remote_root or readiness['root'])
if remote_root!=readiness['root']:
    raise ValueError('Remote root differs from sealed silver worker namespace')
plan=a.root/'controllers'/(a.stage+'.json');manifest=json.loads(plan.read_text())
if any(h.get('address')!='ubuntu@silver-sable'for h in manifest['hosts'].values()):
    raise ValueError('Only local olive dispatch to reviewed silver hosts supported')
local_python='/home/pavlo/.conda/envs/nerfstudio/bin/python'
controller=local_campaign/'frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py'
if not Path(local_python).exists()or not controller.exists():raise ValueError('Qualified local dispatcher runtime/controller missing')
command=[local_python,str(controller),'launch','--root',str(a.root/'controllers'/a.stage),'--manifest',str(plan)]
if a.execute:
    release_path=a.root/'root_release.json';release=json.loads(release_path.read_text())
    phase='technical'if a.stage=='technical'else'first2064'
    if (release.get('reviewer')!='root'or release.get('LC_refinement_controller_'+phase+'_authorized')is not True or
        release.get('LC_refinement_'+phase+'_head_authorized')is not True or release['controller_plan_sha256'][a.stage]!=sha(plan)):
        raise ValueError('Exact ROOT refinement stage release required')
    # Binding paths name SC worker inputs, so verify them ON silver using a
    # CPU-only stdlib command. Also require byte-identical local/remote release.
    checks=dict(release['bound_files']);checks[remote_root+'/root_release.json']=sha(release_path)
    code='import sys,json,hashlib; from pathlib import Path; checks=json.load(sys.stdin); bad=[p for p,d in checks.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=d]; assert not bad,bad; print(json.dumps({"remote_CPU_bound_files_verified":len(checks)}))'
    remote_command=shlex.join([readiness['python'],'-c',code])
    verified=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ubuntu@silver-sable',remote_command],
        input=json.dumps(checks),text=True,capture_output=True,check=True,timeout=60)
    print(verified.stdout.strip(),flush=True)
    subprocess.run(command,check=True)
else:
    print(json.dumps(dict(command=command,dispatcher_host='local_olive',remote_worker_host='silver',remote_root=remote_root,
        CUDA_started=False,authorization=False,controller_started=False,self_SSH=False)))
