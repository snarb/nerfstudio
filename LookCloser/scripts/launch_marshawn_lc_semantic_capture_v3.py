"""Root-gated local standard-controller launch; dry-run unless --execute."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from prepare_marshawn_lc_semantic_capture import sha


def main(a):
    root=a.root.resolve();plan=root/'manifest.json';m=json.loads(plan.read_text())
    controller=root/'code/marshawn_tuning_controller.py'
    controller_plan=root/'controller_plan.json'
    command=[m['python'],str(controller),'launch','--root',str(root/'controller'),'--manifest',str(controller_plan)]
    if a.status:
        result=subprocess.run([m['python'],str(controller),'status','--root',str(root/'controller')],capture_output=True,text=True,check=True)
        d=json.loads(result.stdout)
        out=dict(time=time.time(),controller_alive=d['controller_alive'],jobs={k:dict(status=v['status'],host=v.get('host'),gpu=v.get('gpu'))for k,v in d.get('state',{}).get('jobs',{}).items()})
        for p in [root/'capture/resource_checks.jsonl',root/'queue/jobs/LC_best126316_native_capture/checks.jsonl']:
            if p.exists():
                with p.open('rb')as f:f.seek(0,2);f.seek(max(0,f.tell()-131072));lines=f.read().splitlines()
                if lines:out[str(p)]=json.loads(lines[-1])
        with(root/'manual_supervision.jsonl').open('a')as f:f.write(json.dumps(out)+'\n')
        print(json.dumps(out));return
    if not a.execute:
        print(json.dumps(dict(authorization=False,GPU_started=False,controller_started=False,
            command=command,manifest_sha256=sha(plan),controller_plan_sha256=sha(controller_plan))))
        return
    if Path(__file__).resolve() != (root/'code'/Path(__file__).name).resolve():
        raise ValueError('Run only reviewed packaged launcher')
    worker=root/'code/run_marshawn_lc_semantic_capture_v3.py'
    spec=importlib.util.spec_from_file_location('_local_LC_capture_authorize',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.authorize(plan,root/'root_release.json',Path(m['campaign']))
    release=json.loads((root/'root_release.json').read_text())
    if release.get('controller_plan_sha256') != sha(controller_plan) or sha(controller_plan) != m['controller_plan_sha256']:
        raise ValueError('Controller plan lacks exact root release')
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    print(result.stdout.strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--status',action='store_true')
    main(p.parse_args())
