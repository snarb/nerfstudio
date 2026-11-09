"""Launch exactly one independently root-released local LC semantic stage."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_marshawn_frozen_semantic_cpu import sha


def run(root,stage,execute):
    plan=root/'controllers'/(stage+'.json')
    receipt=json.loads((root/'CPU_readiness.json').read_text())
    python=receipt['python'];controller=root/'code/marshawn_tuning_controller.py'
    command=[python,str(controller),'launch','--root',str(root/'controllers'/stage),'--manifest',str(plan)]
    if not execute:
        print(json.dumps(dict(stage=stage,controller_started=False,CUDA_started=False,authorization=False,
            command=command,controller_plan_sha256=sha(plan))))
        return
    release=json.loads((root/'root_release.json').read_text())
    flag={'technical':'LC_controller_technical_authorized','uncapped_qualification':'LC_uncapped_qualification_authorized',
          'first2064_gate':'LC_controller_first2064_authorized'}[stage]
    if release.get('reviewer')!='root' or release.get(flag)is not True or release['controller_plan_sha256'].get(stage)!=sha(plan):
        raise ValueError('Exact bounded controller stage release required')
    for p,digest in release['bound_files'].items():
        if sha(p)!=digest:raise ValueError('Sealed controller input changed: '+p)
    if stage in ['technical','first2064_gate']:
        fit_release=json.loads(Path(receipt['fit_release']).read_text())
        key='LC_technical_head_authorized'if stage=='technical'else'LC_first2064_head_authorized'
        if fit_release.get('reviewer')!='root'or fit_release.get(key)is not True:
            raise ValueError('Independent fit namespace also requires exact root head release')
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    print(result.stdout.strip())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['technical','uncapped_qualification','first2064_gate'],required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();run(args.root,args.stage,args.execute)
