"""Launch one reviewed stage; never choose the next stage automatically."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('variant',choices=['control','exp_sh']);p.add_argument('end_step',type=int)
    p.add_argument('--parent',type=Path);p.add_argument('--fr',type=float,choices=[1.,.3])
    p.add_argument('--reason',required=True);p.add_argument('--label');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();repo=Path(__file__).resolve().parents[2];root=args.root.resolve()
    if args.parent:
        request=deepcopy(json.loads((args.parent/'request.json').read_text()))
        complete=json.loads((args.parent/'complete.json').read_text())
        if complete['step']>=args.end_step:raise ValueError('Continuation must advance the last completed step')
        expected='softplus' if args.variant=='control' else 'trunc_exp'
        if request['model']['density_activation']!=expected:raise ValueError('Variant differs from parent')
        request['resume']=complete['latest_checkpoint'];request['parent_history']=str(args.parent/'history.json')
    else:
        request=json.loads((repo/'LookCloser/recipes/luster_000470'/f'{args.variant}_02000.json').read_text())
    if args.fr is not None:request['model']['feature_reweighting_strength']=args.fr
    label=args.label or f'{args.variant}_{args.end_step:05d}'
    if Path(label).name!=label:raise ValueError('Label must be one directory name')
    output=root/'runs'/label
    if output.exists() or (root/'requests'/f'{label}.json').exists() and args.parent:
        raise ValueError('Stage already exists; choose an unused label')
    request.update(data=str(root/'data'),output=str(output),end_step=args.end_step,eval_steps=[args.end_step])
    path=root/'requests'/f'{label}.json';job_path=root/'jobs'/f'{label}.json'
    job=dict(log_dir=str(root/'logs'/label),command=[sys.executable,str(repo/'LookCloser/scripts/run_luster_experiment.py'),str(path)],
             progress=str(output/'progress.json'),complete=str(output/'complete.json'))
    if args.dry_run:
        print(json.dumps(dict(request=request,job=job),indent=2));return
    for folder in ['requests','jobs','logs']:(root/folder).mkdir(exist_ok=True,parents=True)
    if path.exists() and json.loads(path.read_text())!=request:raise ValueError('Existing request differs')
    path.write_text(json.dumps(request,indent=2)+'\n');job_path.write_text(json.dumps(job,indent=2)+'\n')
    with (root/'logs'/f'{label}_controller.log').open('x') as log:
        controller=subprocess.Popen([sys.executable,str(repo/'LookCloser/scripts/supervise_luster_job.py'),str(job_path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(time=time.time(),stage=label,reason=args.reason,controller_pid=controller.pid,parent=str(args.parent) if args.parent else None)
    with (root/'logs/campaign_decisions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
