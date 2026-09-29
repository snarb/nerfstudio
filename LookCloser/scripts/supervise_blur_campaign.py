"""Sequential, logged process/GPU supervision; a review gate ends each job list."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def changed_conditions(left, right):
    def flatten(value, prefix=''):
        if isinstance(value,dict):
            return {k:v for name,item in value.items() if name not in {'output','comparison_parent'}
                    for k,v in flatten(item,prefix+name+'.').items()}
        return {prefix.rstrip('.'):value}
    a,b=flatten(left),flatten(right)
    return sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))


def consumed_seconds(root):
    total=0.
    for folder in root.iterdir():
        if not folder.is_dir():continue
        for names in [('complete.json','progress.json'),('frequency_complete.json','frequency_progress.json')]:
            for name in names:
                path=folder/name
                if path.exists():
                    total+=json.loads(path.read_text()).get('seconds',0.)
                    break
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);args=parser.parse_args()
    jobs=json.loads(args.manifest.read_text());root=args.manifest.parent
    repo=Path(__file__).resolve().parents[2]
    env=os.environ.copy();env['PYTHONPATH']=str(repo)+os.pathsep+str(repo/'LookCloser/scripts')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+'/home/brans/repos/nerfstudio/.cuda128-toolchain/bin'+os.pathsep+env['PATH']
    env['CUDA_HOME']='/home/brans/repos/nerfstudio/.cuda128-toolchain'
    env['TORCH_EXTENSIONS_DIR']='/home/brans/.cache/torch_extensions_lookcloser'
    env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2'
    started=time.time()
    with (root/(args.manifest.stem+'_supervision.jsonl')).open('a') as journal:
        for request_path in jobs:
            request=json.loads(Path(request_path).read_text());out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
            if request.get('comparison_parent'):
                parent=Path(request_path).with_name(request['comparison_parent']+'.json')
                differences=changed_conditions(json.loads(parent.read_text()),request)
                if len(differences)!=1:raise ValueError(f'Expected one changed condition: {differences}')
            if (out/'complete.json').exists():continue
            if consumed_seconds(root)>=22*3600:raise SystemExit('Training budget exhausted; evaluation reserve retained')
            with (out/'stdout.log').open('w') as log:
                p=subprocess.Popen([sys.executable,str(repo/'LookCloser/scripts/run_blur_experiment.py'),str(request_path)],
                                   cwd=repo/'LookCloser',env=env,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    code=p.poll()
                    process=psutil.Process(p.pid) if code is None else None
                    progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
                    tail=(out/'stdout.log').read_text(errors='replace')[-4000:]
                    record=dict(time=time.time(),controller_pid=os.getpid(),worker_pid=p.pid,exit=code,
                        children=[] if process is None else [c.pid for c in process.children(recursive=True)],
                        gpu=gpu.stdout.strip(),oom=('out of memory' in tail.lower()),progress=progress,run=out.name)
                    journal.write(json.dumps(record)+'\n');journal.flush()
                    print(json.dumps(dict(run=out.name,exit=code,progress=progress)),flush=True)
                    if code is not None:
                        if code or not (out/'complete.json').exists():raise SystemExit(f'Run failed: {out}')
                        break
                    if time.time()-started>22*3600 or consumed_seconds(root)>=22*3600:
                        p.terminate();p.wait();raise SystemExit('Training budget limit; evaluation reserve retained')
                    time.sleep(30)
    print('visual_review_gate',flush=True)


if __name__=='__main__':main()
