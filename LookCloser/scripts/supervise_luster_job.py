"""Supervise one explicitly authorized preprocessing or training review stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);args=p.parse_args()
    job=json.loads(args.job.read_text());root=Path(job['log_dir']);root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    env=os.environ.copy()
    env.update(PYTHONPATH=str(repo)+os.pathsep+str(repo/'LookCloser/scripts'),
               CUDA_HOME='/home/brans/repos/nerfstudio/.cuda128-toolchain',
               TORCH_EXTENSIONS_DIR='/home/brans/.cache/torch_extensions_lookcloser',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['CUDA_HOME']+'/bin'+os.pathsep+env['PATH']
    started=time.time()
    with (root/'stdout.log').open('x') as log, (root/'supervision.jsonl').open('a') as journal:
        worker=subprocess.Popen(job['command'],cwd=repo/'LookCloser',env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            code=worker.poll()
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
            progress_path=Path(job['progress'])
            try:progress=json.loads(progress_path.read_text()) if progress_path.exists() else None
            except json.JSONDecodeError:progress={'partial_write':True}
            with (root/'stdout.log').open('rb') as stream:
                stream.seek(max(0,os.fstat(stream.fileno()).st_size-12000));tail=stream.read().decode(errors='replace')
            try:children=[p.pid for p in psutil.Process(worker.pid).children(recursive=True)] if code is None else []
            except psutil.NoSuchProcess:children=[]
            record=dict(time=time.time(),controller_pid=os.getpid(),worker_pid=worker.pid,children=children,
                        seconds=time.time()-started,exit=code,gpu=gpu.stdout.strip(),
                        oom='out of memory' in tail.lower(),progress=progress,free_GiB=psutil.disk_usage(root).free/2**30)
            journal.write(json.dumps(record)+'\n');journal.flush()
            temporary=root/'status.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(root/'status.json')
            if code is not None:
                complete=Path(job['complete']).exists()
                (root/'finished.json').write_text(json.dumps(dict(record,complete=complete),indent=2)+'\n')
                if code or not complete:raise SystemExit(f'Job failed or completion receipt absent: {code}')
                print('visual_review_gate',flush=True);break
            time.sleep(30)


if __name__=='__main__':main()
