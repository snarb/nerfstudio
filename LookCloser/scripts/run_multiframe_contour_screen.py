"""CPU-only matched contour controls after each v1 frame finishes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(root,frames,controller):
    scripts=Path(__file__).resolve().parent
    for frame in frames:
        directory=root/'frames'/f'{frame:06d}'
        while not (directory/'v1_complete.json').exists():
            process=Path(f'/proc/{controller}/cmdline')
            try:
                command=process.read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:
                raise RuntimeError('Own v1 controller exited before requested frame completed')
            if 'run_multiframe_transfer.py' not in command or str(root) not in command:
                raise RuntimeError('Own controller identity changed')
            time.sleep(10)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
             'LUSTER_HYBRID_FRAME_MANIFEST':str(directory/'context.json'),
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'PYTHONDONTWRITEBYTECODE':'1'}
        for sigma,displacement in [(2,1),(2,2),(4,1),(4,2)]:
            if shutil.disk_usage(root).free<768*1024**3:
                raise RuntimeError('Preserve training disk reserve')
            name=f'contour_s{sigma}_d{displacement}'
            print(json.dumps(dict(frame=frame,candidate=name,time=time.time())),flush=True)
            jobs=[
                ['contour_guard_luster_hybrid.py','--root',str(directory),
                 '--parent','v1','--name',name,'--sigma',str(sigma),
                 '--displacement',str(displacement)],
                ['evaluate_luster_hybrid.py','--root',str(directory),
                 '--candidate',name,'--device','cpu'],
                ['summarize_multiframe_transfer.py','--root',str(root),'--candidate',name]]
            for args in jobs:
                with (root/'logs'/f'{frame:06d}_{name}_{Path(args[0]).stem}.log').open('a') as log:
                    subprocess.run([sys.executable,str(scripts/args[0]),*args[1:]],
                                   env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                                   cwd=scripts.parents[1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True)
    p.add_argument('--controller',type=int,required=True)
    a=p.parse_args();run(a.root,a.frames,a.controller)
