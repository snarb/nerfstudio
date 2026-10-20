"""Prepare subsequent frozen frame geometry while earlier transfer runs on CPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(root,frames,chunk):
    scripts=Path(__file__).resolve().parent
    for frame in frames:
        directory=root/'frames'/f'{frame:06d}'
        assert (directory/'context.json').exists()
        env={**os.environ,'LUSTER_HYBRID_FRAME_MANIFEST':str(directory/'context.json'),
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'PYTHONDONTWRITEBYTECODE':'1',
             'CUDA_HOME':str(Path(sys.executable).parents[1])}
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
        def command(stage,args,gpu=False):
            if shutil.disk_usage(root).free<768*1024**3:
                raise RuntimeError('Preserve training disk reserve')
            print(json.dumps(dict(frame=frame,stage=stage,time=time.time())),flush=True)
            with (root/'logs'/f'{frame:06d}_prefetch_{stage}.log').open('a') as log:
                subprocess.run([sys.executable,str(scripts/args[0]),*args[1:]],
                    env={**env,'CUDA_VISIBLE_DEVICES':'0' if gpu else ''},
                    cwd=scripts.parents[1],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (directory/'preflight/manifest.json').exists():
            command('prepare',['luster_hybrid_common.py','--root',str(directory)])
        command('targets',['guarded_transfer_inference.py','--root',str(directory),
            '--cameras','7','12','95','150','--occupancy-only','--chunk',str(chunk)],True)
        if not (directory/'preflight/sources.json').exists():
            command('sources',['transfer_luster_hybrid.py','--root',str(directory),
                               'sources','--count','16'])
        sources=json.loads((directory/'preflight/sources.json').read_text())['all_sources']
        command('geometry',['guarded_transfer_inference.py','--root',str(directory),
            '--cameras',*map(str,sources),'--occupancy-only','--chunk',str(chunk)],True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True)
    p.add_argument('--chunk',type=int,choices=[256,512,1024,2048,4096],default=4096)
    a=p.parse_args();run(a.root,a.frames,a.chunk)
