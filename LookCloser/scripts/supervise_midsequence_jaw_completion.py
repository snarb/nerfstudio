"""Bounded fresh-process RGB pool; preserve per-job logs and live checks."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from transfer_close_boundary_midsequence import ROOT
from render_midsequence_jaw_completion import FRAMES, VIEWS


def run():
    jobs = [(f,v) for f in FRAMES for v in VIEWS]; live = []; failed = []
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', OPENCV_IO_ENABLE_OPENEXR='1')
    while jobs or live:
        while jobs and len(live)<6 and not failed:
            frame, view = jobs.pop(0); path = ROOT/'logs'/f'rgb_{frame}_{view}.log'
            stream = path.open('x')
            command = [sys.executable, str(Path(__file__).with_name('render_midsequence_jaw_completion.py')),
                       'render','--frame',frame,'--view',view]
            worker = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            live.append((frame,view,worker,stream,path))
        record = dict(utc=datetime.now(timezone.utc).isoformat(), controller_pid=os.getpid(),
            pending=len(jobs), free_bytes=shutil.disk_usage(ROOT).free,
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                         '--format=csv,noheader'], text=True), workers=[])
        active = []
        for frame,view,worker,stream,path in live:
            status=worker.poll()
            record['workers'].append(dict(frame=frame,view=view,pid=worker.pid,exit_code=status,
                log_tail=path.read_text(errors='replace')[-400:]))
            if status is None:active.append((frame,view,worker,stream,path))
            else:
                stream.close(); print(frame,view,'exit',status,flush=True)
                if status:failed.append((frame,view,status))
        with (ROOT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        live=active
        if failed and not live:raise RuntimeError(f'Failed workers {failed}; pending jobs not launched')
        if live:time.sleep(10)
    print('All nine frame/view workers completed',flush=True)


if __name__=='__main__':run()
