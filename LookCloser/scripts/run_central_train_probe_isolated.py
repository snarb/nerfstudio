"""Resume frozen transfer requests with one renderer-patching scope per process.

The original transfer launcher attempted to install the renderer wrapper twice
in one interpreter. First-time receipts remain valid; this launcher leaves all
frozen requests and completed renders unchanged.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import probe_central_train_pose_transfer as transfer


def child_commands(worker,workers):
    if workers<1 or not 0<=worker<workers:raise ValueError('Invalid worker partition')
    return [[sys.executable,str(Path(__file__).resolve()),'--frame',frame,
             '--worker',str(worker),'--workers',str(workers)] for frame in transfer.FRAMES]


def run(worker,workers,frame=None):
    commands=child_commands(worker,workers);transfer.verify()
    if frame is None:
        for command in commands:subprocess.run(command,check=True)
    else:
        transfer.configure(frame);transfer.probe.render(worker,workers)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',choices=transfer.FRAMES)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();run(a.worker,a.workers,a.frame)
