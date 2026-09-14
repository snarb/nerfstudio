"""Supervised wide-path renders with checksum-bound reuse of real-train masks."""
import argparse
from pathlib import Path
import run_dynamic_grid_workers as base
from wide_dynamic_camera_flight import install_source_masks,OUTPUT


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['supervise','worker']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--worker-index',type=int);p.add_argument('--frames',nargs='+')
    a=p.parse_args();base.renderer.torch.set_num_threads(4)
    # Reuse lifecycle/lock/30-second GPU and process checks, but make child
    # commands enter THIS adapter so source-mask eligibility cannot be skipped.
    base.__file__=__file__
    if a.action=='worker':
        install_source_masks(base.renderer);base.worker(a.output,a.worker_index,a.frames)
    else:base.supervise(a.output,a.workers)
