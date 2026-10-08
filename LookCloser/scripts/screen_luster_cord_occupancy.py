#!/usr/bin/env python3
"""Forward-only cord occupancy admission screen on a pinned trained field."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


def screen(selection_path: Path, radius: float, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Preserve prior occupancy screens')
    selected=json.loads(selection_path.read_text())
    selected=selected.get('selected',selected)
    checkpoint=Path(selected['checkpoint'])
    source=checkpoint.parents[4]
    request=deepcopy(json.loads((source/'request.json').read_text()))
    request.update(output=str(output),resume=str(checkpoint),end_step=selected['step'],
                   eval_steps=[selected['step']])
    if request.get('baseline_preset') or not request['model'].get('cord_sampling_fit_path'):
        raise ValueError('Cord occupancy screen requires a fitted experimental volume model')
    request['model']['cord_occupancy_radius']=radius
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    cfg=configuration(request)
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    model=trainer.pipeline.model
    model.current_train_step=selected['step']
    admission=model.cord_occupancy_admission
    before_occupied=int((model.occupancy_grid.binaries & admission).sum())
    before=field_fingerprint(model)
    row=evaluate(trainer,request,selected['step'])
    after=field_fingerprint(model)
    if after != before:
        raise ValueError('Forward-only screen changed field parameters')
    row['checkpoint']=str(checkpoint)
    write(Path(row['renders'])/'metrics.json',row)
    write(output/'result.json',dict(
        scope='Forward-only occupancy screen; no optimizer steps',checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),fit_sha256=request['model']['cord_sampling_fit_sha256'],
        radius=radius,admission_cells=int(admission.sum()),occupied_before=before_occupied,
        newly_admitted=int(admission.sum())-before_occupied,field_before=before,field_after=after,
        metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
        visual_review='pending'))
    print(json.dumps({key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--radius',type=float,default=.003)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    screen(args.selection,args.radius,args.output)
