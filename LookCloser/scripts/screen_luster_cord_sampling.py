#!/usr/bin/env python3
"""Evaluate a frozen volume checkpoint with pinned train-only cord sampling."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


def screen(checkpoint: Path, fit: Path, radius: float, step_size: float,
           occupancy_radius: float, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Preserve prior screens')
    step=int(checkpoint.stem.split('-')[-1])
    request=deepcopy(json.loads((checkpoint.parents[4]/'request.json').read_text()))
    if request.get('baseline_preset') or request.get('representation','volume')!='volume':
        raise ValueError('Cord screen requires a volume checkpoint with compatible parameters')
    request.update(output=str(output),resume=str(checkpoint),end_step=step,eval_steps=[step])
    request.setdefault('model',{}).update(cord_sampling_fit_path=str(fit),cord_sampling_fit_sha256=sha(fit),
        cord_sampling_radius=radius,cord_sampling_step_size=step_size,
        cord_sampling_max_steps_per_ray=4096,cord_occupancy_radius=occupancy_radius)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    cfg=configuration(request)
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    model=trainer.pipeline.model
    model.current_train_step=step
    before=field_fingerprint(model)
    row=evaluate(trainer,request,step)
    after=field_fingerprint(model)
    if before!=after:
        raise ValueError('Forward-only cord screen changed the field')
    row['checkpoint']=str(checkpoint)
    write(Path(row['renders'])/'metrics.json',row)
    write(output/'result.json',dict(scope='Frozen field forward-only cord-sampling screen',
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),fit=str(fit),fit_sha256=sha(fit),
        radius=radius,step_size=step_size,occupancy_radius=occupancy_radius,
        field_before=before,field_after=after,
        metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
        visual_review='pending'))
    print(json.dumps({key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--radius',type=float,default=.005)
    parser.add_argument('--step-size',type=float,default=.001)
    parser.add_argument('--occupancy-radius',type=float,default=0.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    screen(args.checkpoint,args.fit,args.radius,args.step_size,args.occupancy_radius,args.output)
