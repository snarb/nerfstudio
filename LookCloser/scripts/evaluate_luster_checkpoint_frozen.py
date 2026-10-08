#!/usr/bin/env python3
"""Evaluate a frozen LookCloser checkpoint under a pinned RGB/ROI protocol.

Unlike the continuation runner, this never takes a training step or writes a
new checkpoint. It is useful for comparing fields trained at different source
resolutions against the same held-out pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_luster_opacity import configure_runtime, configuration, evaluate, sha, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True,
                        help='Training-compatible JSON with data, ROIs, and model controls')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier frozen evaluation')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    request = json.loads(args.request.read_text())
    request.update(output=str(args.output), resume=str(args.checkpoint),
                   end_step=int(args.checkpoint.stem.split('-')[-1])+1)
    configure_runtime(request)
    import torch
    torch.manual_seed(request.get('seed',42))
    args.output.mkdir(parents=True)
    write(args.output/'request.json', request)
    write(args.output/'frozen_evaluation_receipt.json',dict(
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        request_source=str(args.request),request_source_sha256=sha(args.request),
        transforms_sha256=sha(Path(request['data'])/'transforms.json'),
        frequency_ready_sha256=sha(Path(request['data'])/'frequency_ready.json'),
        training_steps=0))
    config = configuration(request)
    config.save_config()
    trainer = config.setup(local_rank=0,world_size=1)
    trainer.setup()
    step=int(args.checkpoint.stem.split('-')[-1])
    trainer.step=step
    trainer.pipeline.model.current_train_step=step
    metrics=evaluate(trainer,request,step)
    write(args.output/'result.json',dict(status='complete',metrics=str(args.output/f'eval_{step:06d}'/'metrics.json'),
                                         checkpoint=str(args.checkpoint),
                                         eval_all_psnr=metrics['eval_all_psnr'],
                                         eval_all_ssim=metrics['eval_all_ssim'],
                                         eval_all_lpips=metrics['eval_all_lpips']))
    print(json.dumps({key:metrics[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}),flush=True)


if __name__=='__main__':
    main()
