"""Retain every checkpoint that can still win PSNR/LPIPS selection, plus latest."""
import json
import math
from pathlib import Path
import time
from archive_luster_checkpoint import sha


def retained_checkpoints(history, latest):
    if not history:raise ValueError('Cannot prune without measured checkpoint history')
    if not all(math.isfinite(row[k]) for row in history for k in ['eval_all_psnr','eval_all_lpips']):raise ValueError('Nonfinite selection metrics')
    keep={str(latest)}
    for i,row in enumerate(history):
        dominated=False
        for j,other in enumerate(history):
            better_psnr=other['eval_all_psnr']>=row['eval_all_psnr']
            better_lpips=other['eval_all_lpips']<=row['eval_all_lpips']
            strict=(other['eval_all_psnr']>row['eval_all_psnr'] or other['eval_all_lpips']<row['eval_all_lpips'] or j<i)
            if better_psnr and better_lpips and strict:
                dominated=True;break
        if not dominated:keep.add(row['checkpoint'])
    return keep


def prune_dominated(root, frame_root, history, latest):
    root=Path(root).resolve();frame_root=Path(frame_root).resolve();frame_root.relative_to(root)
    keep=retained_checkpoints(history,latest);seen=set()
    for row in history:
        checkpoint=Path(row['checkpoint']).resolve()
        if str(checkpoint) in keep or checkpoint in seen or not checkpoint.exists():continue
        seen.add(checkpoint)
        relative=checkpoint.relative_to(frame_root/'runs')
        if checkpoint.suffix!='.ckpt' or 'nerfstudio_models' not in relative.parts:raise ValueError('Not a campaign checkpoint')
        record=dict(time=time.time(),path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,
                    reason='Another measured checkpoint has at least as high PSNR and at most as high LPIPS; this checkpoint cannot win the current or a future tie window',
                    retained=sorted(keep),archived=checkpoint.with_suffix('.archive.json').exists())
        with (root/'checkpoint_pruning.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        checkpoint.unlink()
    return keep
