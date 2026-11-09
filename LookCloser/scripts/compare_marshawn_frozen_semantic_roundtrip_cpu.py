"""Measure CUDA continuation differences without assuming deterministic atomics."""
import argparse
import json
from pathlib import Path
from prepare_marshawn_frozen_semantic_cpu import sha, write


def compare(first32, resumed64, full64, output):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    a, b, c = [torch.load(p/'head.pt', map_location='cpu', weights_only=False) for p in [first32, resumed64, full64]]
    midpoint = torch.load(full64/'midpoint32.pt', map_location='cpu', weights_only=False)
    starts = json.loads((resumed64/'initial_state.json').read_text())
    if not all(starts['restoration'].values()):
        raise ValueError('Resume-start restoration failed')
    if a['iteration'] != 32 or b['iteration'] != 64 or c['iteration'] != 64:
        raise ValueError('Wrong actual iteration count')
    if len({x['model_digest'] for x in [a, b, c]}) != 1 or len({x['label_qualification_sha256'] for x in [a, b, c]}) != 1:
        raise ValueError('Native source or qualified labels differ across technical arms')
    if a['camera_history'] != c['camera_history'][:32] or b['camera_history'] != c['camera_history']:
        raise ValueError('Camera sampler trajectories differ')
    def differences(x, y):
        dx = (x['head']['logits']-y['head']['logits']).numpy().astype(np.float64)
        dp = (x['head']['logits'].sigmoid()-y['head']['logits'].sigmoid()).numpy().astype(np.float64)
        return dict(logit_max_abs=float(np.abs(dx).max()), logit_RMSE=float(np.sqrt(np.mean(dx*dx))),
            logit_abs_quantiles=np.quantile(np.abs(dx), [.5, .9, .99, 1]).tolist(),
            probability_max_abs=float(np.abs(dp).max()), probability_RMSE=float(np.sqrt(np.mean(dp*dp))),
            fraction_of_logit_values_bit_identical=float((dx == 0).mean()))
    completions = [json.loads((p/'complete.json').read_text()) for p in [first32, resumed64, full64]]
    report = dict(resume_start_restoration=starts['restoration'], camera_history_exact=True,
        independently_executed_first32=differences(a, midpoint), resumed64_versus_uninterrupted64=differences(b, c),
        native_checkpoint_and_caches_unchanged=all(r['native_checkpoint_and_caches_unchanged'] for r in completions),
        native_optimizers_created=False, source_RGB_targets_used=False, CPU_only=True,
        CUDA_initialized=torch.cuda.is_initialized(), checkpoint_sha256={str(p/'head.pt'): sha(p/'head.pt') for p in [first32, resumed64, full64]},
        interpretation='Exact resume-start head/Adam/RNG and sampler history are required. Subsequent CUDA grid_sample backward atomic trajectories may differ; measured differences are reported, never advertised as bitwise equivalence.')
    write(output/'comparison.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['first32', 'resumed64', 'full64', 'output']: p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); compare(a.first32, a.resumed64, a.full64, a.output)
