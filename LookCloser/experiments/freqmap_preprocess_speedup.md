# LookCloser 2D frequency-map preprocessing speedup

## What was tested

Goal: make per-image 2D frequency-map generation much faster WITHOUT changing the
resulting maps beyond the project tolerance vs the staged 66-map reference
(`/home/brans/freqmap_bench/static_reference_maps`, mean scalar_res 3346, median 2775,
mean max-level frac 0.1188). All work done on `clever-shadow` (RTX PRO 6000 Blackwell,
96 GB). Reference config: patch=8, ssim_thr=0.95, win=7, steps_per_level=1000,
train_batch=8192, lr=1e-2, n_levels=16, min_res=16, max_res=8192, log2_hashmap=23.

Two hypotheses:
1. Hyperparameter/convergence fix: the per-image training (16 levels x 1000 steps x
   batch 8192) is over-doing tiny-batch passes over a fixed 2M-pixel image; a much
   larger batch (and/or higher LR) should reach the SAME converged fit in far fewer
   steps, matching the maps in a fraction of the wall-clock.
2. Throughput fix: if the step budget is irreducible, run many independent images
   concurrently on the idle 96 GB GPU (identical per-image math -> identical maps).

## Bottleneck diagnosis

Per image the pipeline trains a fresh 2D tinycudann HashGrid for 16 levels x 1000
steps = 16,000 gradient steps, then SSIM-assigns each 8x8 patch to the first level
that reaches SSIM >= 0.95. Profiling / observation:

- Model + optimizer already PERSIST across levels (not re-initialized) - that part is fine.
- Training dominates wall-clock. GPU is ~98% util during training but uses only ~2.5 GB.
  The model is a tiny FullyFusedMLP; each step is compute-bound on many small kernels.
- The eval/assign pass originally rebuilt patch index tensors with Python list
  comprehensions per 64-patch batch and extracted GT patches in a Python loop. I
  vectorized this (unfold GT patches once, GPU index tensors, large eval batch) + TF32.
  Result: 76.7 -> 70.4 s/img. Marginal, because eval is not the bottleneck. This
  confirms micro-optimizations (AMP/TF32/eval vectorization) are NOT the win.

Baseline (unmodified pipeline, 4-img subset): 5m7s / 4 = ~76.7 s/img on Blackwell.
(Reference doc quoted ~175 s/img on the slower L40S; Blackwell already ~2.3x faster.)

## Results: batch / steps / LR grid (4-image subset)

Aggregate stats vs the 66-map reference (mean_ratio = out mean / ref mean; ref median
2775). Pearson r on only 4 images is noise (baseline-4 exact recompute gave r=0.31),
so judge these by mean/median ratio + mean-maxlevel-frac diff (tol 0.05).

| config                  | bs     | steps | lr    | s/img | mlf diff | mean_ratio | median_ratio | verdict |
|-------------------------|--------|-------|-------|-------|----------|------------|--------------|---------|
| REFERENCE (baseline)    | 8192   | 1000  | 0.01  | 76.7  | 0.019    | 0.958      | 0.957        | correct |
| fast (vectorized eval)  | 8192   | 1000  | 0.01  | 70.4  | 0.022    | 0.955      | 0.848        | correct |
| bs8k_s700               | 8192   | 700   | 0.01  | 49.3  | 0.025    | 1.125      | 1.285        | drift high |
| bs8k_s500               | 8192   | 500   | 0.01  | 35.3  | 0.120    | 1.353      | 1.450        | FAIL |
| bs64k_s250              | 65536  | 250   | 0.01  | 18.5  | 0.279    | 1.444      | 1.450        | FAIL |
| bs64k_s250_lr2          | 65536  | 250   | 0.02  | 18.5  | 0.610    | 1.949      | 2.952        | FAIL |
| bs64k_s250_lr4          | 65536  | 250   | 0.04  | 18.5  | 0.718    | 2.119      | 2.952        | FAIL |
| bs32k_s300_lr2          | 32768  | 300   | 0.02  | 21.6  | 0.440    | 1.723      | 2.701        | FAIL |
| bs16k_s500_lr15         | 16384  | 500   | 0.015 | 35.5  | 0.194    | 1.366      | 1.285        | FAIL |

## Insights

- **Reducing the step budget systematically over-assigns to higher frequency levels.**
  The maps drift monotonically HIGH as steps drop (median_ratio 0.96 -> 1.29 -> 1.45),
  because under-converged reconstructions never reach SSIM 0.95 at the correct (low)
  level, so patches spill upward to finer levels / max level.
- **Larger batch does NOT let you cut steps at fixed LR** (bs64k/250 is worse than
  bs8k/500), and **raising LR makes it dramatically worse** (lr 0.02/0.04 push the
  median straight to the 8192 ceiling). Hypothesis #1 is refuted: the 16,000-step
  budget is a genuine convergence requirement, not redundant tiny-batch churn. The
  reference lr=1e-2 x 1000 steps/level sits at a convergence sweet spot for these maps.
- Therefore the ONLY correctness-preserving speedup is **throughput**: process many
  independent images concurrently on the idle 96 GB GPU. Each worker runs the exact
  reference computation, so maps match within stochastic tolerance. (See parallel
  section below.)

## Results: parallel throughput (12-image subset, exact reference config)

Round-robin sharding of images across N concurrent worker processes on ONE GPU,
each worker running the exact reference computation (bs=8192, steps=1000, lr=1e-2).

| workers | wall/img | notes |
|---------|----------|-------|
| 1       | 70.5 s   | maps match ref: mlf_diff 0.007, mean_ratio 1.024, median_ratio 1.066 |
| 2       | >2x/img (net negative) | GPU already 98% at N=1; 2 workers just contend, no image completed in the time 1 worker did 2 |

**The GPU is compute-saturated by a single image.** tinycudann's FullyFusedMLP keeps
the SMs ~98% busy even at batch 8192 (only 2.5 GB used). Adding concurrent workers does
NOT increase throughput - they time-slice the same saturated compute and per-image time
grows >=2x, giving zero (slightly negative) net gain. So the 96 GB is irrelevant here:
the bottleneck is compute, not memory or GPU idle time. Data-parallelism is not a win.

## Bottom line

Both proposed levers are exhausted against the hard tolerance:
- **Fewer steps / bigger batch / higher LR**: all break the maps (monotonic over-assignment).
  The 16,000-step budget is a genuine convergence requirement.
- **Parallel images**: no gain; single image already saturates the GPU.

Best achievable WITHOUT changing maps: the vectorized-eval + TF32 fast path at the
reference config = **70.5 s/img** on the Blackwell GPU. This is already ~2.5x faster
than the quoted ~175 s/img L40S baseline (mostly the newer GPU, plus eval vectorization
that removed Python-loop overhead). It is NOT the requested order-of-magnitude, because
the per-image training cannot be shortened without violating the correctness tolerance
and the GPU cannot be further filled.

- sec/image: 70.5
- sec/frame (66 img): ~4653 s (~1.29 h)
- 45 frames: ~58 h (~2.4 days), down from the ~6-day L40S estimate.

If an order-of-magnitude is required, it can only come from RELAXING the correctness
constraint (accept a different map definition) or from MORE GPUs (58 h / N_gpus, since
per-image work is embarrassingly parallel ACROSS machines - each GPU saturated by one
image). Within one GPU + exact-maps, 70.5 s/img is at the floor.

## New / modified scripts (clever-shadow, mirror to orchestrator repo)

- `/home/brans/freqmap_bench/fast_freqmap.py` -> `LookCloser/scripts/fast_freqmap.py`
  (additive fast per-image path: vectorized eval + TF32 + `--file-list`; same map math)
- `/home/brans/freqmap_bench/validate_freqmaps.py` (one-off tolerance check vs the reference
  maps; not mirrored to this repo, since it was only needed to validate this task's output)
- `/home/brans/freqmap_bench/run_parallel.sh` (shards images across N concurrent workers)
- Reference pipeline `nerfstudio/scripts/lookcloser_preprocess.py` was NOT modified;
  work was on a `freqmap-speed` git branch in the brans repo.

## Recommended command (exact maps, fastest safe config)

```bash
# on clever-shadow, as brans
cd /home/brans/repos/nerfstudio
.venv/bin/python /home/brans/freqmap_bench/fast_freqmap.py \
  --images-dir <frame>/images --glob 'frame_train_*.jpg' \
  --out <frame>/lookcloser_frequencies \
  --steps-per-level 1000 --train-batch-size 8192 --eval-patch-batch 16384 --max-res 8192
```

Validation tolerance used to accept this config (checked with a one-off script, not kept
in this repo): shape/dtype/range, mean-maxlevel-frac diff <= 0.05, per-image mean-level pearson > 0.90
vs the 66-map reference). Note: pearson is only meaningful on the full 66-image frame,
which has real per-image variance; on small low-variance subsets pearson is dominated by
training noise even for an exact recompute (baseline-4 exact recompute gave r=0.31).

Full 66-image validation of this config on the staged 007740 frame was launched at
`/home/brans/freqmap_bench/007740_frame/fast_full66` (~77 min) to confirm the pearson
tolerance on the full set; aggregate stats on the 12-image subset already pass
(mlf_diff 0.007, mean_ratio 1.024).
