# Holes Inspection

Date: 2026-06-12

## Short Answer: What Improved Artifacts

First, the old `artifact_score=0` was a measurement bug, not a real clean render. `ns-eval` saves `eval_img_0000.png` as `GT | render`, and the old detector compared GT against the left GT panel. After fixing the detector to crop the right render panel, the bad fake-map run scored:

```text
artifact_score=220.012
serious=True
largest=429233px
```

After that correction, the real improvements came from three changes.

### 1. Later ARM Handoff

Turning Adaptive Ray Marching on too early was unsafe. At early steps the density field, occupancy grid, and frequency grid are still unstable. If ARM starts sampling from this unstable state, thin structures such as the metal stand can be skipped or under-sampled, producing holes.

The useful starting point was:

```text
adaptive_warmup_steps=12288
```

This means: train first with fixed sampling, then switch to ARM. It reduced the extreme post-handoff failure from hundreds of artifact score down to tens, and after one more epoch down to about `8`.

### 2. Higher Per-Ray Sample Cap

Raising the cap helped, but only partially:

```text
max_steps_per_ray: 1536 -> 2048
artifact_score: 8.065 -> 6.855
```

Reason: some rays were hitting the sample cap. Once a ray hits the cap, the remaining part of the ray is effectively under-sampled. That can create missing pieces in thin or complex geometry. But because the improvement was limited, cap pressure was not the main cause.

### 3. Smaller Coarse Traversal And Smaller Max Step

This was the main improvement.

Best trained recipe so far:

```text
adaptive_coarse_step_size=0.00625
adaptive_max_step_size=0.00625
max_steps_per_ray=2048
alpha_thre=0.0025
adaptive_warmup_steps=12288
grid_resolution=64
Feature Reweighting off
FAS off
```

Why this helped:

- `adaptive_coarse_step_size` controls the first nerfacc occupancy traversal.
- If this step is too large, the coarse traversal can pass over a thin stand before frequency subdivision even gets a chance to refine it.
- Frequency subdivision only works inside intervals that the coarse traversal found. If the first pass misses or coarsely brackets the thin structure, later frequency-aware subdivision cannot fully recover it.
- `adaptive_max_step_size=0.00625` also prevents low-frequency or uncertain regions from using an overly sparse step.

Measured improvement:

| Change | artifact_score | Interpretation |
|---|---:|---|
| Real-map ARM baseline continuation | 8.065 | Delayed ARM helps, but holes remain. |
| `max_steps_per_ray=2048` only | 6.855 | Cap pressure is part of the issue. |
| `coarse=0.00625`, cap 2048 | 4.955 | Coarse occupancy traversal was a bigger issue. |
| `coarse=max_step=0.00625`, cap 2048, step 20480 | 4.477 | Reducing non-high-frequency step also helps. |
| Same recipe, step 24576 | 3.568 | Best trained checkpoint so far. |
| Same recipe, step 28672 | 6.504 | Regressed; early-stop at 24576. |
| Dense render-only override on step 24576: `coarse=max_step=0.003125`, cap 4096 | 3.387 | Some remaining error is render-sampling-sensitive, but not all. |
| Fixed-640 control | 2.182 | Still better than ARM. |

Current conclusion: the main artifact source is not simply “too few samples everywhere”. It is the interaction between ARM handoff, occupancy traversal, and step sizing around thin geometry. The remaining gap to fixed-640 likely comes from residual traversal/sampling instability plus learned field quality, not from one single scalar parameter.

## Fake Frequency Maps vs Real Frequency Maps

Short version: fake all-high frequency maps were useful as a diagnostic, but they did not solve the holes and were not clearly better than real maps.

| Comparison | artifact_score | Read |
|---|---:|---|
| Fake all-high, immediate ARM | 220.012 | Catastrophic; fake maps do not fix early ARM. |
| Fake all-high, delayed handoff, step 12288 | 36.842 | Similar to real maps at same point. |
| Real maps, delayed handoff, step 12288 | 38.237 | Very close to fake. |
| Fake best delayed checkpoint | 6.649 | Better, but still worse than fixed-640. |
| Real tuned best trained checkpoint | 3.568 | Better than fake best after ARM parameter tuning. |

Conclusion: real frequency maps are probably not the primary cause. The dominant issue is ARM/occupancy traversal and handoff stability. The fake map ruled out the simple hypothesis “real map low frequencies are starving the stand”.

## What Was Tried

All experiments below kept:

```text
scene_scale=1.5
scale_factor=1.0
max_res=8192
enable_frequency_grid=True
enable_adaptive_ray_marching=True
disable_feature_reweighting=True
disable_fas=True
```

| Area | Parameters / change | Result | Decision |
|---|---|---:|---|
| Detector | Fixed side-by-side `GT|render` handling | Bad fake run changed from false `0` to `220.012` | Keep fix; old zeros invalid. |
| Fake maps | Uniform all-8192 frequency maps | Did not prevent holes | Diagnostic only. |
| Immediate ARM | ARM from early step with fake maps | `220.012` | Reject. |
| Delayed handoff | `adaptive_warmup_steps=12288` | Fake `6.649`, real `8.065` after continuation | Keep delayed handoff. |
| Real vs fake | Same delayed recipe at first ARM checkpoint | Fake `36.842`, real `38.237` | Frequency-map values not main cause. |
| Force max frequency | `adaptive_min_frequency_level=15`, `adaptive_max_frequency_level=15` | `38.237 -> 36.278`, but cap saturation high | More samples alone not enough. |
| Higher cap | `max_steps_per_ray=2048` | `8.065 -> 6.855` | Useful but partial. |
| Alpha threshold | `alpha_thre=0.0` | `11.677`, worse | Reject. |
| Smaller coarse traversal | `adaptive_coarse_step_size=0.00625` | `6.855 -> 4.955` | Important improvement. |
| Smaller max step | `adaptive_max_step_size=0.00625` | `4.893 -> 4.477`, then `3.568` after more training | Best trained path. |
| More training | Continue best recipe to `24576` | `3.568` | Best trained checkpoint. |
| Too much training | Continue best recipe to `28672` | `6.504` | Early-stop at `24576`. |
| Dense render only | Render step `24576` with `coarse=max_step=0.003125`, cap 4096 | `3.387` | Shows render sampling still matters, but does not close gap. |
| Parallel heavy runs | Three concurrent heavy ARM experiments | OOM at batch 4096 | Heavy variants should run sequentially or with smaller batch. |

## Current Best Config

Best trained checkpoint:

```text
run:
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_temp_report_2/lookcloser/real_maps_cont20480_to24576_solo_grid64_cap2048_coarse000625_maxstep000625_alpha0025_r4096_seed42

checkpoint:
step-000024576.ckpt

render:
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_temp_report_2/lookcloser/real_maps_cont20480_to24576_solo_grid64_cap2048_coarse000625_maxstep000625_alpha0025_r4096_seed42/artifact_renders_step-000024576/eval_img_0000.png

PSNR=28.4808
SSIM=0.648240
artifact_score=3.568
largest=1436
```

Best render-only version of that checkpoint:

```text
render override:
adaptive_coarse_step_size=0.003125
adaptive_max_step_size=0.003125
max_steps_per_ray=4096
eval_num_rays_per_chunk=128

PSNR=28.6335
SSIM=0.642302
artifact_score=3.387
largest=2060
```

The dense render lowers total score but leaves a large component. That means pure render density is not enough; the field/checkpoint itself still contains structural weakness.

## What Is Still Worth Trying

These are ordered by expected value, not by ease.

### 1. Train With A Fixed-Sampling Safety Net During ARM

Hypothesis: ARM sometimes misses thin geometry because occupancy traversal and frequency subdivision are not a perfect replacement for uniform coverage. Fixed-640 works because it always samples along the whole ray densely enough.

Worth trying:

```text
ARM + mandatory fallback uniform samples per ray
```

Examples:

| Variant | Idea | Why it may help |
|---|---|---|
| ARM intervals + 64 fixed background samples | Always add sparse uniform samples along ray | Prevents complete miss of thin structures. |
| ARM intervals + 128 fixed samples | Stronger safety net | Closer to fixed-640 while still cheaper than fully fixed. |
| ARM only inside occupied intervals, plus fixed samples around occupied interval boundaries | Boundary guard | Holes often come from missed or clipped occupancy intervals. |

This is probably the most direct route toward fixed-640 behavior without turning ARM off.

### 2. Progressive ARM Ramp Instead Of Hard Handoff

Current handoff is still abrupt: before warmup, fixed renderer; after warmup, ARM renderer.

Worth trying:

| Stage | `adaptive_coarse_step_size` | `adaptive_max_step_size` | Reason |
|---|---:|---:|---|
| Early ARM | 0.003125 | 0.003125 | Very conservative transition. |
| Mid ARM | 0.00625 | 0.00625 | Current best stable setting. |
| Later ARM | Maybe 0.00625 / 0.0125 | Only if artifact score stays low | Recover speed only after geometry is stable. |

The goal is to avoid a sudden change in sampling distribution exactly when the geometry is still fragile.

### 3. Train The Dense Render Setting, Not Only Render With It

Dense render-only improved `3.568 -> 3.387`. That is small but real.

Worth trying:

```text
adaptive_coarse_step_size=0.003125
adaptive_max_step_size=0.003125
max_steps_per_ray=4096
train_num_rays_per_batch=2048 or 1024
```

Risk: OOM or slower training. Use one run, not parallel. If batch is reduced, compare carefully because batch-size change can alter optimization.

### 4. Occupancy Grid Update Schedule Around Handoff

Current successful path used a conservative `grid_update_interval=4096`. That helped avoid early instability, but after handoff the occupancy grid may adapt too slowly or preserve bad decisions.

Worth trying:

| Variant | Reason |
|---|---|
| Keep long warmup, then update occupancy more frequently after ARM starts | Thin geometry may need faster occupancy correction. |
| Rebuild or refresh occupancy grid at handoff | Avoid carrying stale fixed-phase occupancy into ARM phase. |
| Larger occupancy update batch after handoff | More stable occupancy estimates for thin objects. |

This is a likely remaining factor because dense rendering still leaves large connected components.

### 5. Start ARM From A Stronger Fixed-Sampling Checkpoint

The best control is fixed-640 at step `45564`, with artifact score `2.182`. Current ARM tuning starts much earlier from a `12288` fixed/warmup checkpoint.

Worth trying:

```text
continue from fixed-640-like clean checkpoint
enable ARM with conservative settings
coarse=max_step=0.00625 or 0.003125
cap=2048/4096
```

Reason: if the field already learned the stand cleanly, ARM may preserve it better than trying to learn it during/after handoff.

### 6. Re-test Fake vs Real Under The Best ARM Recipe

We compared fake vs real mostly before discovering the best `coarse/max step=0.00625` recipe.

Worth trying:

```text
fake all-high maps + best ARM recipe
real maps + best ARM recipe
same checkpoints / same schedule
```

Expected outcome: probably similar. But if fake maps now beat real maps under the best traversal recipe, then frequency-grid spatial values still matter in later training.

### 7. Localized Artifact-Weighted Validation

The run at `28672` improved some global metrics but artifact score regressed badly. Global PSNR/SSIM/LPIPS are not enough.

Worth adding:

```text
artifact_score on eval_img_0000
largest component
selected crop scores for the metal stand
```

Use these as checkpoint selection criteria, not just eval loss. This is already proven necessary because step `24576` is structurally better than `28672`.

## Practical Next Experiment Order

Recommended next sequence:

| Priority | Experiment | Stop rule |
|---:|---|---|
| 1 | Continue from best trained `24576` with ARM + fixed fallback samples, 64 or 128 per ray | Stop if artifact_score does not beat `3.568`. |
| 2 | Train dense ARM: `coarse=max_step=0.003125`, cap 4096, smaller batch | Stop on OOM or if artifact_score does not beat dense render-only `3.387`. |
| 3 | Start ARM from fixed-640 control checkpoint with conservative settings | Stop if it fails to preserve score near `2.182`. |
| 4 | Occupancy refresh/update schedule after handoff | Stop if no improvement over `3.568`. |
| 5 | Fake-vs-real under best recipe | Use only as diagnosis, not main optimization. |

The most important target is not just reducing average score. The largest connected component must also shrink. Current best trained score `3.568` still has `largest=1436`, and dense render-only has `largest=2060`; that means the remaining failure is still structurally meaningful.
