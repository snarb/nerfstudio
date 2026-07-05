# Occupancy Grid / Coarse Traversal Tests

Date: 2026-06-12

Scope: runs focused on occupancy-grid traversal, `adaptive_coarse_step_size`, `alpha_thre`, and related ARM sampling parameters. All listed artifact scores use the corrected detector that crops the right render panel from `ns-eval` side-by-side `GT|render` outputs.

Common settings unless noted:

```text
scene_scale=1.5
scale_factor=1.0
max_res=8192
enable_frequency_grid=True
enable_adaptive_ray_marching=True
disable_feature_reweighting=True
disable_fas=True
```

## Results

| Maps | `adaptive_warmup_steps` | `adaptive_coarse_step_size` | `adaptive_max_step_size` | `alpha_thre` | `max_steps_per_ray` | Checkpoint | PSNR | SSIM | artifact_score | largest | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fake all-8192 | 0 | `0.0125` | `0.0125` | `0.0025` | 1024 | 8192 | 14.4489 | 0.571334 | 220.012 | 429233 | Reject: immediate ARM handoff collapses. |
| fake all-8192 | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 12288 | 21.8520 | 0.656154 | 36.842 | 48589 | First ARM checkpoint still very bad. |
| fake all-8192 | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 16384 | 27.3453 | 0.656542 | 6.649 | 1974 | Best fake-map checkpoint. |
| fake all-8192 | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 20480 | 26.9686 | 0.666478 | 8.577 | 2322 | Regressed; early-stop fake branch at 16384. |
| real | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 12288 | 21.7693 | 0.653534 | 38.237 | 48042 | Similar to fake at first ARM checkpoint. |
| real, level 15 clamp | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 12288 | 21.1524 | 0.654763 | 36.278 | 46508 | Forcing max frequency barely helps; more samples alone not enough. |
| real | 12288 | `0.0125` | `0.0125` | `0.0025` | 1536 | 16384 | 26.8607 | 0.665240 | 8.065 | 1509 | Baseline real delayed-handoff ARM. |
| real | 12288 | `0.0125` | `0.0125` | `0.0025` | 2048 | 16384 | 28.1763 | 0.641391 | 6.855 | 2040 | Higher cap helps, but only partially. |
| real | 12288 | `0.0125` | `0.0125` | `0.0` | 1536 | 16384 | 26.7439 | 0.631437 | 11.677 | 6935 | Reject: disabling alpha threshold worsens artifacts. |
| real | 12288 | `0.00625` | `0.0125` | `0.0025` | 2048 | 16384 | 28.3587 | 0.646969 | 4.955 | 1813 | Strong improvement from smaller coarse traversal. |
| real | 12288 | `0.00625` | `0.0125` | `0.0025` | 2048 | 20480 | 28.4213 | 0.646301 | 4.893 | 2106 | Marginal improvement; largest component increased. |
| real | 12288 | `0.00625` | `0.00625` | `0.0025` | 2048 | 20480 | 28.3625 | 0.646506 | 4.477 | 2234 | Smaller max step helps further. |
| real | 12288 | `0.00625` | `0.00625` | `0.0025` | 2048 | 24576 | 28.4808 | 0.648240 | 3.568 | 1436 | Best trained checkpoint. |
| real | 12288 | `0.00625` | `0.00625` | `0.0025` | 2048 | 28672 | 28.3284 | 0.645417 | 6.504 | 2599 | Regressed; early-stop trained branch at 24576. |
| real, render-only override | 12288 | `0.003125` | `0.003125` | `0.0025` | 4096 | 24576 | 28.6335 | 0.642302 | 3.387 | 2060 | Best render-only score; not a trained setting. |
| fixed-640 control | n/a | n/a | n/a | n/a | n/a | 45564 | 28.5622 | 0.657815 | 2.182 | 2151 | Reference target; ARM still worse. |

## Key Reads

- `adaptive_warmup_steps=12288` means ARM handoff is delayed. Before that point, training render uses fixed sampling, while the occupancy grid can still be updated from the current density field.
- `adaptive_coarse_step_size=0.0125` was too coarse for the thin stand. Lowering it to `0.00625` was the largest single improvement in the occupancy-grid path.
- `alpha_thre=0.0` did not fix missing structure. It made the run heavier and structurally worse, so the current best keeps `alpha_thre=0.0025`.
- Raising `max_steps_per_ray` from `1536` to `2048` helped but did not solve the issue. Cap saturation was a contributing factor, not the main cause.
- Reducing both `adaptive_coarse_step_size` and `adaptive_max_step_size` to `0.00625` gave the best trained result: `artifact_score=3.568` at step `24576`.
- Rendering the best checkpoint even denser with `0.003125` and cap `4096` reduced score to `3.387`, but still did not match fixed-640. This suggests remaining artifacts are partly render-sampling-sensitive, but also partly baked into the learned field/checkpoint.

## Current Best

Best trained occupancy/ARM setting:

```text
adaptive_warmup_steps=12288
adaptive_coarse_step_size=0.00625
adaptive_max_step_size=0.00625
alpha_thre=0.0025
max_steps_per_ray=2048
checkpoint=24576
artifact_score=3.568
```

Best render-only override:

```text
adaptive_coarse_step_size=0.003125
adaptive_max_step_size=0.003125
alpha_thre=0.0025
max_steps_per_ray=4096
checkpoint=24576
artifact_score=3.387
```

Open gap to fixed-640:

```text
fixed-640 artifact_score=2.182
best trained ARM artifact_score=3.568
best dense ARM render artifact_score=3.387
```
