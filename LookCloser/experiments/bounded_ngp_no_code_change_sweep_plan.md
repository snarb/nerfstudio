# Bounded Instant-NGP No-Code-Change Sweep Plan

## Goal

Improve bounded Instant-NGP eval quality from the current wrapper defaults without changing Instant-NGP model code and without tuning optimizer, MLP, or grid/hash-grid settings.

Primary selection uses combined PSNR+SSIM rank. LPIPS and eval loss are diagnostic and reported for every candidate.

## Guardrails

- Use `LookCloser/scripts/run_bounded_ngp_quiet.py` only.
- Do not modify `nerfstudio/models/instant_ngp.py` for this sweep.
- Do not add or tune new Instant-NGP losses, appearance embedding behavior, proposal sampling, camera optimizer, optimizer/scheduler settings, MLP architecture, or hash-grid settings.
- Only use existing wrapper/model/dataparser parameters:
  `scene_scale`, `center_method`, `orientation_method`, `scale_factor`, `near_plane`, `far_plane`, `render_step_size_mult`, `alpha_thre`, `cone_angle`, `background_color`, `use_gradient_scaling`, and `train_num_rays_per_batch`.
- Do not test white background. Background candidates are `black` and `random` only.
- Keep early stop enabled. The selected checkpoint is the wrapper's best eval-loss checkpoint.

## Anchor

Current wrapper defaults:

- `scene_scale=1.5`
- `center_method=focus`
- `orientation_method=up`
- `auto_scale_poses=True`
- `train_num_rays_per_batch=8192`
- `background_color=black`
- `near_plane=0.01`
- `far_plane=1000.0`
- `alpha_thre=0.0`
- `cone_angle=0.0`
- `render_step_size=null`
- `loss_type=mse`
- `use_gradient_scaling=False`
- no scene contraction

Documented baseline final eval, latest checkpoint protocol:

| Checkpoint protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Full run, latest checkpoint `step-000060751.ckpt` | 24.417955 | 0.639772 | 0.460250 |

Clean early-stop reproduction at `scene_scale=1.5` showed enough variance that every candidate below must be repeated three times.

## Repeat Protocol

Run three experiments for each candidate value:

- Use fixed seeds `42`, `43`, and `44`.
- Use deterministic timestamp suffixes: `<stage>_<param>_<value>_seed42`, `seed43`, `seed44`.
- Keep all non-tested parameters at the current best setting.
- Use `--max-num-iterations 100000` as a ceiling only.
- Keep `--eval-checkpoint best`, early stopping enabled, and final render/eval enabled.

For each candidate value, report:

- each run's selected checkpoint, eval loss, PSNR, SSIM, LPIPS;
- mean and max for PSNR and SSIM;
- mean and min for LPIPS;
- mean, min, and max for eval loss;
- combined score based on mean PSNR rank plus mean SSIM rank, tie-broken by max PSNR, then mean LPIPS.

## Sweep Order

Start with parameters most likely to dominate quality. This is a staged greedy sweep: after each stage, carry forward only the best candidate value.

### Stage 0: Control

Run the anchor three times:

```bash
python LookCloser/scripts/run_bounded_ngp_quiet.py --timestamp control_scene150_seed42 --seed 42 --scene-scale 1.5 --max-num-iterations 100000
python LookCloser/scripts/run_bounded_ngp_quiet.py --timestamp control_scene150_seed43 --seed 43 --scene-scale 1.5 --max-num-iterations 100000
python LookCloser/scripts/run_bounded_ngp_quiet.py --timestamp control_scene150_seed44 --seed 44 --scene-scale 1.5 --max-num-iterations 100000
```

Use this to establish the current variance band before judging improvements.

### Stage 1: Scene Scale

Scene scale should move in coarse steps because run variance can hide small changes.

| Param | Values |
|---|---|
| `scene_scale` | `1.0`, `1.25`, `1.5`, `2.0`, `2.5` |

Keep `center_method=focus`, `orientation_method=up`, and all model knobs at the anchor. Carry forward the best scene scale by mean PSNR+SSIM rank.

### Stage 2: Sampling Density

These directly change ray marching resolution and are likely to affect detail and floaters.

Test one parameter at a time from the best Stage 1 config:

| Param | Values |
|---|---|
| `render_step_size_mult` | `0.5`, `0.75`, `1.0`, `1.25` |
| `near_plane` | `0.005`, `0.01`, `0.02` |
| `alpha_thre` | `0.0`, `0.0025`, `0.005` |
| `cone_angle` | `0.0`, `0.001`, `0.00390625` |

Carry forward the best value after each parameter. If `render_step_size_mult=0.5` is too slow or OOM-prone, keep its partial results but do not continue that value.

### Stage 3: Pose Normalization

Only run this if Stage 1 and Stage 2 do not produce a clear winner over the control band.

| Param | Values |
|---|---|
| `center_method` | `focus`, `poses` |
| `orientation_method` | `up`, `none` |
| `scale_factor` | unset, `0.85`, `1.15` |

Test one parameter at a time. Do not combine `orientation_method=none` with changed `center_method` unless it wins alone.

### Stage 4: Existing Lower-Priority Knobs

These are lower priority because they are less likely to dominate bounded-scene geometry than scene bounds and sampling.

| Param | Values |
|---|---|
| `background_color` | `black`, `random` |
| `use_gradient_scaling` | `False`, `True` |
| `train_num_rays_per_batch` | `8192`, `12288`, `16384` |

For `train_num_rays_per_batch`, skip values that OOM. Do not reduce batch size below `8192` unless debugging.

## Final Report

Save results in `LookCloser/experiments/bounded_ngp_param_sweep.md`.

For every tested parameter value include:

- parameter value and fixed inherited config;
- three run timestamps;
- per-run selected checkpoint, eval loss, PSNR, SSIM, LPIPS;
- mean and max PSNR;
- mean and max SSIM;
- mean and min LPIPS;
- mean, min, and max eval loss;
- final rank and whether the value is carried forward.

The final recommendation must report:

- best params;
- best single-run metrics;
- mean metrics over the three repeats for the winning config;
- comparison against the `scene_scale=1.5` control mean and max;
- render directory for the best single run.
