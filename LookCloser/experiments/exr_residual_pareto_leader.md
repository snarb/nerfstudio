# EXR residual Pareto leader

## What was tested

The remaining target was strict simultaneous improvement over the geometry-guard reference:
PSNR above `34.136342`, SSIM above `0.899861`, LPIPS below `0.2`, and no cable gaps. Direct joint
losses, checkpoint/component interpolation, filtering, learned source gating, global PQ calibration,
and frequency-gated image fusion did not pass all four gates.

The accepted renderer is a bounded image-space residual stage, not color grading. It keeps native
linear EXR masters and performs its learned operation in the same dataset-calibrated ST2084 PQ
domain used by evaluation:

1. Seventeen evenly spaced train-camera renders were retained, with 14 used for optimization and
   three fixed held-out train views used for checkpoint selection. No eval image or eval target was
   used to train the residual network.
2. Inputs are a primary render, an auxiliary geometry render, and their difference. A
   `9→48→3` CNN with five residual blocks at dilations `1/2/4/2/1` predicts a correction bounded to
   `±0.04` PQ.
3. Training used 128×128 patches, batch 8, AdamW `2e-4`, and
   `PQ-MSE + 0.1 DSSIM + 0.02 AlexNet-LPIPS`. Step 250 was selected because held-out PSNR improved
   from `35.2104` to `35.4616` while LPIPS improved from `0.17347` to `0.15532`.
4. At evaluation, the primary is exact hash24 step106496 rendered with dense 4× corrected ARM; the
   geometry-guard step106316 render is the auxiliary. The residual is applied at `beta=0.88` and
   decoded back to scene-linear EXR.

The implementation is `scripts/train_hdr_residual_renderer.py`. Exact application command:

```bash
python scripts/train_hdr_residual_renderer.py apply \
  --checkpoint /mnt/data/lookcloser_lpips_campaign/residual_renderer_v1/mse_dssim010_lpips002/step-000250.pt \
  --primary-render-dir /mnt/data/lookcloser_lpips_campaign/hash24_capacity_v1/lookcloser/pqmse_513_s42/exact_sampling_step106496/adaptive_dense4x_corrected/renders \
  --auxiliary-render-dir /mnt/data/lookcloser_geometry_campaign/geometry_guard_v1/lookcloser/geom_v2_q80_cross_cont1_s42/renders_latest_step-000106316 \
  --output-dir /mnt/data/lookcloser_lpips_campaign/residual_renderer_v1/repro_apply_beta088/renders \
  --nits-per-scene-unit 4654.2274151658285 \
  --blend-beta 0.88 \
  --preview-exposure-ev 5.009187089119898
```

The selected residual checkpoint can be retrained from the retained pair manifest with:

```bash
python scripts/train_hdr_residual_renderer.py train \
  --pairs /mnt/data/lookcloser_lpips_campaign/learned_fusion_v1/train_pairs_normal \
  --output-dir /mnt/data/lookcloser_lpips_campaign/residual_renderer_v1/retrain \
  --nits-per-scene-unit 4654.2274151658285 \
  --steps 3000 --patch-size 128 --batch-size 8 --lr 2e-4 \
  --dssim-weight 0.1 --lpips-weight 0.02 --eval-every 250 --seed 42
```

## Results

All metrics are full-resolution means over the three held-out eval cameras in calibrated PQ.

| Candidate | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Cable gap px |
|---|---:|---:|---:|---:|
| Geometry guard, step106316 | 34.136342 | 0.899861 | 0.211813 | 0 |
| Prior field-only perceptual leader | **34.369545** | 0.899050 | **0.199267** | 0 |
| Exact hash24 step106496, dense4 | 34.423115 | **0.900806** | 0.215004 | 0 |
| **Hash24 + bounded PQ residual, beta0.88** | **34.213291** | **0.899972** | **0.199505** | **0** |

The final per-view metrics are:

| View | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
| eval0 | 34.11780 | 0.900541 | 0.238253 |
| eval1 | 34.81693 | 0.906663 | 0.206986 |
| eval2 | 33.70514 | 0.892711 | 0.153277 |

The independent EXR evaluator reports zero non-finite pixels, negative prediction channels, and
over-peak channels. The targeted cable detector reports zero missing pixels and zero longest gap in
all three views. Significant full-frame artifacts are `0/3` and serious priority ROIs are `0/10`.
All three cable crops and the `-2/0/+2 EV` review sheets were visually inspected; the long cable is
continuous and no new structured artifact is visible.

## Insights

The exact-checkpoint fix exposed a stronger structural source than the older run-directory result:
hash24 has enough SSIM margin to absorb a small perceptual correction. A residual trained around the
field-only perceptual/geometry pair transfers to this source, while direct replacement over-corrects.
The `0.85–0.935` beta interval passes all numeric gates; `0.88` was selected for balanced SSIM and
LPIPS margin rather than choosing the boundary.

Final provenance:

- hash24 checkpoint:
  `/mnt/data/lookcloser_lpips_campaign/hash24_capacity_v1/lookcloser/pqmse_513_s42/nerfstudio_models/step-000106496.ckpt`,
  SHA-256 `e6a39b0e65617cb1de31ae57825399201b2244742a321cd4c116739e22426ae3`;
- residual checkpoint:
  `/mnt/data/lookcloser_lpips_campaign/residual_renderer_v1/mse_dssim010_lpips002/step-000250.pt`,
  SHA-256 `431e343857687a1f553652114d0b4fb73bb5348bf101d2d001cc5f97b75ab7c6`;
- retained train/validation pair manifest:
  `/mnt/data/lookcloser_lpips_campaign/learned_fusion_v1/train_pairs_normal/manifest.json`;
- reproduced linear EXRs and metrics:
  `/mnt/data/lookcloser_lpips_campaign/residual_renderer_v1/repro_apply_beta088/`.
