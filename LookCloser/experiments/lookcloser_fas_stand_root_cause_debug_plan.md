# LookCloser FAS Stand Artifact Debug Goal

## Goal

Find why enabling FAS can make parts of the vertical metal stand disappear in `eval_img_0000.png`, then fix the smallest responsible cause while keeping Feature Re-weighting disabled.

The no-FAS LookCloser baseline is the stable reference where the stand is visible. Some FAS runs improve global PSNR/SSIM but fail visually because the stand becomes broken, hollow, or partly absent. Treat visual correctness of this crop as the first gate before spending time on long runs or metric sweeps.

## Context

Dataset:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns
```

Baseline params:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/recomended_params.md
```

Stable no-FAS renders:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_tuning/lookcloser/adaptive_fg_arm_iso_h40_maxfreq12_coarse00125_continue36864_r4096/renders_full_step-000034816
```

Known visually rejected FAS renders:

```text
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/007740_hd_aabb4_multicamera_eval3_ns_fg_arm_iso_fas_tuning/lookcloser/fas_mix035_w2048_r4096_seed43/renders_best_step-000034816
```

Target artifact:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/problem.png
```

Strict visual gate:

- render: `eval_img_0000.png`
- crop: `left_stand_connector_eval0`
- crop box: `xyxy=(320, 0, 617, 530)`
- script: `scripts/render_lookcloser_crop_gate.py --crop-name left_stand_connector_eval0`

Reject a run early if the vertical metal stand has missing sections, holes, hollow-looking parts, disconnected connector geometry, or floating pieces. If the stand already fails after the first meaningful FAS-active checkpoint, stop that run and try the next hypothesis.

More detailed prior notes:

```text
/home/ubuntu/repos/nerfstudio/LookCloser/experiments/lookcloser_fas_stand_artifact_debug.md
```

## High-Level Plan

Work in a separate branch such as:

```bash
git switch -c debug/fas-stand-root-cause
```

Use the recommended no-FAS params as the base. Remove `--disable-fas`, keep `--disable-feature-reweighting`, and debug FAS only.

1. **Reproduce and timestamp the failure.** Find the earliest checkpoint where the stand disappears or breaks under the current FAS settings. Always inspect the target crop before checking global metrics.

2. **Check whether frequency maps are the cause.** Try a uniform/debug frequency map or a sampler mode that ignores map variation while keeping FAS mechanics enabled. If uniform FAS fixes the stand, focus on preprocessing/map thresholds/noise. If uniform FAS still breaks it, focus on sampler mechanics, coverage, indexing, or training imbalance.

3. **Measure actual sampling coverage.** Add or use a debug heatmap showing which train-image pixels FAS really samples. Look for stand-related regions that are rarely/never sampled, shifted coordinates, over-sampling of brick/wall texture, or bucket imbalance.

4. **Simplify FAS until the stand is safe.** Gradually reduce FAS strength, flatten level probabilities, increase uniform-sampling floor, or simplify patch/bucket selection. The goal is to identify the smallest FAS behavior that reintroduces the artifact.

5. **Fix the minimal cause, then validate.** Only after the stand passes the early visual gate, run longer training and 3 seeds. Final acceptance requires the stand crop to pass visually and PSNR/SSIM to improve over the no-FAS baseline with Feature Re-weighting still disabled.

Save each experiment's command, crop path, visual verdict, and final render path. Metrics are useful only after the target crop passes.
