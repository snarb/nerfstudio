# LookCloser Architecture

LookCloser extends Instant-NGP with four frequency-aware modules: Frequency Grid, Adaptive Ray
Marching (ARM), Feature Reweighting (FR), and Frequency-Averaged Sampling (FAS). All are enabled
in the current leader recipe.

---

## Current Leader

**Budget-aware ARM + Feature Reweighting + FAS** (`ns-train lookcloser` defaults as of Jun 23 2026).

| Metric | Leader (Budget-ARM + FR + FAS) | Old fixed-640 baseline |
|--------|--------------------------------|------------------------|
| PSNR   | **29.618**                     | 29.565                 |
| SSIM   | **0.6685**                     | 0.683                  |
| LPIPS  | **0.2311**                     | 0.365                  |
| ROI artifacts | **0**                   | 0                      |

Notes:
- Metrics above are the re-eval at the confirmed surviving checkpoint (step 106316, `fromscratch_s42_A_fw03`).
- Original nofas_long run peak at step 91128 (pruned): PSNR 29.917 / LPIPS 0.280. Surviving step 106316: 29.858 / 0.272.
- FAS parallel run at step 106316: PSNR 29.877 / LPIPS 0.258. FAS is now enabled by default.
- Best observed LPIPS with FAS at step 121504: 0.249 (PSNR drifts; use `eval_all_psnr` selection).

**Archived leader checkpoint** (network disk, accessible from any server):
```
/fsx/oregon/tank_bkup/6A_4_EXR/artifacts/static_lookcloser_leader_007740/
  step-000106316.ckpt            ← checkpoint weights
  config.yml
  dataparser_transforms.json
  eval JSON
  LEADER_INFO.md
```

**Source run:**
```
/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/fromscratch_repro/007740_fromscratch_s42_A_fw03/lookcloser/20260624_002610/
```

**Previous ARM-only leader** (no FR, artifact-free reference):
- Step 40576: PSNR 29.535 / SSIM 0.693 / LPIPS 0.396 / significant artifact 0.000 / micro artifact 0.256
- Run: `007740_hd_aabb4_multicamera_eval3_ns_arm_h40_grid128_huber_delta02_occ0001_dense` (local repro_runs)

---

## Architecture Components

### 1. Frequency Grid
A 3D voxel grid (resolution 128) storing a scalar frequency level per voxel. Updated periodically
during training by projecting pixels through train cameras, computing 2D SSIM-based frequency from
preprocessed maps, and writing the level into intersected voxels.

Drives all three other components. Must be preprocessed before training:
```bash
python scripts/run_lookcloser_preprocess_quiet.py
```
Preprocessing settings: `patch_size=8`, `ssim_window_size=7`, `high_frequency_level=13`,
`train_steps_per_level=1000`, `ssim_threshold=0.95`. Maps saved to `lookcloser_frequencies/`.

Key param: `max_res` of frequency maps **must** match the model `max_res` (both 8192 for HD).

### 2. Adaptive Ray Marching (ARM) — budget-aware
Subdivides each occupancy-grid traversal interval based on local frequency level: high-frequency
voxels get finer steps, low-frequency voxels get coarser steps (paper interval `dt = 1/(2*N_l)`
normalized to ray `t` units).

**Budget-aware fix (Jun 22 2026, commit `daee59bf`):** The original implementation used front-to-back
rank ordering to clip to `max_steps_per_ray`, which exhausted the budget on early high-frequency
intervals, leaving a gap at the far end of the ray (metal stand holes, cable holes). The fix scales
`dt` per-ray proportionally so total samples ≤ `max_steps_per_ray` while preserving relative density
ratios. This eliminated all ROI artifacts.

Implemented in `nerfstudio/model_components/lookcloser_samplers.py`.

### 3. Feature Reweighting (FR)
Applies paper Eq. 6 weights to hash-grid features in `LookCloserField`: suppresses high-frequency
hash levels in low-frequency regions, giving cleaner gradients in smooth areas. Re-enabling FR after
the ARM bug fix gave −26% LPIPS improvement over the ARM-only baseline.

`feature_reweighting_strength=1.0` uses full paper weights; 0.0 = disabled; intermediate values blend.
Implemented in `nerfstudio/fields/lookcloser_field.py`.

### 4. Frequency-Averaged Sampling (FAS)
During training, oversamples image patches from high-frequency regions (up to ~3× more) to focus
training on detail. Enabled by default. PSNR difference vs no-FAS is within noise; FAS gives
consistently better LPIPS. Late PSNR drift means checkpoint selection must use `eval_all_psnr`.

Implemented in `nerfstudio/lookcloser_pixel_sampler.py`.

---

## Leader Recipe — Recommended Parameters

These are the current `ns-train lookcloser` defaults (from `nerfstudio/configs/method_configs.py`):

| Group | Parameter | Value |
|-------|-----------|-------|
| **Training** | `max_num_iterations` | 200000 |
| | `train_num_rays_per_batch` | 4096 |
| | `steps_per_save` | 2000 |
| **ARM** | `ray_sampling_mode` | adaptive |
| | `enable_adaptive_ray_marching` | True |
| | `max_steps_per_ray` | 1024 |
| | `adaptive_coarse_step_size` | 0.00625 |
| **FR** | `enable_feature_reweighting` | True |
| | `feature_reweighting_strength` | 1.0 |
| **FAS** | `enable_fas` | True |
| | `fas_strength` | 1.0 |
| | `fas_level_count_alpha` | 0.0 |
| | `patch_size` / `stride` | 8 / 8 |
| **Frequency Grid** | `grid_resolution` | 128 |
| | `max_res` | 8192 |
| | `num_frequency_levels` | 16 |
| | `grid_update_interval` | 1024 |
| **Loss** | `reconstruction_loss_type` | charbonnier |
| | `distortion_loss_mult` | 0.01 |
| **Rendering** | `background_color` | black |
| **Occupancy** | `occupancy_warmup_steps` | 4096 |
| | `occupancy_binary_warmup_steps` | 4096 |
| | `alpha_thre` | 0.0 |
| | `transmittance_threshold` | 0.0 |
| | `near_plane` | 0.01 |
| | `cone_angle` | 0.0 |

---

## Training & Evaluation

### Step 0 — Preprocess frequency maps (once per dataset)
```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python scripts/run_lookcloser_preprocess_quiet.py
```

### Step 1 — Train with leader defaults
```bash
python scripts/run_lookcloser_quiet.py
```
The quiet runner: redirects noisy output to log files, monitors `metrics_compact.csv`, stops on
eval-loss plateau, runs `ns-eval` on best checkpoint, writes `run_summary.json`.

For a dry run (print command without training):
```bash
python scripts/run_lookcloser_quiet.py --dry-run
```

### Step 2 — Checkpoint selection
Use `eval_all_psnr` (highest PSNR across all 3 eval views; LPIPS as tie-breaker within 0.07 dB).
The quiet runner does this automatically. For artifact-sensitive runs use `--eval-checkpoint artifact`
with `--keep-all-checkpoints`.

### Step 3 — Evaluate metrics
```bash
python scripts/summarize_lookcloser_runs.py <experiment_dir>
```
Reports: PSNR / SSIM / LPIPS / artifact score / ROI artifact / stand-connector score / train time.

### Artifact detection
```bash
python scripts/detect_structural_artifacts.py --preset significant ...
```
Scores below 0.000 = no qualifying structural artifact. Include alongside PSNR/SSIM/LPIPS.
ROI audit: `scripts/score_artifact_rois.py` (curated crop list). Fixed s640 baseline: ROI=0/3 seeds,
stand-connector=0.000. Budget-ARM leader: ROI=0, stand-connector=0.000.

---

## Key Source Files

| File | Role |
|------|------|
| `nerfstudio/models/lookcloser.py` | Model, ARM integration, loss |
| `nerfstudio/model_components/lookcloser_samplers.py` | Budget-aware ARM sampler |
| `nerfstudio/fields/lookcloser_field.py` | Feature Reweighting, hash field |
| `nerfstudio/model_components/lookcloser_grid.py` | 3D Frequency Grid |
| `nerfstudio/lookcloser_pixel_sampler.py` | FAS pixel sampler |
| `nerfstudio/pipelines/lookcloser_pipeline.py` | Training pipeline, grid updates |
| `nerfstudio/configs/method_configs.py` | `lookcloser` method defaults |
| `nerfstudio/scripts/lookcloser_preprocess.py` | 2D frequency map preprocessing |

---

## Known Limitations

- **Micro artifact floor (0.256):** Small diagnostic artifacts remain in off-ROI regions (thin wires
  in eval1). Debug shows `grid_miss_likely=false`, `field_issue_likely=true` — not a binary
  occupancy miss. Significant (official) artifact score is 0.000.
- **FAS late PSNR drift:** FAS PSNR drifts down slowly past step ~100k. Use `eval_all_psnr`
  checkpoint selection (already the default).
- **Scale matching:** Current runs use `scene_scale=1.5`. Setting `max_res` in frequency maps must
  match model `max_res` (both 8192). Using `scene_scale=1.5` with default model `max_res=8192` is
  validated; the model no longer auto-derives `max_res` from scene scale.
- **Dense-to-ARM checkpoint transfer:** Loading a fixed-step checkpoint into ARM mode causes
  catastrophic PSNR collapse. Always start ARM training from an ARM-native source checkpoint.
