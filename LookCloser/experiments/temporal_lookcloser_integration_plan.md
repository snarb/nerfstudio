# Phase 4 — Port LookCloser onto the Temporal 4D model (integration plan)

Goal: replicate ALL LookCloser stages on `instant-ngp-time` / `TemporalNGPField` to chase the
single-frame LookCloser targets (PSNR ~30 / SSIM ~0.70 + low artifact score), on the temporal dataset.
Start only after the current temporal experiments finish. Build on the winning temporal config
(H2 concat 3D+4D hash, static branch log2=21/max_res=4096, appearance embedding, long training).

## LookCloser's 8 stages (single-frame) — what to match
1. **2D frequency quantification (offline preprocess)** — `nerfstudio/scripts/lookcloser_preprocess.py`.
   Per image: train a small 2D Instant-NGP, progressive level regression, `f_2d` per patch = min level with
   SSIM>0.95. Output `{data}/lookcloser_frequencies/{stem}.pt` (H_p×W_p float32) + `.json` (patch_size=8,
   stride=8, min_res=16, max_res=8192, n_levels=16, ssim_thresh=0.95).
2. **3D frequency grid init from SfM** — 128³ grid, `f_grid = max_j f_3D`, `f_3D = f_2D·(focal/depth)`.
3. **FAS pixel sampler** — `nerfstudio/lookcloser_pixel_sampler.py`. Bucket pixels by freq level [0..15],
   oversample high-freq ~1:3 (sampling_ramp_start=1→end=3). Needs cached 2D maps per image.
4. **Feature re-weighting (Eq.6)** — in the field: down-weight hash levels above the grid's quantified
   level via erf smoothing. Config: `enable_feature_reweighting`, `feature_reweighting_strength`.
5. **Adaptive ray marching (ARM)** — `ray_sampling_mode="adaptive"`, nerfacc coarse traverse
   (`adaptive_coarse_step_size=0.00625`) + frequency-driven fine subdivision, `max_steps_per_ray=1024`.
6. **Losses** — charbonnier recon + distortion (0.01) + early depth loss (0.001, 5k steps).
7. **Runtime grid update every 1024 steps** — `lookcloser_pipeline._update_frequency_grid`: sample rays,
   render depth, project 2D→3D freq, grid `max()` update.
8. **Artifacts/defect metric** — `LookCloser/scripts/detect_structural_artifacts.py`: local-SSIM blob
   detection → `artifact_score` / `serious_artifact_score`. Runner can select checkpoint by artifact score.

Method config: `method_configs["lookcloser"]` (VanillaDataManager + LookCloserPixelSampler +
LookCloserPipeline + LookCloserModel, 200k iters, charbonnier). Runner: `run_lookcloser_quiet.py`.

## Temporal-specific challenges (decide during Phase 4)
- **DataManager conflict:** LookCloser uses `VanillaDataManager` + the FAS `LookCloserPixelSampler`.
  Temporal uses `ParallelDataManager(load_from_disk=True)` whose `RayBatchStream` has its OWN internal
  pixel sampler → FAS not directly compatible. Options: (a) port FAS sampling into a streaming-compatible
  sampler / custom RayBatchStream; (b) switch temporal back to VanillaDataManager with bounded
  `train_num_images_to_sample_from` (RAM-safe rolling cache, ~400 imgs) so FAS works as-is. (b) is simpler.
- **Per-image freq maps at temporal scale:** need a `.pt`+`.json` per (camera×frame) — ~2,523 train images.
  2D-NGP preprocessing per image is expensive. Mitigations: cameras are static across frames → the freq
  map for a given camera changes only where the scene moves; could (i) preprocess once per camera on a
  representative frame and reuse across that camera's frames, or (ii) preprocess a strided subset. Start
  with per-camera-once reuse (52 maps) as the cheap baseline, refine if needed.
- **Frequency grid temporality:** single-frame uses one 128³ spatial grid. For temporal start with ONE
  shared spatial grid (Option A, simplest — implicit temporal max); revisit per-time grids only if needed.
- **Feature re-weighting on which branch:** apply Eq.6 re-weighting to the 4D dynamic branch and/or the
  3D static branch of H2. Likely both (each is a hash grid). Match level→grid mapping.
- **Artifacts metric on temporal eval:** run detector over the 63 eval renders (3 cams × 21 times),
  aggregate artifact_score across times.

## Suggested implementation order (Phase 4)
1. Decide datamanager approach (lean (b): VanillaDataManager + bounded image cache + FAS).
2. Preprocess freq maps (per-camera-once reuse → 52 maps) into `{ds}/lookcloser_frequencies/`.
3. New `instant-ngp-time-lookcloser` method: temporal field + LookCloser field logic (feature reweighting),
   LookCloser pipeline (grid + updates), FAS sampler, ARM, charbonnier+distortion losses.
4. Wire artifacts metric into the runner's final eval.
5. Validate stage-by-stage (freq maps sane → FAS sampling → reweighting → ARM → full), then full run;
   compare PSNR/SSIM/LPIPS/artifact_score vs temporal H2 baseline and the LookCloser single-frame targets.

(Full stage details + exact file/line references captured from the 2026-06-28 exploration.)
