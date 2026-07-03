# Temporal 4D Instant-NGP — Recipe & Handoff (read me first)

Self-contained guide for a fresh session. Goal: extend bounded Instant-NGP from `F(x,y,z)→σ,c` to
**`F(x,y,z,t)→σ,c`** (time as a 4th hash-grid coordinate), then port the LookCloser frequency-aware
pipeline (ARM + feature-reweighting) onto it. This is NOT deformation / canonical-space / 4D-occupancy.

Detailed experiment log (everything tried) lives in `time.md`. This file is the **high-level** version:
what the leaders are, exact hyperparameters, how to launch, and how the data was made.

---

## 1. Dataset

- **Path:** `/opt/dlami/nvme/temporal_ds_stride7_45f`  (local NVMe; `images/`, `transforms.json`,
  `build_manifest.json`, `sparse_pc.ply`)
- **Source clip:** **6A_4_EXR** multicamera capture (base `/fsx/oregon/tank_bkup/6A_4_EXR/<frame>/<CAM>.exr`,
  6144×3072 float16 linear HDR; frames available 7389–8050). Same rig as the static single-frame scene
  `007740_hd_aabb4_multicamera_eval3_ns`. Cameras STATIC across time; only people move.
- **Frames:** **45 at stride 7**, starting at **007740** (7740, 7747, …, 8048). 007740 is the first frame.
- **Train:** 45 timesteps × **52 cameras** = **2340** images. **Eval:** 45 timesteps × **3 held-out cameras**
  (D004_A014, E004_B014, I004_D014) = **135** images. Total **2475** (1920×1080).
- **`transforms.json`:** `--eval-mode filename` (train=`frame_train_<CAM>_<frame>.jpg`, eval=`frame_eval_...`);
  each entry: rig intrinsics + transform_matrix reused verbatim, `"frame_idx"`, `"time"=(frame-7740)/308 ∈ [0,1]`.
- **Frequency maps for ARM/FR:** NOT yet regenerated for this dataset — rebake per-image freq maps + the 3D
  grid (`bake_frequency_grid.py`) before running ARM/FR here.

### How the images were color-graded (FIXED — matches static 007740)
EXR→JPG via `/home/ubuntu/repos/look-closer/hollywood_grade_batch.py` (the REAL static pipeline; its default
input is literally `.../6A_4_EXR/007740`). Config: `GRADE=True, EXPOSURE_TARGET=0.44, DISPLAY_BRIGHTNESS=1.10,
SATURATION=0.96, BLACK_LIFT=0.055, VIGNETTE_STRENGTH=0.16, CONTRAST_STRENGTH=4.0`, ACES filmic + teal/orange
split-tone, **16:9 center-crop (cw≈5461) → 1920×1080**. **NO per-frame auto-exposure:** each camera's gain is
computed ONCE on frame 007740 and frozen for all 45 frames (gains 50.5–64.0) → temporally consistent, no flicker.
Verified vs static 007740: mean/p99 within ±0.002 (tonally exact); SSIM ≈0.93 capped only by a sub-pixel
resample-sharpness residual (a tiny Gaussian blur → SSIM 0.99). This REPLACES the earlier dull grade
(old `color_corretion.py`, EXPOSURE_TARGET 0.30, mean≈0.40/p99≈0.84).
**NOTE:** all leader metrics in §3 were measured on the OLD dull dataset — retrain on this dataset to get
final numbers (they will change).

---

## 2. How to train

Env: `conda activate /home/ubuntu/anaconda3/envs/nerfstudio`. Runner:
`LookCloser/scripts/run_temporal_ngp_quiet.py` (quiet: logs compact CSV, early-stops on eval plateau,
runs final `ns-eval` + renders + LookCloser artifact score). **Runner defaults = the NO-ARM leader**, so:

### A) No-ARM leader (default — no hyperparameters needed)
```bash
python LookCloser/scripts/run_temporal_ngp_quiet.py --experiment-name temporal_leader_noarm
```
Encodes: method `instant-ngp-time`, `--hypothesis H2` (concat 3D-static + 4D-dynamic hash → 1 MLP),
static branch log2=21/max_res=4096, 4D branch log2=21/max_res=4096, occ warmup+binary 4096,
MSE loss, black bg, near 0.01, 8192 rays/batch, max 76000 iters, dataset eval-12th above.

### B) Perceptual leader = ARM + frequency grid + feature-reweighting
Two steps. First bake a real 3D frequency grid from a converged checkpoint (ARM needs a REAL per-scene
frequency signal — a constant fallback gives no benefit):
```bash
python LookCloser/scripts/bake_frequency_grid.py \
  --config <run>/config.yml \
  --output /opt/dlami/nvme/temporal_runs/freq_grid.pt \
  --resolution 128 --min-res 16 --max-res 8192 --num-levels 16 \
  --freq-map-dir lookcloser_frequencies --pixels-per-image 20000
```
Then warm-restart with ARM + FR (LR-reset = fresh high LR; vary/log the seed):
```bash
python LookCloser/scripts/run_temporal_ngp_quiet.py \
  --experiment-name temporal_leader_armfr --seed 43 \
  --static-log2-hashmap-size 23 --static-max-res 8192 \
  --load-checkpoint <converged>.ckpt --load-scheduler False \
  --enable-arm --adaptive-coarse-step-size 0.0125 --max-steps-per-ray 1024 \
  --transmittance-threshold 0.0 --adaptive-interval-level-mode midpoint \
  --mm frequency-grid-path=/opt/dlami/nvme/temporal_runs/freq_grid.pt \
  --mm frequency-grid-resolution=128 --mm adaptive-max-frequency-level=12.0 \
  --mm enable-feature-reweighting=True --mm feature-reweighting-strength=1.0 \
  --max-num-iterations 120000 --stop-on-no-improve
```
Notes: `--load-scheduler False` = the LR-reset "warm restart" ritual (fresh scheduler → LR back to ~0.01;
loading always resumes the global step). FR is only wired for **H2** (separate enc3/enc4 expose raw hash
features); it needs a loaded frequency grid. ARM+FR also raises static capacity to log2=23/max_res=8192.

---

## 3. Current leaders (on the dull-graded dataset — see caveat)

| model | recipe | PSNR↑ | SSIM↑ | LPIPS↓ | artifacts↓ |
|---|---|---|---|---|---|
| **No-ARM leader** (best PSNR) | H2, static+4D log2=21/r4096, occ 4096, MSE, ~76k | **26.21** | 0.786 | 0.341 | 0.004 |
| **ARM+FR leader** (best perceptual) | above + static l23/r8192 + ARM(freq grid) + feature-reweight, warm-restart LR-reset, ~112k | 25.40 | **0.800** | **0.308** | **0.000** |

Verdict: **ARM (with real baked frequency maps) + FR** wins SSIM/LPIPS and gives a perfect 0/51 artifact
score; the plain no-ARM model keeps the PSNR crown (ARM trades ~0.8 dB PSNR for perceptual quality —
matches the LookCloser paper). Both far exceed the static single-frame baseline
(24.42 / 0.640 / 0.460) on the 51-frame clip.

### Checkpoints & renders (all 51 eval views: GT|pred 2-panel + depth + accumulation)
- No-ARM leader: `/opt/dlami/nvme/temporal_runs/007700_007900_temporal_h2_4dr4096_EVAL12_CONTINUE_52cam/instant-ngp-time/20260629_154233/` (best `step-000075940`; renders `renders_best_step-000075940/`)
- ARM+FR leader: `/opt/dlami/nvme/temporal_runs/RECIPE_mse_cap_ARMfreq_FR/instant-ngp-time/20260702_040131/` (best `step-000112000`; renders `renders_best_step-000112000/`)
- Baked frequency grid: `/opt/dlami/nvme/temporal_runs/freq_grid_mse_cap.pt`

---

## 4. Key code (all under the temporal port)
- `nerfstudio/fields/temporal_ngp_field.py` — TemporalNGPField (H1/H2/H3) + optional feature-reweighting.
- `nerfstudio/models/temporal_instant_ngp.py` — model + max-over-time occupancy; ARM; Charbonnier/distortion
  options (Charbonnier/distortion HURT in 4D — leave off); `frequency_grid_path` loads the baked grid → ARM.
- `nerfstudio/model_components/temporal_arm_sampler.py` — budget-aware ARM; `freq_level_fn` = grid.query.
- `nerfstudio/model_components/lookcloser_grid.py` — FrequencyGridManager (query/update_max/level↔res).
- `nerfstudio/configs/method_configs.py` — methods `instant-ngp-time` (leader), `-personsample`, `-decomp`.
- `LookCloser/scripts/run_temporal_ngp_quiet.py` — runner (leader defaults). `bake_frequency_grid.py` — grid baker.

## 5. What did NOT help (don't redo — see time.md)
- Person/motion oversampling (no gain; artifacts already ~0). Charbonnier + distortion loss (hurt PSNR &
  artifacts in 4D). 4D-branch capacity bump l22 (raised artifacts). ARM with constant fallback level (needs
  the real baked frequency grid). Decomposition H2D (collapses unless sparsity weight tiny).
- Ultimate target from the static single-frame leader was PSNR≈29; on the 51-frame 4D clip the PSNR ceiling
  is ~26 (more content per model). Fix the color grade before chasing further absolute-metric gains.
