# From-Scratch Reproduction Recipe (LookCloser leader)

> **Audience: a future agent session with NO prior context.** This file is self-contained.
> Read it top to bottom before doing anything. Do not assume any other file is loaded.

---

## 0. TL;DR — what you are doing

Reproduce (or beat) the current LookCloser **leader** on the bounded HD scene `007740`,
**training from scratch (no checkpoint load)**, then **reduce run-to-run variance** so that
nearly every seed reaches leader-or-better metrics.

- **Phase A** — reproduce the leader from scratch with a single fixed seed. Iterate on
  config (not seed) until one run reaches the target.
- **Phase B** — only after Phase A succeeds: run multiple seeds and tune the occupancy/ARM
  warmup so the *spread* collapses (every run ≈ leader or better).

You are done when Phase A target is met **and** Phase B shows low variance across ≥3 seeds.

---

## 1. Target metrics (the bar to clear)

Evaluation = PSNR, SSIM, LPIPS over the **3 eval images**, plus the structural-artifact ROI
score (must be clean). Do **not** report loss.

| Run | PSNR | SSIM | LPIPS | ROI artifact |
|-----|------|------|-------|--------------|
| **Leader (reproducible, step 106316)** | **29.858** | **0.695** | **0.272** | **0** |
| Leader (peak online eval, step 91128 — checkpoint was pruned) | 29.917 | 0.700 | 0.280 | 0 |
| Fixed-step baseline (must beat) | 29.565 | 0.683 | 0.365 | 0 |

**Phase A success = a from-scratch run that reaches `PSNR ≥ 29.86`, `SSIM ≥ 0.695`,
`LPIPS ≤ 0.272`, `ROI artifact = 0`.** Beating it (higher PSNR/SSIM, lower LPIPS) is welcome.
Note: we found ~29.9 dB is roughly the quality ceiling for this scene/architecture; do not
chase PSNR 32, it is not reachable here.

---

## 2. How the historical leader was actually produced (important caveat)

**The leader was NOT trained from scratch in one shot.** It is the end of a multi-stage path:

1. ARM trained from scratch with finer coarse traversal (`adaptive_coarse_step_size=0.00625`)
   and artifact-aware checkpoint selection → clean checkpoints (seed-dependent windows).
2. Continued with a raised ARM frequency ceiling (`adaptive_max_frequency_level=13`) →
   checkpoint `maxfreq13_cont3/step-000038912.ckpt` (PSNR 29.535, ROI 0, FW off).
3. **Leader stage:** loaded that checkpoint and continued with **budget-aware ARM**
   (`max_steps_per_ray=1024`, per-ray dt scaling) + **Feature Reweighting on** +
   **charbonnier** loss. It *dropped* the frequency floor/ceiling
   (`adaptive_min_frequency_level=0.0`, `adaptive_max_frequency_level=None`). Reached
   PSNR 29.917 at step 91128.

Two consequences you must internalize:

- **Adaptive-from-scratch was historically the core failure mode** (empty/noisy frequency
  grid + weak early density → underfit or sample-cap saturation; several from-scratch runs
  collapsed to PSNR ~14 at the first eval). The **budget-aware ARM fix** (per-ray dt scaling
  so total samples ≤ `max_steps_per_ray`) removed the saturation/front-loading cause, so
  from-scratch is *more viable now than in that history* — **but it has not been verified
  end-to-end from scratch.** Treat Phase A as a real experiment, not a sure thing.
- The leader inherited a well-formed density field and frequency grid from stage 1–2. From
  scratch you must let those form first. The main lever for that is **`adaptive_warmup_steps`**
  (use fixed-step ray marching for the first N steps, then switch to adaptive). The leader
  had `adaptive_warmup_steps=0` *because it started from a good checkpoint*; from scratch you
  will likely need it > 0. See §6 fallbacks.

---

## 3. Environment & prerequisites (verify before training)

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
```

- **Repo / branch:** `/home/ubuntu/repos/nerfstudio`, branch `lookcloser/budget-aware-arm`
  (the budget-ARM fix + leader-recipe defaults live here). Confirm:
  `git -C /home/ubuntu/repos/nerfstudio rev-parse --abbrev-ref HEAD`
- **Dataset:** `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns`
  Must parse as **66 train + 3 eval** with `--eval-mode filename`.
- **Frequency maps (required for Frequency Grid + FAS):** the dataset must contain a
  `lookcloser_frequencies/` dir with ~132 files (66 train × `.pt` + `.json`). Verify:
  `ls .../007740_hd_aabb4_multicamera_eval3_ns/lookcloser_frequencies/ | wc -l` → expect 132.
  These maps were generated for `min_res=16, max_res=8192, n_levels=16`. **If you change the
  frequency schedule (`max_res`, `max_res_base`, `num_frequency_levels`) you MUST regenerate
  the maps**, or metadata validation will (correctly) fail.
- **Output dir:** `/fsx` has been full before. Write runs to local
  `/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs`.
- **Quiet runner** (use this; do not run `ns-train` in a TTY — the progress bars waste
  context): `LookCloser/scripts/run_lookcloser_quiet.py`. It launches `ns-train` with output
  redirected to `train_stdout.log`, prints compact `step=` / eval lines, monitors
  `metrics_compact.csv`, runs `ns-eval` + the artifact detector at the end, and writes
  `run_summary.json`.

---

## 4. Phase A — reproduce the leader from scratch (single fixed seed)

**Seed policy:** until Phase A target is met, use **one fixed seed and never change it** while
you iterate on config (so every result is reproducible and comparisons are clean). Use
`--seed 42` (the leader's seed; a principled fixed choice, not a cherry-picked lucky seed).
Do **not** seed-shop in Phase A — if a run misses, change the *recipe*, not the seed.

### Primary command (attempt 1: pure leader recipe, no checkpoint)

```bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
python /home/ubuntu/repos/nerfstudio/LookCloser/scripts/run_lookcloser_quiet.py \
  --experiment-name 007740_fromscratch_repro_s42 \
  --output-dir /home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs \
  --seed 42 \
  --data /fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns \
  --eval-mode filename \
  --scene-scale 1.5 \
  --scale-factor 1.0 \
  --max-res 8192.0 \
  --ray-sampling-mode adaptive \
  --max-steps-per-ray 1024 \
  --adaptive-coarse-step-size 0.00625 \
  --reconstruction-loss-type charbonnier \
  --distortion-loss-mult 0.01 \
  --grid-resolution 128 \
  --background-color black \
  --occupancy-warmup-steps 4096 \
  --occupancy-binary-warmup-steps 4096 \
  --max-num-iterations 200000 \
  --no-stop-on-no-improve \
  --eval-checkpoint best \
  --keep-all-checkpoints
```

Notes on the flags (do not "simplify" without understanding):
- **No `--load-checkpoint`** → from scratch. This is the whole point.
- **Feature Reweighting and FAS are ON by default** in this branch — do not disable them.
  (FAS gives clearly better LPIPS; PSNR difference vs no-FAS is within noise.)
- `--scene-scale 1.5`, `--scale-factor 1.0`, and `--max-res 8192.0` are **mandatory** and
  differ from the runner's generic defaults (`2.0` / `1.15` / `None`). The frequency maps
  require the 8192 schedule, and the leader's geometry normalization used scene_scale 1.5 +
  scale_factor 1.0 — getting these wrong silently changes the scene and breaks reproduction.
- `--no-stop-on-no-improve` + monitor manually: from-scratch evals can dip then recover
  (e.g. an FW adaptation dip), so don't let early-stop kill it prematurely. You will stop it
  yourself at plateau (see §5).
- `--keep-all-checkpoints` so you can re-select by artifact/ROI if needed.
- `--eval-checkpoint best` selects by `eval_all_psnr` (highest mean-PSNR over all 3 eval
  images, LPIPS as tie-break within 0.07 dB). If artifacts appear, switch to
  `--eval-checkpoint roi` (renders every checkpoint, picks an artifact-clean one — slower).

### Monitoring behavior (do this, don't wait to be asked)

- **Always monitor a launched training run with `/loop`** until it succeeds or fails. Read
  `metrics_compact.csv` in the run dir each eval boundary (evals land at multiples of the
  `--step-interval`, default **15188** → 15188, 30376, 45564, 60752, 75940, 91128, 106316…).
- At each eval, log `step / PSNR / SSIM / LPIPS`. Watch the PSNR trajectory.
- The leader's continuation reached the target around step 75940–106316. From scratch will
  likely need **more** steps to get there (the field has to form first). Expect the first
  eval or two to be lower; that is normal **unless it collapses** (see §6).

---

## 5. When to stop, and how to confirm success

**Stop the run when** either:
- PSNR has plateaued (no improvement beyond ~+0.02 dB) for **2–3 consecutive evals**, or
- the target is clearly met and metrics are flattening.

To stop: kill the runner process, then `pkill -f "ns-train.*<experiment-name>"` to clear any
orphaned `ns-train`.

**Confirm success** on the selected checkpoint:
1. Read `run_summary.json` → final `ns-eval` PSNR/SSIM/LPIPS and the `artifact` block.
2. **ROI artifact score must be 0** (`artifact.roi` in the summary). If non-zero, the run is
   not acceptable even if PSNR is high — re-select with `--eval-checkpoint roi`, or treat the
   run as failed and apply a §6 fallback.
3. Compare against the §1 target. Save the rendered eval images and metrics to a new file
   under `LookCloser/experiments/` (one file per topic; include: What was tested, Results
   table with render paths, Insights/next steps).

---

## 6. If Phase A does NOT reproduce — fallback options (try in this order)

Change the **recipe**, keep **seed 42** fixed. After each change, relaunch and monitor.

1. **Collapse at the first eval (PSNR < ~20, often ~14):** classic adaptive-from-scratch
   failure — the field/frequency grid are not formed yet. Add a fixed-marching warmup:
   `--adaptive-warmup-steps 2048`. If still collapsing, try `4096`, then `8192`. This uses
   fixed ray marching for the first N steps before switching to adaptive ARM, giving the
   density field and frequency grid time to form. **This is the single most likely fix for
   from-scratch.**
2. **Trains but PSNR stalls well below target (e.g. < 29.3 and flat):** it is undertrained
   or under-detailed.
   - Confirm it actually ran long enough (≥ ~90k steps of real improvement).
   - Re-introduce the historical detail lever from stage 2: cap the ARM frequency ceiling
     with `--adaptive-max-frequency-level 13` (this was the documented best ARM-only detail
     lever). Optionally add a moderate floor `--adaptive-min-frequency-level 4` (helped thin
     cables in some seeds; can perturb others — try ceiling-only first).
3. **PSNR good but ROI artifacts appear:** switch selection to `--eval-checkpoint roi`
   (artifact-aware, renders all checkpoints). If artifacts persist across checkpoints, the
   field entered a dirty regime — raise `--occupancy-warmup-steps` and
   `--occupancy-binary-warmup-steps` (see §7; bad early occupancy pruning of thin/peripheral
   detail is a known cause).
4. **LPIPS too high (PSNR/SSIM fine):** FAS is already on (it is the LPIPS lever). Make sure
   you did not accidentally disable it. Do not over-tune here; ~0.27 is the target.
5. **Last resort — staged path:** if single-stage from-scratch refuses to reach target,
   reproduce the historical path explicitly: (a) train ARM from scratch with
   `--adaptive-coarse-step-size 0.00625 --adaptive-max-frequency-level 13` and warmup until
   you get a clean ~29.5 checkpoint, then (b) continue that checkpoint with the budget-ARM +
   FW + charbonnier leader recipe (`--load-checkpoint <that ckpt>`, FW on, no freq cap). This
   is exactly how the leader was made and is the highest-confidence route if (1)–(4) fail.

**Do not** start from a *dense* (fixed-step, e.g. 16384 samples/ray) checkpoint and switch to
ARM — that caused a catastrophic ~5 dB PSNR drop historically (dense→ARM incompatibility).
Any checkpoint you continue from must itself be ARM-trained.

---

## 7. Phase B — variance reduction (only after Phase A succeeds)

Goal: make the recipe **stable** — nearly every seed reaches leader-or-better, not just a
lucky one. Now you *do* vary the seed.

1. **Measure the baseline spread.** Run the Phase-A winning recipe on **≥3 seeds**
   (e.g. 42, 43, 44; you may use `scripts/run_lookcloser_sweep.py` for multi-seed sweeps, or
   launch the quiet runner per seed with distinct `--experiment-name` and `--seed`). Record
   PSNR/SSIM/LPIPS/ROI per seed and the min/mean/spread.
2. **Diagnose bad runs.** A primary variance source is **bad occupancy-grid initialization**
   (early binary pruning removes thin/peripheral detail and never recovers). If one or more
   seeds land clearly below the leader or show ROI artifacts:
   - **Double `--occupancy-warmup-steps`** (4096 → **8192**). This is the user-designated
     first lever. Re-run the bad seeds (and ideally all seeds) and re-measure spread.
   - If still unstable, also raise **`--occupancy-binary-warmup-steps`** (4096 → 8192). This
     keeps the binary occupancy fully occupied longer during cold start, preventing premature
     pruning while `occs` continue to update.
   - If early *adaptive* instability contributes, add/raise `--adaptive-warmup-steps`
     (e.g. 2048–4096) so every seed forms density before switching to ARM.
   - Other safe knobs to experiment with for stability (change one at a time, keep notes):
     `--occupancy-ema-decay` (0.95 default; higher = smoother but slower to adapt),
     `--occupancy-update-interval`. Avoid aggressive occupancy changes that historically
     regressed quality (dilation, very low `occ_thre`); prefer the warmup levers.
3. **Acceptance for Phase B:** across the seed set, the **worst** run still meets the §1
   target (PSNR ≥ 29.86, SSIM ≥ 0.695, LPIPS ≤ 0.272, ROI = 0), or is within noise of it,
   with a clearly tighter spread than the baseline sweep. Document the final stable recipe and
   the per-seed table.

---

## 8. Reporting & hygiene

- Save results under `LookCloser/experiments/` (one file per topic). Use the standard
  structure: **What was tested** (hypothesis/config), **Results** (tables + render paths),
  **Insights** (what was learned + next steps). Keep it concise.
- After a major recipe change that becomes the new accepted path, update
  `LookCloser/architecture.md` (keep it concise) and, if defaults should change, the
  `lookcloser` entry in `nerfstudio/configs/method_configs.py`.
- Keep noisy `ns-train`/`ns-eval` output in log files, not the chat.
- Commit code/doc changes on the `lookcloser/budget-aware-arm` branch (or a child branch).
- When a long run finishes or plateaus, send a one-line push notification with the outcome.

---

## 9. Quick reference — the leader recipe values (verified from the leader's `config.yml`)

| Param | Value |
|-------|-------|
| ray_sampling_mode | `adaptive` (budget-aware ARM, per-ray dt scaling) |
| max_steps_per_ray | `1024` |
| adaptive_coarse_step_size | `0.00625` |
| adaptive_min_frequency_level / max | `0.0` / `None` (no floor/ceiling) |
| enable_feature_reweighting / strength | `True` / `1.0` |
| enable_fas | `True` (default on this branch) |
| reconstruction_loss_type | `charbonnier` |
| distortion_loss_mult | `0.01` |
| grid_resolution | `128` |
| num_frequency_levels / min_res / max_res / max_res_base | `16` / `16` / `8192` / `2048` |
| background_color | `black` |
| occupancy_warmup_steps / binary_warmup_steps | `4096` / `4096` |
| occupancy_occ_thre / ema_decay / update_interval | `0.01` / `0.95` / `16` |
| train_num_rays_per_batch | `4096` |
| optimizer (fields) | Adam lr `1e-2` → `1e-4`, max_steps `200000` |
| dataparser | scene_scale `1.5`, scale_factor `1.0`, center `focus`, orientation `up`, auto_scale_poses `True`, downscale `1`, eval_mode `filename` |
| seed | `42` |
| adaptive_warmup_steps | `0` **in the leader (it continued from a checkpoint); expect to need > 0 from scratch** |
