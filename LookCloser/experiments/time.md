# Temporal Instant-NGP (`instant-ngp-time`) — Running Log

Chronological log of important results, decisions, and insights for the temporal
bounded Instant-NGP effort (`F(x,y,z,t)` via a 4D hash grid, hypothesis H1).
Newest entries at the bottom. Times are UTC.

---

## 2026-06-26 — Phase 2 implementation complete

**What was built (existing `instant-ngp` / `instant-ngp-bounded` untouched):**
- `nerfstudio/data/dataparsers/nerfstudio_dataparser.py`: additive per-frame `time` read →
  populates `Cameras.times` only when *every* frame has `time` (backward compatible).
- `nerfstudio/field_components/encodings.py`: `HashEncoding(in_dim=...)`; tcnn passes
  `n_input_dims=in_dim`; torch path raises `NotImplementedError` for `in_dim!=3`.
- `nerfstudio/fields/temporal_ngp_field.py`: `TemporalNGPField` + `TemporalHashMLP`.
  Density from a 4D grid over `concat([normalized_xyz, clamp(t,0,1)])`; color head =
  SH(dir) + geo features (time enters color only via geo features, per H1). Time-aware
  `density_fn(positions, times)` for the occupancy path.
- `nerfstudio/models/temporal_instant_ngp.py`: `TemporalInstantNGPModel/Config`.
  Keeps the 3D nerfacc occupancy grid; `occ_eval_fn` = **max over training times**
  (point- and time-chunked to bound VRAM). Training times sourced from
  `pipeline.datamanager.train_dataparser_outputs.cameras.times`.
- `nerfstudio/configs/method_configs.py`: registered `instant-ngp-time` (cloned from
  bounded; default dataparser `NerfstudioDataParserConfig(eval_mode="filename")`).
- `LookCloser/scripts/run_temporal_ngp_quiet.py`: quiet runner; exposes
  `--log2-hashmap-size/--max-res/--num-levels/--features-per-level/--hidden-dim/`
  `--occ-time-chunk/--occ-update-times`; full-run defaults (no early stop).

**Stage-0 smoke (PASSED).** 120-iter `ns-train instant-ngp-time` on the *reference*
single-frame dataset (007740, no `time` field → exercises `times=None` fallback):
- EXIT=0, no NaN/error/traceback.
- Train PSNR 12.8 → 18.6 over 110 steps; loss monotonically down.
- Eval@60 ran full metric+render path: eval_all PSNR 17.5 / SSIM 0.44.
- ~45–53k rays/s, ~10 GB VRAM. Occupancy update worked.
- **Insight:** confirms the whole pipeline + the additive dataparser change are
  backward-compatible (no regression to bounded).

**Field micro-test (PASSED).** Field forward, time-conditioned `density_fn`, and the
max-over-time occupancy loop (201 times, chunked) all produce finite outputs on CUDA.

---

## 2026-06-26 — Dataset prep + HARDWARE REALITY (scope change)

**Dataset agent established the camera↔EXR mapping (the hard part):**
- Reference `transforms.json`/COLMAP use only renamed `frame_train_*/frame_eval_*`;
  no original EXR camera names, and intrinsic/pose matching were unreliable.
- Mapping recovered by **image-content matching** ref `frame_*.jpg` ↔
  `/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_data/007740/hd/<CAM>.jpg`: clean bijection,
  second-best/best confidence ratio ≥ 6357× (unambiguous).
- **Eval cameras (held out):** `D004_A014`, `E004_B014`, `I004_D014`.
- All 201 source frames `007700..007900` present (incl. 007900); 69 cams/frame; EXR 6K.
- **Conversion geometry:** center-crop width 6144→5461 (keep full height 3072) then
  resize to 1920×1080 — reverse-engineered against the `hd/` ground truth (geom MSE 0.04
  vs 0.80 for naive squash). Grade = exact `color_corretion.py` math (bit-identical).
- **Caveat:** new grade does NOT pixel-match the old reference look (reference was made
  by an earlier grading step not preserved in-repo; MAE ~42). Internally consistent
  across all frames (fine for training); calibration reused verbatim (unaffected).

**HARDWARE WALL discovered:** machine has **4 CPU cores, 30 GB RAM** (22 free), 1×L40S 46GB.
- EXR→JPG ≈ 29 s/EXR, CPU-bound → all 13,266 train imgs ≈ **28 h** convert.
- Nerfstudio caches *all* train imgs in RAM (uint8): 13,266 × full-res ≈ **82 GB** → won't fit.
- ⇒ "all 66 cams × all 201 frames @ full res" is **infeasible to train here**.

**Decision (user):**
- **Training frames = every 4th** → 66 cams × 51 frames {7700,7704,…,7900} = **3,366 train imgs**.
- **Eval unchanged** = 3 held-out cams × every 10th frame {7700,7710,…,7900} = **63 imgs**.
- **Full HD (1920×1080)**, but **stream images from disk — do NOT cache all in RAM.**
- Eval times include off-grid values (e.g. t=0.05) → tests temporal interpolation.

**Actions in flight:**
- Re-tasked data-prep agent: stop the 28h full conversion; rebuild `transforms.json` for the
  reduced set; reconvert only the 3,429 needed images (idempotent). New convert ETA ~7h.
- Exploring whether nerfstudio exposes a disk-streaming dataloader flag (vs. having to
  implement lazy per-batch loading) for the full-HD-no-cache requirement.

**Open / next:**
- Confirm disk-streaming path (CLI flag or code change).
- Time-conditioning smoke on the real temporal dataset (verify `time` reaches field, >1 unique time).
- Then full 60,752-iter H1 baseline → H1-A/B/C/D sweep → H2/H3; results table + selection.

---

## 2026-06-26 — Disk streaming resolved + scope finalized

**Final dataset scope (user-confirmed):**
- Train stride **4** → frames {7700,7704,…,7900} = 51 timestamps × 66 cams = **3,366 train imgs** (7900 incl).
- Eval unchanged: 3 held-out cams × every-10th frame = **63 imgs**. transforms.json rebuilt & verified.
- Conversion restarted for the 3,429-image subset; ETA ~6.5–7h (4 CPU cores), running in background.

**Disk-streaming decision (important):**
- `VanillaDataManager` (used by `instant-ngp` via `DynamicBatchPipeline`) has NO disk-streaming flag —
  always caches all images (3,366 full-HD ≈ 21 GB → would OOM the 30 GB box).
- True streaming = `RayBatchStream(load_from_disk=True)`, exposed only by `ParallelDataManager`.
- `ParallelDataManager` subclasses `DataManager` (not `VanillaDataManager`) → fails
  `DynamicBatchPipeline`'s `isinstance` check, and RayBatchStream has no `train_pixel_sampler`
  (dynamic batching can't propagate to worker procs). So DynamicBatch + streaming is incompatible.
- **Resolution:** switched `instant-ngp-time` to **`VanillaPipeline` + `ParallelDataManagerConfig(load_from_disk=True,`**
  **`dataloader_num_workers=3, prefetch_factor=4)`** — existing, tested streaming code; no custom class needed.
  Trade-off: lose DynamicBatch's dynamic ray-count tuning (fixed 8192 rays/batch) — acceptable.
- Verified `RayBatchStream` generates rays via `RayGenerator(dataset.cameras)` → **ray `times` propagate**
  (since the dataparser populates `cameras.times`).

**Streaming smoke (PASSED).** 120-iter `instant-ngp-time` (VanillaPipeline + ParallelDataManager,
`--load-from-disk True`) on reference 007740 (times=None fallback): EXIT=0, no NaN/pickle errors,
"Training Finished", train PSNR→19.9, eval_all PSNR 17.6/SSIM 0.44, ~80–108k rays/s, ~11 GB VRAM.
⇒ disk-streaming pipeline + temporal model validated end-to-end.

**Next:** wait for conversion (~6.5h) → time-conditioning smoke on real temporal data
(assert cameras.times populated, >1 unique time reaches field) → full 60,752-iter H1 baseline →
H1-A/B/C/D sweep → H2/H3.

---

## 2026-06-27 — Conversion crash + restart; time plumbing verified on real data

**Incident:** EXR→JPG conversion **died at 2026-06-26 18:10** at only **659/3,366 train** images
(all 63 eval done). Likely the data-prep agent's background worker process-group was torn down when
that agent completed. My first completion-waiter ALSO had a bug: its command line contained the string
`convert_worker`, so `pgrep -f convert_worker` matched the waiter itself → never read 0 → never fired.
(Lesson: monitor by output/count, not by grepping a process name the monitor command also contains.)

**Fix:** Relaunched the 4 idempotent workers fully detached via `setsid nohup` (base anaconda python
`/home/ubuntu/anaconda3/bin/python3`, which has OpenEXR 3.4.11; the `nerfstudio` env lacks it). Workers
skip existing JPGs and resume. New monitor (`bbas57xdc`) polls the **train-jpg count only** (no
process-grep): fires on count≥3366 (done) or 30-min stall. Observed rate ~12/min total ⇒ ETA ~3–4h.

**Time plumbing VERIFIED on the real temporal transforms.json** (no images needed — dataparser only
reads transforms.json to build cameras):
- train: 3,366 imgs, **51 unique times**, min 0.0 / max 1.0.
- eval: 63 imgs, **21 unique times**, 0.0–1.0.
⇒ additive dataparser read works; `cameras.times` populated correctly; time will reach the field.

Everything is now validated except a real time-conditioned training run, which is gated only on the
in-progress image conversion.

**Root cause of the crash (confirmed by data-prep agent):** transient **/fsx disk-full** event (shared
21T FS briefly hit 100%) → `No space left on device` + EXR read failures. Recovered to ~1.7T free.

**Ownership cleanup:** both main and the data-prep agent had (re)launched workers → risk of two sets
sharing the same `dst.tmp` path (corruption). Resolved: killed ALL workers, **validated all 766 existing
JPGs (0 corrupt, no .tmp litter)** — no collision damage occurred — then relaunched exactly **4 clean
detached workers**. Told the data-prep agent to **stand down**; main is now sole conversion owner with one
count-based monitor (`bbas57xdc`). ~707/3366 train, ETA ~4–6h.

---

## 2026-06-27 — /fsx filled AGAIN → relocate dataset to local NVMe

**Second stall:** workers FINISHED their shards but with **849 errors** — `/fsx` hit **100%** again
(shared 20TB FS; our own output is only ~2GB, so an external writer filled it). Result: 2,767/3,366
train converted; the 849 missing were ~16 whole cameras (L*/M*/N*/O*) being written when disk hit 0.
Verified the "Unable to open .exr" errors were disk-full artifacts — those source EXRs exist & read
fine (75MB each) once space returned.

**Fix — relocate dataset to local NVMe (`/opt/dlami/nvme`, 215G free):**
- Decouples our writes from the flaky shared `/fsx`, AND makes training disk-streaming far faster
  (local NVMe vs network Lustre — big win since we stream images every batch).
- New dataset dir: **`/opt/dlami/nvme/temporal_ngp_ds`** (transforms.json + sparse_pc.ply + images/).
- Copied the 2,830 already-good JPGs locally; launched 4 workers (`convert_worker_local.py`, OUT=local)
  to convert the remaining ~597, reading EXRs from `/fsx` (read-only, works even at 100%), writing JPGs
  to local. Monitor `bv8c27rkx` (local count). ETA ~70 min.
- **Quiet runner `DEFAULT_DATA` updated to `/opt/dlami/nvme/temporal_ngp_ds`.**
- (The `/fsx` copy at .../007700_007900_full_temporal_ngp_ns remains as-is; local is now canonical.)

---

## 2026-06-27 — /fsx hard-degraded; proceed on 52-cam frozen set; H1 baseline launched

**/fsx is hard-down for the remaining reads:** scan of the 843 missing source EXRs → **0 readable**,
0 missing, 843 `Errno 61 No data available` (Lustre OSTs full → file data can't be served though
metadata exists). `/fsx` still 100% (18G). Backfill passes made +0 progress. ⇒ the **17 missing train
cameras (L/M/N/O block) are unrecoverable until /fsx gets free space** (external infra issue).
Backfill loop left running best-effort (writes only `transforms_full_regen.json`, never the live file).

**Decision — proceed now on the available data (don't block the GPU):**
- **Frozen experiment dataset:** `/opt/dlami/nvme/temporal_ngp_ds`, live `transforms.json` =
  **2,523 train (52 cams: 49 full + 3 partial) + 63 eval (all 3 eval cams, complete)**, 51 train times.
  Full 3,429 manifest preserved as `transforms_full_3429.json` for later if /fsx recovers.
- Whole sweep runs on this frozen set for consistency. Caveat to note in results: missing L/M/N/O
  camera block = a contiguous viewpoint gap; eval cams (D/E/I) are present.

**Tooling lesson:** long `ns-train` runs MUST be launched detached/background (foreground hit the
2-min tool timeout). Using the quiet runner via `nohup ... &`.

**H1 baseline launched (background):** `run_temporal_ngp_quiet.py`,
exp `007700_007900_temporal_h1_pure4d_l20_r512_52cam`, log2=20/max_res=512/levels=16/feat=2,
full schedule (max 60,752; eval/save 15,188), output `/opt/dlami/nvme/temporal_runs`. Disk-streaming
from local NVMe. Monitoring CSV next.

**H1 baseline confirmed training:** GPU ~75% / 18.8 GB, no errors. Rate **~7–8 it/s** ⇒ ~2.2 h for
60,752 iters (slower than vanilla NGP: max-over-time occupancy ≈32× density evals + full-HD streaming
on 4 cores). First eval/CSV row at step 15,188 (~30 min). Waiters: `bs6qsf0sx` (completion),
`b8nav6cc4` (first eval). Run dir:
`/opt/dlami/nvme/temporal_runs/007700_007900_temporal_h1_pure4d_l20_r512_52cam/instant-ngp-time/20260627_093142`.
⚠️ Sweep cost note: ~2.2 h/run × 7 planned runs ≈ 15 h of GPU. If too slow, levers = reduce occupancy
update frequency or occ_update_times.

**✅ FIRST RESULT — temporal method works.** H1 baseline @ step 15,188 (25% of training):
**eval_all PSNR 20.54, SSIM 0.636** (LPIPS at final eval). Reference *static single-frame* baseline =
PSNR 24.42 / SSIM 0.640 / LPIPS 0.460. So at only 25% training, on the harder 4D task (52 cams × 51
timesteps, eval includes off-grid interpolation times t∉train set), **SSIM already ≈ static baseline** —
the 4D hash grid is learning genuine spatiotemporal structure, time is reaching the field, occupancy
union-over-time is working. PSNR lower (expected: 4D collision pressure, partial training, fewer cams).
Run continues to 60,752; completion waiter `bs6qsf0sx` will capture final PSNR/SSIM/LPIPS.

---

## 2026-06-27 — H1 baseline COMPLETE; full sweep automated; H2/H3 implemented

**H1 baseline (log2=20, max_res=512, 52 cams) — FINAL:** best ckpt step 45,564 (by eval_loss):
**PSNR 20.31 / SSIM 0.643 / LPIPS 0.660** (lpips_std 0.14). Walltime ~88 min, ~11 GB VRAM, eval
469k rays/s. Trend over training: PSNR 20.54→20.31 (slight ↓), SSIM 0.636→0.643 ↑, **LPIPS 0.706→0.660 ↑**
— LPIPS (primary) and SSIM improve with training; PSNR mildly trades off. (Note: runner selects best by
eval_loss, which aligned with best LPIPS here.)

### Running results table (52-cam frozen set; eval = 3 cams × 21 times incl. off-grid)
| exp | hyp | log2 | max_res | levels | feat | iters | walltime | VRAM | PSNR | SSIM | LPIPS | sel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h1_l20_r512 | H1 | 20 | 512 | 16 | 2 | 60752 | ~88m | ~11GB | 20.31 | 0.643 | 0.660 | — |
| h1a_l19_r512 | H1 | 19 | 512 | 16 | 2 | 60752 | ~88m | ~11GB | 20.40 | 0.636 | 0.674 | — |
| **h1c_l21_r512** | H1 | **21** | 512 | 16 | 2 | 60752 | ~88m | ~11GB | 20.40 | **0.650** | **0.645** | ★ H1 leader |
| h1d_l20_r1024 | H1 | 20 | 1024 | 16 | 2 | 60752 | ~88m | ~11GB | **20.50** | 0.644 | 0.650 | — |
| **h2_concat3d4d** | H2 | 19(4D)+19(3D) | 512 | 16 | 2 | 60752 | ~95m | ~30GB | 20.64 | **0.695** | **0.491** | ★★ OVERALL LEADER |
| h3_3dplusT | H3 | 19(3D)+t | 512 | 16 | 2 | 60752 | ~88m | ~11GB | 18.44 | 0.641 | 0.571 | — |

### 🏆 H2 WINS DECISIVELY (architecture comparison done)
LPIPS↓: **H2 0.491** ≪ H3 0.571 < H1-best 0.645. SSIM↑: H2 0.695 ≫ rest. H2's separate 3D static
branch frees the 4D grid from static content → far less collision pressure → LPIPS 0.491 (approaching
the *static single-frame* baseline 0.460, on a much harder temporal task). H3 (scalar-t) too weak
(PSNR 18.4) → confirms genuine 4D hash capacity is needed, not just weak time conditioning.

### Max_res sweep v2 — applied to the WINNER (H2) [DONE] — user insight confirmed
| exp | static_max_res (log2=21) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| h2_s2048 | 2048 | 20.99 | 0.712 | 0.442 |
| **h2_s4096** | 4096 | 20.97 | 0.712 | **0.4346** ★ best LPIPS |
| h2_s8192 | 8192 | **21.04** | 0.712 | 0.4355 (best PSNR) |
| h1_l21_r8192 (pure 4D) | 8192 | 20.46 | 0.643 | 0.633 |

**Confirmed: static branch max_res=512 WAS a major bottleneck.** H2 LPIPS 0.491→0.435 (PSNR→21.0)
by raising static branch 512→4096. **Returns saturate ~4096** (4096≈8192 within noise) → no 16384 needed.
**LPIPS 0.435 now beats the static single-frame baseline (0.460)** on a 51-frame temporal task.
Pure-H1 r8192 helped only mildly (0.650→0.633) ⇒ architecture (H2) >> resolution for the 4D case.

### Variance check (running) — top-2 @ seed 7
s4096 vs s8192 differ <0.001 LPIPS (within variance). Running 2nd seed (seed=7) of both
(`variance_driver.sh`, waiter `bp7kvw5yw`); will pick the simpler (4096) unless seed-7 separates them.

### CURRENT BEST: **H2, static_log2=21 / static_max_res=4096, 4D branch log2=19/r512**
PSNR 20.97 / SSIM 0.712 / LPIPS 0.4346. (seed-42; variance confirmation pending)

---

## 2026-06-28 — Match static baseline's params: ADD APPEARANCE EMBEDDING

**User insight:** `instant-ngp-bounded` defaults gave the good static metrics; we should match them.
Audit vs my `TemporalNGPField`: matched hidden_dim(64)/num_layers(2)/color(3×64)/geo_feat(15)/features(2),
but **the static NerfactoField uses a 32-dim per-image appearance embedding by default** (it had
`appearance_embedding_dim=32`) — my temporal field had **none**. For multi-camera captures (per-camera
color/exposure), this is the likely missing lever. Also static used max_res=2048 (my 4D branch=512; static
branch already 4096≥2048).

**Implemented:** optional appearance embedding in `TemporalNGPField` (port of NerfactoField logic:
`Embedding(num_images, dim)`, per-image in train, average/zeros at eval), config field
`appearance_embedding_dim` (default 0 = prior behavior; set 32 to match static) + `use_average_appearance_embedding`;
runner flag `--appearance-embedding-dim`. Micro-tested finite in train+eval. H1/H2/H3 paths unchanged when dim=0.

**Queued confirmation runs** (`appearance_driver.sh`, after variance, waiter `br0vaykij`):
1. H2 winner (static log2=21/r4096) **+ appearance=32**.
2. same + **4D branch max_res=2048** (match static spatial res on dynamic branch too) + appearance=32.
Expect appearance embedding to lift metrics toward/past the static baseline.

---

## 2026-06-28 — ITERATION→QUALITY analysis (user directive) + under-reporting bug

**Target metrics (baseline_ngp_params.md, static single-frame 007740):** PSNR 24.42 / SSIM 0.640 /
LPIPS 0.460. (Our temporal task is harder — 51 frames — but H2 already BEATS the LPIPS target.)

**Eval curves (iteration → quality) — all monotonically improving, NO plateau by 45k:**
| run | LPIPS 15k→30k→45k | SSIM 15k→30k→45k |
|---|---|---|
| H1 base | 0.706→0.674→0.660 | 0.636→0.642→0.643 |
| H2 base | 0.515→0.500→0.491 | 0.687→0.693→0.695 |
| H2 s4096 | 0.457→0.442→**0.435** | 0.700→0.708→0.712 |
| H2 s8192 | 0.457→0.442→0.436 | 0.701→0.709→0.712 |
SSIM/LPIPS rise steadily with iterations; PSNR is noisier (±0.2). ⇒ **more iterations help.**

**⚠️ Under-reporting bug:** schedule evals every 15,188 but 4×15,188=60,752 > final step 60,751, so the
**4th eval never fires** — runner selected step-45,564 and never evaluated the saved `step-000060751.ckpt`
(15k more iters of monotonic improvement). All my reported numbers are from 45,564 = pessimistic.
→ Launched background eval of the true final (60,751) checkpoints (`eval60751.sh`).
→ Future/winner runs: will also eval the final checkpoint (fix off-by-one).

**Iteration study queued** (`iterstudy_driver.sh`, after appearance): H2 s4096 to **137k iters**, eval every
15,188 (9 points 15k→137k) to locate the quality plateau and pick the right iteration budget. waiter set next.

**True final-checkpoint (step-60,751) evals [DONE]** — confirms monotonic gain past the selected 45,564:
| run | 45,564 (was reported) | 60,751 (true final) |
|---|---|---|
| H2 s4096 | PSNR 20.97 / SSIM 0.712 / LPIPS 0.435 | **PSNR 21.05 / SSIM 0.7145 / LPIPS 0.432** |
| H2 s8192 | PSNR 21.04 / SSIM 0.712 / LPIPS 0.436 | PSNR 21.06 / SSIM 0.7145 / LPIPS 0.432 |
| H1 base | PSNR 20.31 / SSIM 0.643 / LPIPS 0.660 | PSNR 20.34 / SSIM 0.645 / LPIPS 0.649 |
⇒ s4096 ≈ s8192 at the finish (both LPIPS 0.432) → **pick static_max_res=4096** (simpler, tied). Always
eval the final checkpoint going forward. Iteration study will extend the curve past 60k.

---

## 2026-06-28 — Added significant_artifacts_score metric (user directive; target 0)

LookCloser's structural-artifact detector now tracked as an extra metric (no retraining of old runs).
New `LookCloser/scripts/score_artifacts_temporal.py` splits 2-panel eval renders (GT|pred), runs
`detect_defects(..., preset="significant")`, aggregates `serious_artifact_score` over the 63 eval images.
Wired into `run_temporal_ngp_quiet.py` (prints `artifacts significant_mean/max` + new SigArtifact column;
future runs auto-report).

**Baseline (winner H2 s4096, step-45564 renders):** significant_artifacts_score **mean 1.11 / max 7.46**,
**31/63 eval frames have significant artifacts** → NOT zero. This quantifies the artifact gap and is the
core motivation for Phase 4 (LookCloser port), whose feature-reweighting + ARM + FAS target exactly these
structural defects. Goal: drive significant_artifacts_score → 0 while keeping LPIPS≤~0.43.

---

## 2026-06-28 — Variance confirmed; appearance run live

**Variance (2nd seed) — top-2 tied within noise:**
| config | seed 42 LPIPS | seed 7 LPIPS |
|---|---|---|
| H2 s4096 | 0.4346 | 0.4332 |
| H2 s8192 | 0.4355 | 0.4346 |
Run-to-run σ ≈ 0.001–0.002 LPIPS ⇒ s4096 ≡ s8192. **Selected: H2, static_max_res=4096** (simpler, tied).

**Now running:** appearance run `H2 s4096 + appearance-embedding-dim=32` (auto-reports artifact score via
the edited runner). Then `+4D max_res=2048 +appear32`, then the 137k iteration study. Waiters br0vaykij / b3af2ffg7.

---

## 2026-06-28 — Occupancy warmup = artifact fix (user insight); motion-sampling research

**User practical insight:** non-zero artifact score was caused by too few occupancy-grid WARMUP iterations;
raising warmup removed artifacts (LookCloser static used `occupancy_warmup_steps=4096`; my temporal used the
nerfacc default **256**). Also: for video, init the 3D occupancy grid from ALL frames — already done (my
occ_eval unions over all 51 times every warmup update since 51 ≤ all-times-threshold 64). Fix = more warmup.

**Implemented:** `TemporalInstantNGPModelConfig.occ_warmup_steps` (**default 4096**) + `occ_update_n` (16),
passed to `OccGridEstimator.update_every_n_steps(warmup_steps=, n=)`. Runner flag `--occ-warmup-steps`.
Applies to ALL future runs incl. Phase 4. (Verified nerfacc accepts warmup_steps/n.)

**Queued warmup A/B test** (`warmup_test_driver.sh`, after the appearance sweep, waiter bopa6qsa5):
H2 s4096, 20k iters, eval every 5k, artifact auto-scored — warm256 (control) vs warm4096 (fix). Expect
warm4096 artifact ≈ 0 right after warmup (no need to train to the end). Baseline was 1.11 mean (warm256).

**Iteration study (137k) DEPRIORITIZED** for now (iteration→quality trend already shown monotonic; warmup
artifact fix is higher priority). Can re-run later.

**Appearance embedding result (H2 s4096 + appear=32):** best ckpt step 30,376 → PSNR 20.52 / SSIM 0.710 /
LPIPS 0.439, **artifacts mean 0.843 / max 7.125 / 29-of-63** (vs 1.11 / 31-of-63 without appearance).
⇒ appearance embedding modestly REDUCES artifacts; LPIPS ~flat (selection landed at 30k, not iso-iteration).
Auto artifact-scoring in the runner confirmed working. (run 2 = +4D r2048 +appear32 still training.)

## 2026-06-28 — ROOT CAUSE of bad dynamic frames: temporal-interpolation gap (stride mismatch)

Inspecting eval renders: off-grid eval frames reconstruct moving people badly (e.g. eval_img_0061 =
frame 7890 / t=0.95 → red fencer smears into a diffuse cloud), while on-grid frames (eval0000 t=0,
eval0031 t=0.5) are sharp. Cause: **train stride 4** (offsets 0,4,8,…) vs **eval stride 10** (0,10,20,…)
→ only offsets divisible by 20 coincide; **10 of 21 eval times are off-grid** (fall between trained
frames). Fast motion can't be interpolated by the 4D grid at unseen times → smear. Compounded by
max-over-time occupancy keeping moving-object cells "occupied" at all times → renderer paints faint
density at interpolated times. ⇒ the off-grid frames drag aggregate PSNR/LPIPS down and drive the
non-zero artifact score; on-grid frames are good.

**FIX (biggest quality lever): train denser — stride 2.** Eval times are all EVEN offsets; stride-2
training covers all even offsets → every eval time becomes a trained time → interpolation gap removed.
Now feasible: disk-streaming (ParallelDataManager load_from_disk) removed the RAM cap that forced stride-4,
and /fsx recovered (1.6T free). Cost: convert ~2,700 more frames (101 frames × 52 cams = 5,252; have 2,523)
~5–6h on 4 CPUs, then retrain winner (H2 s4096 + warmup 4096). PENDING user go-ahead.

## 2026-06-28 — Why even ON-grid eval frames are soft (eval0031, t=0.5, trained time)

Two causes (distinct from the off-grid smear):
1. **Held-out camera (novel view).** Eval cams (D/E/I004) never seen in training; only 52 train cams
   (lost 17 to /fsx) → sparse angular coverage near eval cams → inherently softer NVS. (Static baseline
   got 24.4 on its held-out cams; temporal ~21 — gap = 4D capacity split + fewer cams.)
2. **Moving people live in the LOW-RES 4D branch.** H2 static content = 3D branch @ max_res 4096 (sharp);
   dynamic content (people) = 4D branch @ **max_res 512** (never raised!). We swept the STATIC branch
   512→8192 but never the 4D/dynamic branch → people are the softest element. 

**Three complementary levers (each fixes a different symptom):**
| symptom | cause | fix |
|---|---|---|
| off-grid frames smear (eval0061) | train stride4 vs eval stride10 | stride-2 training |
| people soft on trained frames (eval0031) | 4D branch max_res=512 | **4D dynamic-branch res sweep (1024/2048, log2 20/21)** |
| general novel-view softness | only 52 train cams | backfill 17 cams (/fsx recovered) |
Caveat: raising 4D max_res also resolves TIME finer (51 timestamps) → test 1024/2048, not 8192.

## 2026-06-28 — PROTOCOL CHANGE (user): short screening runs, full run only for final pick

Going forward: **candidate selection uses SHORT runs** (15,000 iters, eval @5,000, artifact auto-scored;
~25 min/H2 run vs ~2.3h full). Justified by eval curves — ranking is stable well before 60k. The single
final winning combination gets a full 60,752-iter run at the end. (Warmup A/B at 20k already fits this.)

**Queued (short screen): 4D dynamic-branch resolution** (people-sharpness lever, never swept), static
branch fixed at log2=21/r4096, occ_warmup=4096 (new default): control l19/r512 vs l20/r1024 vs l21/r2048.
Driver `dynres_screen_driver.sh` (after warmup A/B), waiter `bs0stdriq`. Queue now: appearance run2 →
warmup A/B → dynres screen.

**Appearance run 2 (H2 s4096 + 4D max_res=2048 + appear32):** best ckpt 45,564 → PSNR 20.66 / SSIM 0.715 /
LPIPS **0.431** / artifacts 0.84 (32/63). Marginally best LPIPS/SSIM so far → hint that raising the 4D
DYNAMIC branch (512→2048) helps; dynres screen will isolate it cleanly (no appearance confound).
Warmup A/B now running (warm256 then warm4096).

## 2026-06-28 — WARMUP A/B RESULT: occupancy warmup is NOT the temporal artifact fix

Matched 15k-iter runs (H2 s4096):
| | PSNR | SSIM | LPIPS | SigArtifact mean | frames |
|---|---|---|---|---|---|
| warm256 | 20.66 | 0.701 | 0.459 | 1.089 | 29/63 |
| warm4096 | 20.91 | 0.702 | 0.457 | 0.999 | 31/63 |
Warmup 256→4096 nudged artifacts ~8% (1.09→1.00), did NOT zero them (+0.25 PSNR though). **Keep 4096 as
default (slight benefit, no downside), but it is NOT the artifact lever here.** The static-scene intuition
(warmup fills occupancy holes) doesn't transfer: temporal artifacts live in MOVING-PEOPLE regions, caused
by off-grid temporal interpolation (eval0061) + low-res 4D dynamic branch (eval0031) — occupancy warmup
can't fix dynamic interpolation. Real artifact levers = 4D-branch resolution (screening now) + stride-2
training + motion-aware sampling + LookCloser feature-reweighting/FAS/ARM.

## 2026-06-28 — 4D dynamic-branch resolution IS an artifact lever (short screen, 15k)

| 4D branch | PSNR | SSIM | LPIPS | SigArtifact mean/max | frames |
|---|---|---|---|---|---|
| l19/r512 (old) | 20.76 | 0.696 | 0.468 | 0.898 / 6.29 | 34/63 |
| l20/r1024 | 21.03 | 0.698 | 0.467 | 0.802 / 6.47 | 34/63 |
| **l21/r2048** | 20.82 | 0.700 | **0.463** | **0.586 / 4.06** | 34/63 |
Bigger 4D dynamic branch → lower artifact SEVERITY (0.90→0.59) + best LPIPS/SSIM (moving people were
under-resolved). Artifact COUNT stays 34/63 → dominated by OFF-GRID frames → needs stride-2.
**Adopt 4D = l21/r2048** for the winner config (static stays l21/r4096, warmup 4096).
Confirm screen queued: 4D l21/**r4096** (max_res ~free, bounded by log2). Stride-2 dataset prep delegated.

## 2026-06-28 — stride-2 dataset built (converting); r4096 confirm screen running

Stride-2 dataset at **`/opt/dlami/nvme/temporal_ngp_ds_s2/`** (separate from frozen stride-4 ds):
transforms.json = **5,252 train (52 cams × 101 stride-2 frames {7700,7702,…,7900}) + 63 eval**, images
symlinked to the shared dir (reuses stride-4 JPGs). Converting ~2,645 in-between frames (4 workers, ETA
~5–6h, waiter bqzrvydie). /fsx filling again externally (→316G) but we only read it + write local → insulated.
Once converted, SHORT-screen stride-2 vs stride-4 with the winner config (then full run on the best combo).
4D r4096 confirm screen running on GPU in parallel (waiter b8pbds9w4); CPU contention may slow it (~45min).

## 2026-06-28 — HARD GATE: no full 60k run until significant_artifacts_score = 0

User directive: keep ALL runs short (screens) until the artifact metric hits 0; only then do the full
60,752-iter run on the winning config. (No full run currently queued — GPU is on a 15k screen; stride-2 is
CPU conversion.) Spawned clean-context research agent (a6bedf54) to mine LookCloser/experiments/*.md +
*.json (esp. `arm_occwarmup_retest_summary.json`, `feature_reweighting_*_summary.json`,
`micro_artifact_scores/`) + architecture.md to extract the EXACT recipe that zeroed artifacts on the static
3D scene — what removed the last artifacts IN ADDITION TO warmup (likely ARM coarse step / transmittance
threshold / occupancy binary-warmup / feature reweighting / charbonnier) — then adapt to temporal.
Known so far on temporal: warmup 256→4096 only ~8%; 4D-branch res 0.90→0.59; remainder = moving-people +
off-grid frames (→ stride-2 + ARM/feature-reweighting from the LookCloser recipe).

## 2026-06-28 — ARTIFACT-ZEROING RECIPE (mined from LookCloser records) + metric nuance

**Metric nuance (critical):** on the static scene the FULL-FRAME significant_artifacts_score never cleanly
hit 0 (floated ~0.13–0.47 from floor/edge/equipment texture). What hit hard **0.000** was the **ROI /
structural-ROI** score (`artifact_roi_serious_score`, `stand_connector_score`). So "artifacts=0" = ROI-serious=0
at an artifact-selected checkpoint. ⇒ my full-frame ~0.59 isn't directly comparable; add ROI scoring +
artifact-aware checkpoint selection. (Cite: arm_occwarmup_retest_summary.json H40 full=0.28/ROI=0.0, H41 full=0.469/ROI=0.0.)

**Static recipe that zeroed (ROI) artifacts, ranked:**
1. **Budget-aware ARM** + `adaptive_coarse_step_size=0.00625`, `max_steps_per_ray=1024`,
   `transmittance_threshold→early_stop_eps=0.0` — THE decisive lever (3/3 seeds clean). **My temporal model
   has NO ARM (fixed-step) → this is the missing piece.**
2. occupancy `warmup_steps=4096` + `binary_warmup_steps=4096` (necessary, weak — matches my −8%).
3. **artifact-aware dense checkpoint selection** (`--eval-checkpoint roi/artifact`, keep all ckpts; clean
   window is NON-monotonic — can't just take final step).
4. raise hash/field capacity (matches my dynamic-branch 0.90→0.59).
NOT artifact levers (rejected in records): feature-reweighting (LPIPS only; ADDS ROI artifacts on unclean
field — layer last), FAS (LPIPS), occupancy conservativeness (dilation/low occ_thre/high ema_decay).

**Temporal adaptation priority:** (1) port budget-aware ARM (coarse 0.00625, cap 1024, eps 0.0);
(2) artifact-aware dense ckpt selection across all 63 eval times + ROI scoring; (3) keep raising dynamic
hash capacity; (4) motion sampling for moving-people residual; (5) off-grid-time supervision (densify →
stride-2 already prepping). FR/FAS only AFTER ARM is clean, as LPIPS levers. NO full 60k until ROI-artifact=0.
Spawning Explore agent to produce a port-ready ARM spec.

## 2026-06-28 — ARM IMPLEMENTED (Phase 4 begins); 4D r4096 adopted

4D dynamic-branch confirm: **l21/r4096 → artifacts 0.531 (29/63)**, slightly beats r2048 (0.586/34).
max_res free (bounded by log2) → **adopt 4D l21/r4096**.

**Ported budget-aware ARM** (the static recipe's decisive artifact lever) to the temporal model:
- New `nerfstudio/model_components/temporal_arm_sampler.py` (`TemporalARMSampler`): faithful port of
  LookCloser's FrequencyAwareVolumetricSampler with (a) per-ray `times` threaded into the sigma fn +
  `ray_samples.times`, (b) a pluggable/constant frequency level (`fallback_frequency_level`) so ARM runs
  WITHOUT the frequency-map pipeline. Coarse occ traversal → Nyquist fine dt=1/(2f) → budget-aware per-ray
  dt scaling (≤ max_steps_per_ray) → packed RaySamples.
- `TemporalInstantNGPModelConfig`: added `enable_adaptive_ray_marching` (default off), `adaptive_coarse_step_size`
  =0.00625, `max_steps_per_ray`=1024, `transmittance_threshold`=0.0, `adaptive_*`, `arm_*`, and
  `occ_binary_warmup_steps`=4096 (binary warmup: keep grid fully occupied first N steps).
- `get_outputs` overridden to use ARM when enabled (else parent fixed-step path). Runner flags `--enable-arm`,
  `--adaptive-coarse-step-size`, `--max-steps-per-ray`, `--transmittance-threshold`, `--occ-binary-warmup-steps`.
- Imports/config verified; H1/H2/H3 fixed-step paths unchanged when ARM off.

**First ARM screen running** (waiter b2j7agkvv): H2 + static l21/r4096 + 4D l21/r4096 + warmup/binary 4096 +
ARM(coarse 0.00625, budget 1024), 15k iters, real temporal data — tests time-threading through ARM + the
artifact effect. THIS is the lever expected to cut artifacts most. (Still gated: no full 60k until ROI-artifact≈0.)

## 2026-06-28 — ARM RESULT: did NOT help temporal artifacts (negative result) + pivot

| config (15k screen) | PSNR | SSIM | LPIPS | SigArtifact mean/frames |
|---|---|---|---|---|
| 4D r4096, no ARM | 21.09 | 0.700 | 0.463 | **0.531 / 29** |
| 4D r4096, +ARM (const freq) | 20.90 | 0.698 | 0.474 | 0.566 / 34 |
ARM (constant fallback freq level, no artifact-aware ckpt selection) slightly WORSENED LPIPS + artifacts.
Consistent with the static records' own note: residual artifacts are a FIELD/trajectory issue, not grid-miss
→ sampling-side levers (warmup, ARM) don't move the temporal metric; FIELD CAPACITY does (4D res 0.90→0.53).
ARM's static benefit likely needed the real frequency grid + dense artifact-aware checkpoint selection.

**Two course corrections:**
1. **Metric:** the static FULL-FRAME score never cleanly hit 0 either (floated 0.3–0.5); only the ROI metric
   reached 0. My 0.53 full-frame is likely near that floor → switch to a **motion-ROI artifact metric**
   (score restricted to the moving-people region) — the actually-zeroable target. TODO: add to scorer.
2. **Levers:** stop throwing sampling levers at it. Temporal artifacts = field/temporal capacity. Pursue:
   (a) **stride-2** (off-grid frames — the artifact COUNT driver; ~72% converted), (b) keep 4D capacity high,
   (c) motion-aware sampling for moving-people detail. ARM kept OFF by default (available, didn't help here).

**Current best config (NO ARM):** H2 / static l21·r4096 / 4D l21·r4096 / warmup+binary 4096 → LPIPS 0.463,
artifacts 0.531. Next: stride-2 screen on this config + ROI metric.

## 2026-06-29 — METRIC GROUNDED on static no-4D leader; stride-2 screen launched

User directive: "start from reproducing significant=0 for one frame without 4D." Scored the known-clean
static LookCloser ARM leader (007740, single frame, no 4D) with MY full-frame scorer →
**0.043 mean / 0.13 max / 1-of-3 frames** (records' ROI-serious=0 confirmed). ⇒ the static "0" = **~0.04 on
my metric**; full-frame IS a valid target, clean floor ≈ 0.04–0.13. My temporal best 0.53 mean = ~12× that
floor → genuine gap (moving-people + off-grid). NOTE: the static 0.04 required the FULL ARM recipe + 200k
iters + artifact-aware checkpoint selection — so iterations + ckpt-selection are also part of closing it.

**Stride-2 dataset conversion DONE** (5,252 train). Stride-2 covers all EVEN offsets → all 21 eval times
become TRAINED times → should kill the off-grid smear (the artifact-count driver). Screening best non-ARM
config (H2 / static l21·r4096 / 4D l21·r4096 / warmup+binary 4096) on `temporal_ngp_ds_s2`, 15k, vs the 0.53
stride-4 baseline. Target → ~0.04.

## 2026-06-29 — eval-every-12th plan; ground artifact-0 on static (2 agents); insight rule saved

User course-corrections:
- **Eval every 12th instead of 10th** (12=3×4 → all eval offsets land on stride-4 trained times → no
  off-grid gap, NO extra training cost, no need for stride-2). Plan: after the stride-2 15k screen finishes
  (for comparison), switch eval split to every-12th on the stride-4 dataset (convert eval-cam imgs at
  frames {7700,7712,…,7892}, regen transforms). This is the cheap version of choice (A).
- **stride-2 likely too slow** for ongoing experiments → prefer eval-12th. Still finishing the stride-2 15k
  screen to draw the comparison.
- **Ground artifact≈0 on ONE static frame first** (the 007740 LookCloser training frame). Two SEPARATE
  agents (keep main context clean):
  • research agent (a83c729e): mine experiment files for warmup + total iters empirically needed for
    artifact≈0 + stability.
  • GPU agent (a3148933): actually reproduce artifact≈0 on static 007740, tied to warmup/iters; coordinates
    GPU (waits for the running screen to finish). Reference: clean static leader = 0.043 on our scorer.
- Saved memory rule [[insight-generation-rule]]: generalize trends, propose threshold experiments, require
  STABLE target (≥85%, not variance flukes), split fast-ongoing vs final-max-quality param values.

## 2026-06-29 — Iterations-for-artifact-0 (from records); insight

Research agent mined the static records:
- **Metric:** ROI-serious=0 is the zeroable target; full-frame `significant` never stably 0 (floats
  0.13–0.47; our scorer read the clean leader at 0.043 mean / 0.13 max).
- **Warmup:** canonical 4096/4096 (warmup_steps + binary). Necessary but WEAK — the static→0 came from
  budget-aware ARM, not warmup (matches our temporal A/B: 256→1.09, 4096→1.00, never 0).
- **Iterations:** ROI-serious=0 appears EARLY + intermittently (seed42 step **8192**; fast staged recipe
  ROI-serious=0 by **30k**, ~29 min). Stable high-quality ROI=0 (PSNR≥29.86) needed the multi-stage leader
  ~**91k–106k** steps. Clean window is NON-monotonic + seed-dependent → MUST keep all ckpts + scan
  (`--eval-checkpoint roi`). "3/3 seeds clean" only with full budget-aware ARM (coarse 0.00625, cap 1024).
- **Fast vs final (per insight rule):** fast ROI≈0 = ARM + warmup 4096/4096 + ~16–30k iters + ckpt-scan,
  FAS/FR OFF. Final/stable = full leader recipe (ARM+charbonnier+distortion, max_res 8192, 75k–106k iters,
  FR/FAS as last LPIPS layers, artifact-aware ckpt selection).

**INSIGHT (generalized):** occupancy-warmup↑ lowers artifact score monotonically but plateaus >0 (256→1.09,
4096→1.00) on BOTH static and temporal → warmup is NOT the zeroing lever. The zeroing lever on static is
**budget-aware ARM + artifact-aware checkpoint scanning over enough iters (~8–30k for intermittent ROI=0)**.
Our temporal ARM with a CONSTANT freq level didn't help (0.531→0.566) → ARM likely needs the REAL frequency
grid to zero artifacts, OR the temporal residual is the moving-people ROI (needs motion + 4D capacity).
Next experiments must (1) scan checkpoints not take final, (2) measure ROI not just full-frame, (3) test ARM
WITH a real frequency signal. GPU agent reproducing static→0 now (a3148933).

## 2026-06-29 — STRIDE-2 RESULT: eval-time coverage is a HUGE quality lever (+3.4 dB)

stride-2 screen (15k, step-10000, same H2/static l21·r4096/4D l21·r4096 config):
**PSNR 24.54 / SSIM 0.743 / LPIPS 0.412** vs stride-4 **21.09 / 0.700 / 0.463**. **+3.4 dB PSNR**, and now
**BEATS the static single-frame baseline (24.42)** on a temporal eval (artifact pending).

**INSIGHT (generalized):** the dominant quality killer was OFF-GRID eval times (temporal extrapolation),
NOT model capacity — covering the eval timestamps (stride-2, or equivalently eval-every-12th on stride-4)
recovers +3.4 dB. ⇒ **switch eval to every-12th on the cheap stride-4 dataset** (12=3×4 → all eval offsets
trained, no extra training cost) — gets the stride-2 benefit without the ~2× training cost. stride-2 itself
is too slow for ongoing experiments (confirmed). This reframes the prior "temporal artifacts" largely as an
eval-coverage artifact for the interpolation-eval setup we'd chosen.

**Next:** (1) implement eval-every-12th split (convert eval-cam frames {7700,7712,…,7892}, regen transforms);
(2) re-baseline the winner on it; (3) static grounding run (a3148933) in progress on GPU for artifact≈0.

## 2026-06-29 — stride-2 ARTIFACT score: 0.53 → 0.17 (eval-coverage cuts artifacts 3×)

stride-2 screen artifact (step-10k, 63 eval): **significant_mean 0.169 / max 1.93 / 26-of-63** vs stride-4
0.531/29. ⇒ eval-time coverage cuts the artifact score ~3× AND lifts PSNR +3.4 dB. Remaining 0.17 (vs static
clean floor ~0.04) = genuine moving-people softness on held-out eval cams at trained times → address with
more iters + 4D capacity + motion sampling. **eval-12th will give this same artifact/quality benefit at
stride-4 cost** (re-baseline pending GPU). Updated scorecard of artifact levers:
| lever | artifact mean | note |
|---|---|---|
| stride-4 baseline (warm256) | 1.09 | |
| + occ warmup 4096 | 1.00 | weak |
| + 4D r4096 capacity | 0.53 | real |
| + eval-time coverage (stride-2 ≈ eval-12th) | **0.17** | biggest single drop |
| static clean floor (target) | ~0.04 | |

## 2026-06-29 — eval-12th dataset READY + verified; re-baseline queued

`/opt/dlami/nvme/temporal_ngp_ds_eval12/` built: 2,523 stride-4 train (verbatim) + 51 eval (3 cams ×
every-12th {7700,7712,…,7892}). Dataparser check: **all 17 eval times ⊂ 51 train times (full coverage)** —
novel-view-at-trained-times at stride-4 cost. This is the adopted eval going forward (replaces the off-grid
every-10th). Re-baseline of the winner (H2/static l21·r4096/4D l21·r4096/warmup 4096, NO ARM) queued on
eval-12th (15k screen, `eval12_rebaseline_driver.sh` auto-launches when GPU idle after the grounding run;
waiter b7kzh6fmk). Expect ~24 dB / ~0.17 artifact (like stride-2) at stride-4 cost.

## 2026-06-29 — EVAL-12th is the new best (cheap) + iteration-map launched

eval-12th re-baseline (winner config, 15k, step-10k): **PSNR 25.70 / SSIM 0.750 / LPIPS 0.400 /
artifact 0.136 (17-of-51)** — beats stride-2 (24.54/0.169) AND the static baseline (24.42), at stride-4 cost.
**Adopt eval-12th as the standard eval.** Artifact 0.136 @ 10k, trending toward the ~0.04 floor.
Updated artifact-lever scorecard: 1.09 (base) → 1.00 (warmup) → 0.53 (4D r4096) → **0.136 (eval-12th @10k)** → ~0.04 (target).

Per insight rule (find the iterations where artifact→floor): launched eval-12th **30k** run (ckpts every 5k,
waiter bxz3xipt8) to map artifact-vs-iterations. Grounding run (static) finished — rendered ckpts 5k–50k;
its agent will deliver the static artifact-vs-iter table (iters needed for ~0.04 on one frame).
Still gated: full 60k only once we confirm artifact reaches ~floor.

## 2026-06-29 — 🎯 ARTIFACT ≈ 0 ACHIEVED on temporal (gate satisfied)

eval-12th 30k, best ckpt step 25k: **PSNR 25.93 / SSIM 0.769 / LPIPS 0.367 / artifact 0.0061 (1-of-51)** —
below the static clean floor (0.043) ⇒ effectively 0. Quality climbs monotonically (PSNR 25.1→25.9 over
5k→25k, LPIPS 0.43→0.37).

**INSIGHT (recipe that zeroes temporal artifacts):** eval-time COVERAGE (eval-12th) + 4D dynamic-branch
CAPACITY (l21/r4096) + ~20–25k ITERATIONS. NOT warmup (weak) and NOT ARM (didn't help with const freq).
Artifact-vs-iters: 0.136@10k → 0.006@25k. **Fast experiments:** ~15–20k iters. **Final max-quality:** full
60k+ (LPIPS still dropping at 25k). TODO per insight rule: confirm stability (2nd seed) that artifact stays ≈0.

Gate (no full 60k until artifact≈0) now SATISFIED → launching the full-quality run on eval-12th winner.

## 2026-06-29 — directives: train-until-eval-plateau; motion sampling AS PART OF FAS

- **Long run: do NOT stop while eval still improves** (extend iterations past 60752 if needed). Current
  full-quality run going; eval was still climbing at 25k on the 30k run. Plan: after it finishes, inspect
  eval curve (15188/30376/45564) — if still improving, CONTINUE from the 60751 ckpt with more iters until
  eval plateaus (also fixes the off-by-one where the 60752 eval never fired). Waiter bhia3ban7.
- **Motion sampling integrated INTO FAS** (3-way split uniform/freq/motion) — plan written in
  `motion_sampling_plan.md`. Prereqs: IST maps (computing), per-camera FREQ maps (preprocess after GPU
  frees, 52 maps reused across frames), combined sampler. Run split-ratio/α screens in parallel (light).
- **Parallelism:** winner config = 35.6GB VRAM (solo); light screens (~13–18GB) run 2–3 in parallel.

## 2026-06-29 — Full run done (artifact≈0 holds); CONTINUING (eval still improving)

Full eval-12th run (step 45564, since 60752 eval off-by-one): **PSNR 26.10 / SSIM 0.779 / LPIPS 0.351 /
artifact 0.0094 (2/51)**. Eval curve still climbing (SSIM 0.760→0.772→0.779; LPIPS 0.382→0.362→0.351) →
per directive (don't stop while improving), **CONTINUING from step-60751 ckpt with --stop-on-no-improve,
cap 150k** (waiter be2wx6k8b). Artifact stays ≈0 throughout. Best temporal result so far far exceeds the
static baseline (24.42/0.640/0.460) AND keeps artifacts ≈ the clean floor.

## 2026-06-29 — ✅ FINAL CONVERGED TEMPORAL RESULT (core deliverable)

Continuation trained to plateau (stop-on-no-improve at 91128), best ckpt **step 75940**:
**PSNR 26.21 / SSIM 0.786 / LPIPS 0.341 / artifact 0.0039 (1-of-51)**.
Curve: 60752 (26.08/0.784/0.345) → 75940 (26.21/0.786/0.341) → 91128 (26.10/0.787/0.339) → plateau.

**Beats the static single-frame baseline on ALL metrics, with artifacts ≈ clean floor, on the full 51-frame clip:**
| | PSNR | SSIM | LPIPS | artifact |
|---|---|---|---|---|
| static baseline (1 frame) | 24.42 | 0.640 | 0.460 | ~0.04 |
| temporal winner (51 frames) | **26.21** | **0.786** | **0.341** | **0.004** |

**Winning recipe:** `instant-ngp-time` H2 (concat 3D static + 4D dynamic hash), static branch log2=21/max_res=4096,
4D branch log2=21/max_res=4096, occ warmup+binary 4096, NO ARM, **eval-every-12th (full eval-time coverage)**,
disk-streamed, ~76k iters (train-to-plateau). Dataset `/opt/dlami/nvme/temporal_ngp_ds_eval12` (52 cams ×
stride-4 train, 3 held-out cams × every-12th eval). Run dir: `.../EVAL12_CONTINUE_52cam/.../`.

**LEADER RENDERS (view_000 + folder):** `eval_img_0000.png` (GT|pred 2-panel) at
`/opt/dlami/nvme/temporal_runs/007700_007900_temporal_h2_4dr4096_EVAL12_CONTINUE_52cam/instant-ngp-time/20260629_154233/renders_best_step-000075940/`
(CONVENTION from here on: every new leader records the full path to its renders_best_* folder here.)

ENHANCEMENT NEXT (motion-as-part-of-FAS): FAS freq-map prep (starting now, GPU free) + person masks (CPU,
~4h) → combined sampler (Variant A 3-way split, Variant B region-gated FAS 80/20) → screens.

**Motion-sampling research DONE** → `motion_sampling_plan.md`. DyNeRF IST signal (static cameras → clean
temporal differencing), hard (1-α)·uniform+α·motion (α≈0.3, sweep 0.2/0.3/0.4), schedule warmup→ramp, fuse
with frequency via 3-way split (never product). Implement during/after Phase 4 (needs VanillaDataManager).

**Motion-aware sampling — research agent spawned** (clean context): survey DyNeRF ISG/IST and similar
motion+uniform importance sampling; cameras are STATIC so per-camera temporal differencing gives motion.
Plan: sample `(1-α)·uniform + α·motion`, sweep α (80/20, 70/30, 60/40); later fuse motion WITH the
LookCloser frequency-aware sampler. Implementation deferred to after current tests / within Phase 4.

**Note:** hit the pkill self-match bug again (a heredoc containing `*_driver.sh` made `pkill -f` kill its own
shell). No damage (driver file wasn't written, no collision). Lesson reinforced: manage drivers by PID, and
don't put kill-pattern strings in the same shell that runs pkill.

**H1 hash-sweep insight:** LPIPS↓ ranking: **h1c(log2=21) 0.645 < h1d(r1024) 0.650 < h1_base(log2=20) 0.660 < h1a(log2=19) 0.674.** Bigger 4D hash table clearly helps (collision pressure); log2=21 best on BOTH LPIPS and SSIM. max_res=1024 helps PSNR (best, 20.50) but not LPIPS. **Promising follow-up:** combine log2=21 + max_res=1024 (both helpful axes) as a final "best candidate" confirmation, + a 2nd seed of the top-2 for variance (per protocol).

**User insight (2026-06-27):** max_res 512/1024 is likely far too low — static single-frame NGP benefited
greatly from max_res up to 8192. H1-D (1024) already gave best PSNR, so the spatial-res axis isn't
saturated. KEY: raising max_res is ~free (tcnn hash memory is bounded by log2_hashmap_size, not max_res;
only per_level_scale changes). Nuance: tcnn uses one per_level_scale for all 4 dims, so high max_res also
resolves TIME to 8192 (we only have 51 train times) → very-fine temporal levels lean on coarser levels for
off-grid interpolation; spatial detail should dominate but watch eval. If time over-resolution hurts →
anisotropic res via time-input scaling (follow-up).
**Queued max_res sweep** (`maxres_driver.sh`, after H2/H3): log2=21 × max_res ∈ {2048, 4096, 8192},
levels=16. Will extend to 16384 if 8192 still improving.

**VRAM note:** H2 (two hash grids: 3D + 4D) uses ~30 GB vs ~11 GB for H1 — fine on 46 GB L40S.

**Sweep fully automated:** driver1 (`h1_sweep_driver.sh`) runs H1-A→H1-C→H1-D; driver2
(`h23_driver.sh`) waits then runs H2→H3. Final waiter `b5zr1bqwf` fires on `SWEEP_H23_DONE`. ~88m/run ⇒
remaining 5 runs ≈ 7.5h.

**H2/H3 implemented & micro-tested (finite density/rgb/density_fn):**
- H2: `concat(HashEncoding(in_dim=3), HashEncoding(in_dim=4))` → MLP. Static 3D branch params
  `static_{num_levels,max_res,log2_hashmap_size}` (defaults 16/512/19).
- H3: `concat(HashEncoding(in_dim=3 features), scalar t)` → MLP (cheap control, no 4D hash).
- H1 path kept byte-identical (TemporalHashMLP) for fair comparison. New config field
  `--pipeline.model.hypothesis {H1,H2,H3}`; runner `--hypothesis` re-enabled.

## 2026-06-30 — Decomp (Variant C) vs winner: clarified nuances + person-oversampling experiment

**Q&A clarifications (logged for the record):**
- **Variant C (H2D decomposition) vs winner (H2 concat) WITH sparsity loss OFF:** still a *different
  architecture*, not the same model. Winner H2 **concatenates** features `[enc3(x), enc4(x,t)]` and feeds
  ONE shared MLP → one density + one color head (grids merged early, at the feature level; the MLP learns
  the combination). Variant C uses **two separate MLPs**: static MLP(enc3)→σ_s,c_s and dynamic
  MLP(enc4)→σ_d,c_d, merged late at the OUTPUT: density `σ=σ_s+σ_d`, color density-weighted blend
  `c=(σ_s·c_s+σ_d·c_d)/(σ_s+σ_d+ε)`. So loss-off C (24.49 @5k) still ≠ winner (25.14 @5k): the additive
  two-branch structure is slightly less expressive/efficient here than the joint concat-MLP.
- **Merge mechanism:** concat → merge at FEATURES (one joint MLP); decomposition → merge at PHYSICAL OUTPUT
  (densities add, colors alpha-blend by density share). Grids themselves never mix in C.
- **Training speed leader vs C:** leader FASTER. C runs two MLPs + two color heads + an extra per-ray
  dynamic-accumulation pass → more compute/sample (measured C ≈ 35k rays/s). Both full-speed (tens of k
  rays/s), unlike the broken VanillaDataManager FAS path (178 rays/s).
- **Decomp collapse cause (confirmed):** `dynamic_sparsity_mult=0.05` was too strong → PSNR 14, artifact 2.49.
  dyn0 diagnostic (loss weight 0) trains fine (24.49/0.721/0.456 @5k) → architecture sound, loss weight was
  the killer. Tuned retry would use ~0.001.

**NEW EXPERIMENT — person-oversampling on the WINNER's fast streaming path (no FAS, no frequency).**
Implements the user's idea: "sample person pixels more often + uniform". Done CORRECTLY on the streaming
setup (ParallelDataManager, load_from_disk) so there is NO VanillaDataManager RAM/setup penalty that hurt
the earlier FAS+motion runs.
- New files: `nerfstudio/data/person_weighted_pixel_sampler.py` (PersonWeightedPixelSampler:
  `person_frac` of rays from inside YOLO person masks, rest uniform; vectorized — per-batch person-cell
  pool + two `torch.randint`s, no per-ray loops, runs in dataloader workers),
  `nerfstudio/data/datamanagers/person_stream_datamanager.py` (PersonStreamDataManager: wires the sampler
  + mask path + per-image stems into the RayBatchStream before workers spawn).
- New method `instant-ngp-time-personsample` = winner config + person sampler. CLI:
  `--ps person-frac=0.3`. Masks: `/opt/dlami/nvme/temporal_ngp_ds_eval12/person_masks/person_masks.pt`
  (2574 masks, 270×480 float16, ~8.9% person area per frame).
- **Screen (25k iters, identical winner config, seed 42):** baseline uniform (leader method) +
  person_frac ∈ {0.3, 0.4, 0.2}, sequential (35GB VRAM → no parallel). Driver
  `LookCloser/scripts/person_sample_screen.sh`. Compare PSNR/SSIM/LPIPS/artifact + rays/sec.
- **Expectation:** throughput ≈ unchanged (ray count/iter constant 8192; person sampling is cheap CPU-side)
  — NOT faster (clarifies the user's hypothesis). Convergence/metric effect is what we measure.
- **Decision rule (user):** if notably better → keep + tune optimal person_frac. If not better → discard,
  return to leader config and port the proven LookCloser recipe to push 4D toward the static PSNR≈29 target.

## 2026-07-01 — Person-oversampling screen RESULT: no gain → DISCARDED

Screen done (25k iters, winner config, seed 42; runner selects best-eval-loss ckpt = step 15k for all).
| run | PSNR | SSIM | LPIPS | artifact_mean | n_artifacts |
|---|---|---|---|---|---|
| baseline uniform (leader) | **26.013** | 0.7595 | 0.3819 | **0.032** | 8/51 |
| person_frac 0.3 (70/30) | 25.951 | 0.7600 | 0.3812 | 0.048 | 9/51 |
| person_frac 0.4 (60/40) | 26.022 | 0.7597 | 0.3815 | 0.049 | 10/51 |
| person_frac 0.2 (80/20) | 26.012 | 0.7599 | 0.3830 | 0.035 | 9/51 |

**Verdict:** person-oversampling gives NO improvement — all metrics within run-to-run noise (PSNR ±0.07,
SSIM ±0.0005, LPIPS ±0.002) and artifacts marginally WORSE, not better. **Speed unchanged** (wall-clock
87–89 min for all four, same 25k iters → person sampling adds negligible overhead; it does NOT speed up
training, as predicted — ray count/iter is constant). Root cause: at eval-12th (full temporal coverage)
the winner already reaches artifact≈0, so the person region is not the bottleneck; biasing samples there
just steals rays from the (already-clean) background.
**INSIGHT (generalizes):** sample-location reweighting (person/motion oversampling) only helps when a
region is *under-fit*; once eval-time coverage + 4D capacity already zero the artifacts, importance
sampling has no headroom. Matches the earlier FAS/decomp finding (no headroom over the ~0.006 floor).
**DECISION:** discard person-sampling. Code kept (method `instant-ngp-time-personsample`) but not adopted.
→ Next: port the proven single-frame LookCloser recipe (ARM + Charbonnier + distortion loss, higher
max_res, longer iters) onto the 4D winner to push PSNR from ~26.2 toward the static target ~29.

## 2026-07-01 — LookCloser recipe port to 4D: Charbonnier+distortion FAILS, capacity HELPS

Ported the static PSNR-29.4 leader knobs onto the 4D winner (added Charbonnier RGB + Mip-NeRF360
packed distortion loss to `temporal_instant_ngp.py`). Ablation ladder, 40k iters, selected step-32000:
| run | PSNR | SSIM | LPIPS | artifact_mean | n |
|---|---|---|---|---|---|
| baseline40k (MSE, winner cfg) | **25.99** | 0.7735 | 0.3615 | **0.023** | 6/51 |
| +Charbonnier+distortion 0.01 | 24.80 | 0.7741 | 0.3653 | 0.222 | 24/51 |
| +Charb+dist +static l23/r8192 | 25.29 | **0.7794** | **0.3543** | 0.204 | 18/51 |
| +Charb+dist +cap +ARM | 25.46 | 0.7733 | 0.3642 | 0.190 | 21/51 |

**FINDINGS:**
- **Charbonnier HURTS the 4D model:** lower PSNR (24.8 vs 26.0) + ~10× worse artifacts (0.22 vs 0.023,
  24/51 vs 6/51). Charbonnier is L1-like → doesn't minimize MSE → lower PSNR; and it tolerates floaters
  in the 4D dynamic branch. The static leader's gain does NOT transfer.
- **Distortion port was mis-scaled (caveat):** used global far=1000 for spacing normalization → normalized
  spacing collapses to ~[0,0.005] → distortion loss ≈ 0 (never truly tested). Proper per-ray near/far
  (from occupancy AABB) would be needed to actually exercise it. Deferred — low priority given Charbonnier
  already regresses artifacts.
- **Capacity HELPS (the one useful lever):** static 3D branch l23/r8192 gave best SSIM (0.779) & LPIPS
  (0.354). The artifact regression in those rows is from Charbonnier, NOT capacity (MSE baseline stayed
  clean). **INSIGHT:** on the 4D model the quality lever is spatial hash CAPACITY, not the loss; keep MSE.
- **ARM (const-freq fallback):** no benefit (temporal has no frequency grid), as predicted.

**NEXT:** isolate capacity with plain MSE — winner + static l23/r8192, no Charbonnier/distortion (run
`RECIPE_mse_cap`). Expect clean artifacts (~0.02) + the SSIM/LPIPS gain. If it beats the winner, extend to
plateau as the new leader.

## 2026-07-01 — Capacity + MSE = new direction (cleanest artifacts + best LPIPS)

Isolated the capacity lever with plain MSE (no Charbonnier/distortion), 40k, selected step-32000:
| run | PSNR | SSIM | LPIPS | artifact_mean | n |
|---|---|---|---|---|---|
| baseline40k (winner l21/r4096) | 25.99 | 0.7735 | 0.3615 | 0.023 | 6/51 |
| **mse_cap** (static l23/r8192, 4D l21/r4096) | 25.64 | 0.7793 | **0.3458** | **0.0085** | **2/51** |
| mse_cap4d (+4D l22) | 25.45 | 0.7801 | 0.3442 | 0.033 | 6/51 |

**mse_cap (static 3D capacity l23/r8192 + MSE) is the best variant:** best LPIPS (0.346), cleanest
artifacts (0.0085, 2/51 — below the static clean floor 0.043), SSIM 0.779 — all while eval loss was STILL
IMPROVING at 40k (not plateaued). SSIM/LPIPS at 32k already near the original winner PLATEAU (0.786/0.341).
Bumping the 4D branch (mse_cap4d, log2=22) slightly RAISED artifacts (0.033) and lowered PSNR → keep 4D at
l21/r4096; the useful capacity is in the STATIC 3D branch.
**INSIGHT (confirmed):** the transferable LookCloser lever for the 4D model is spatial hash CAPACITY on the
STATIC branch (log2 23 / max_res 8192), NOT the loss changes (Charbonnier/distortion regress) and NOT ARM.
→ Continuing mse_cap from step-39999 to plateau (stop-on-no-improve, cap 120k) as the new-leader candidate.

## 2026-07-01 — 3D winner RITUAL extracted: multi-stage + LR-reset warm restart

Per user directive, replicating the static leader's "magic" (PSNR 29.5→29.86 via checkpoint continuations):
- **Multi-stage:** static leader = 3 short pre-stages (~20k/35k/50k) each loading the previous checkpoint,
  then a long stage-2 (200k, FR on, Charbonnier) — reached 29.86 @ step 106316.
- **LR-RESET mechanism (verified in code, trainer.py:437,442,455):** `self._start_step = ckpt_step+1`
  ALWAYS (global step resumes), BUT the scheduler LR is reset iff the scheduler state is NOT loaded.
  `--load-scheduler False` → fresh scheduler (last_epoch from 0) → **LR jumps back to ~initial 0.01**,
  giving a high-LR "fresh phase". `--load-dir` defaults load_scheduler=True → LR stays decayed (what my
  first continuation did → LR 0.004→0.0028, PSNR trailing). Runner now exposes `--load-scheduler` /
  `--load-optimizers` passthrough.
- **Checkpoint interpolation** (`interpolate_lookcloser_checkpoints.py`, lerp of hash+MLP params):
  EXPERIMENTAL, the 29.86 leader did NOT use it. Keep in reserve.
- **Seeds:** static used 42; 3-seed ARM spread ~±0.1–0.2 dB PSNR (seed 43 hit ROI-artifact 0). → vary seed
  per fresh phase and log it (per user).
- **ARM in static:** helped LONG-term (29.86 vs 29.57 baseline) but ONLY with the frequency grid + multi-
  stage; ARM-from-scratch is fragile and short-term WORSE. Temporal has NO frequency grid → expect little
  ARM benefit; will test cleanly on MSE+cap.

**mse_cap continuation (load_scheduler=True, decayed LR) @56k, STILL improving:** PSNR 25.70 / SSIM 0.7863 /
LPIPS 0.331 — already **beats the winner on SSIM & LPIPS** (0.786/0.341); PSNR trails (25.70 vs 26.21).
PLAN: let it plateau, then **warm-restart phase** from its best ckpt with `--load-scheduler False` (LR reset)
+ seed 43 → test if the ritual lifts PSNR (as in 3D). Then a clean ARM test on MSE+cap.

## 2026-07-01 — ARM with REAL frequency grid (baked) + LR-reset warm restart [RUNNING]

Disk-full (100%) crashed the mse_cap continuation at ~step64000 (that ckpt truncated/corrupt); freed 80GB by
deleting scored/discarded runs (charbdist*, PERSAMP*, mse_cap4d, baseline40k, old decomp/30k). Last GOOD ckpt
= **step-56000** (eval PSNR 25.70 / SSIM 0.7863 / LPIPS 0.331 — already beats winner on SSIM/LPIPS).

**Baked a real 3D frequency grid** (`bake_frequency_grid.py`) from step-56000: rendered depth over all 2523
train images × 20k px, combined with the existing 2D freq maps (`f_3d=f_2d·focal/depth → level`, scatter-max
union). Result `freq_grid_mse_cap.pt`: 128³, 88,269 non-empty voxels (4.21%), levels [0,15]. This is the REAL
per-scene frequency signal ARM was missing (previously constant fallback → no adaptation → no benefit).

**Leader-stage warm restart LAUNCHED** (mirrors static 1c→2): from step-56000, `--load-scheduler False`
(**LR reset confirmed: 0.0028→0.00999 ≈ initial**), **seed 43** (varied per user), **ARM ON** fed the baked
grid (adaptive_coarse 0.0125, max_steps/ray 1024, max_freq_level 12, midpoint), static l23/r8192 + 4D l21/r4096,
MSE, cap 120k, stop-on-no-improve. Model log confirms: "loaded baked frequency grid (128³, 88269 voxels) -> ARM".
Next: (a) no-ARM LR-reset control (seed 43) to isolate ARM's contribution; (b) if ARM wins → feature-reweighting
(feasible for H2); (c) eval-camera time-sequence render + view_000 for the new leader.

## 2026-07-01 — ARM+real-frequency RESULT: perceptual win (SSIM/LPIPS↑), PSNR↓ (as in paper)

ARM+freq warm restart done (stop-on-no-improve @96k, best step-88000):
| config | PSNR | SSIM | LPIPS | artifact_mean | n |
|---|---|---|---|---|---|
| original winner (l21/r4096) | 26.21 | 0.786 | 0.341 | 0.004 |
| mse_cap base (step-56000) | 25.70 | 0.786 | 0.331 | ~0 |
| **ARM+freq + LR-reset (seed43)** | 25.23 | **0.794** | **0.317** | 0.043 | 7/51 |

**ARM WORKS in 4D with real per-scene frequency maps** (no bug — user was right): SSIM 0.786→0.794 and
LPIPS 0.331→0.317 are BOTH best-ever, matching the paper's perceptual-quality claim. PSNR regressed
(25.70→25.23) — the known ARM/frequency vs MSE tradeoff; the static leader recovered PSNR via FEATURE
REWEIGHTING + long stage-2 (29.5→29.86). Artifacts 0.043 (≈ static clean floor, fine).
Confound: this run changed LR-reset + ARM + seed together. NEXT: (1) no-ARM LR-reset control (seed 43) to
attribute the SSIM/LPIPS gain to ARM vs the warm restart; (2) add feature-reweighting (feasible for H2) on
top of ARM to try to recover PSNR while keeping LPIPS/SSIM. Renders (single-view) at
`.../RECIPE_mse_cap_ARMfreq/instant-ngp-time/20260701_154553/renders_best_step-000088000`.

## 2026-07-01 — Control isolates ARM: perceptual gain is ARM's; PSNR drop is the warm-restart's

No-ARM LR-reset control (seed 43, from step-56000) vs ARM+freq, both warm restarts:
| config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| base step-56000 (converged, LR 0.0028) | 25.70 | 0.786 | 0.331 |
| control: LR-reset, NO ARM @96k | 25.21 | 0.789 | 0.328 |
| ARM+freq: LR-reset + ARM @88k | 25.23 | **0.794** | **0.317** |

**Attribution:** ARM's specific effect (over the warm restart) = SSIM +0.005, LPIPS −0.011 — a real perceptual
gain, matching the paper. The PSNR drop 25.70→25.2 is from the LR-RESET WARM RESTART itself (both control &
ARM land ~25.2), NOT from ARM. Unlike the static single-frame (LR reset LIFTED PSNR 29.5→29.86), on this
already-converged high-capacity 4D base the reset trades PSNR for perceptual quality.
**Current bests:** PSNR → original winner 26.21 (SSIM 0.786/LPIPS 0.341); perceptual → ARM+freq
(SSIM 0.794/LPIPS 0.317, PSNR 25.23). Tradeoff. NEXT: ARM+FR (feature reweighting = static leader's PSNR
lever) to try to keep ARM's LPIPS/SSIM while recovering PSNR.

## 2026-07-02 — ARM+FR CONVERGED (best perceptual + zero artifacts); ARM-vs-noARM verdict

ARM+FR trained to full convergence (step-112000) — earlier "wash" call was premature; FR helps at length:
| config | PSNR | SSIM | LPIPS | artifact_mean | n |
|---|---|---|---|---|---|
| no-ARM winner (l21/r4096) | **26.21** | 0.786 | 0.341 | 0.004 | 1/51 |
| no-ARM + capacity (l23/r8192, step-56000) | 25.70 | 0.786 | 0.331 | ~0 | - |
| ARM + freq grid (step-88000) | 25.23 | 0.794 | 0.317 | 0.043 | 7/51 |
| **ARM + freq + feature-reweight (step-112000)** | 25.40 | **0.800** | **0.308** | **0.000** | **0/51** |

**VERDICT (ARM vs no-ARM):** ARM (esp. ARM+FR) WINS on SSIM/LPIPS/artifacts; no-ARM wins only on PSNR.
ARM trades ~0.8 dB PSNR for best-ever SSIM 0.800, LPIPS 0.308, and a PERFECT 0/51 artifact score. FR (feature
reweighting on enc3/enc4) added the final push (LPIPS 0.317→0.308, SSIM 0.794→0.800, artifacts 0.043→0.0) —
matching the static leader pattern where FR was decisive. ARM confirmed to work in 4D with real baked
frequency maps (user was right; no bug).
**ARM+FR renders (all 51 eval views):** `/opt/dlami/nvme/temporal_runs/RECIPE_mse_cap_ARMfreq_FR/instant-ngp-time/20260702_040131/renders_best_step-000112000`
**CAVEAT:** all metrics are on the DULL-graded dataset (mean 0.40 / highlights clipped 0.84 vs static 0.53/0.99);
absolute numbers will shift after the color-grade rebuild. Leader-render video (no-ARM winner, cam0 time seq)
delivered. Pending: re-grade EXR→JPG to match static, retrain.
