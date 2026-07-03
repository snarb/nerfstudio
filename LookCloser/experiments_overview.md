# Experiments Overview

Chronological history of what we tested, what worked, and what failed. For raw data see the
individual files in `experiments/`. For current recipe and params see `architecture.md`.

---

## Timeline

### Phase 0 — Baseline (May 27, 2026)
**Instant-NGP bounded** on the HD multicamera scene (`007740`, 66 train + 3 eval views).
- Result: PSNR 24.42 / SSIM 0.640 / LPIPS 0.460
- Role: reference point for all subsequent comparisons.

---

### Phase 1 — LookCloser Integration (early June 2026)
Rough baseline LookCloser implementation in fixed-step mode. Frequency Grid + ARM + FR + FAS
integrated but untested. Many bugs in initial commit.

**Early Feature Reweighting strength sweep** (strength 0.25, 0.50, 1.0 from a dirty ARM recipe):
- Result: massive PSNR regression (−4 to −10 dB vs baseline). All rejected.
- Root cause: ARM had catastrophic hole artifacts at the time; FR on top amplified them.
- Stand connector ROI score 2–11 on all variants.

---

### Phase 2 — Fixed-Step Rendering as Clean Reference (June 2026)
With ARM producing holes, we pivoted to fixed-step rendering as a clean baseline.

**Fixed-640 leader** (`fixed_num_samples_per_ray=640`, 3-seed average):
- PSNR 29.565 / SSIM 0.683 / LPIPS 0.365
- ROI artifacts: 0 / 9 seeds; stand-connector score: 0.000
- This became the metric target that ARM had to beat.

Fixed768 / Fixed1024: reduced artifact score (1.478 vs 1.91) but LPIPS regressed. Rejected.
FAS on top of fixed768: LPIPS 0.394, artifact worse. Rejected.

**Takeaway:** Fixed-640 is the clean production reference for artifact comparisons.

---

### Phase 3 — ARM Artifact Investigation (June 21–22, 2026)
ARM enabled; lots of holes in metal stand and cables.

**Artifact score progression during debugging:**
| Stage | Artifact score |
|-------|---------------|
| Immediate ARM (fake freq maps) | 220.012 (catastrophic) |
| Real freq maps + delayed handoff (12288 warmup steps) | 8.065 |
| Coarse step 0.00625 + cap 2048 | 3.568 |
| Fixed-640 control | 2.182 |

ARM still worse than fixed-640 at this stage.

**Occupancy grid tuning attempts** — all rejected:
- EMA decay 0.99, dilation radius 1, occ_thre 1e-3, fallback samples 32.
- Dilation caused catastrophic failure (score 68.8).
- Debug showed artifacts project through *occupied* voxels → `grid_miss_likely=false`,
  `field_issue_likely=true`. Occupancy is not the bottleneck.

**ARM only (H40/H41) with occupancy warmup:**
- H40: PSNR 28.50 / LPIPS 0.447 / artifact 0.280 (excellent)
- H41: PSNR 29.41 / LPIPS 0.404 / artifact 0.469
- 3-seed clean rate: 1/3 on some configs. Not variance-safe.

---

### Phase 4 — ARM Bug Found and Fixed (June 22, 2026, commit `daee59bf`)

**Root cause:** `max_steps_per_ray` clipping used front-to-back rank ordering. Early high-frequency
intervals at the front of the ray exhausted the budget, leaving a gap at the far end → stand holes,
cable holes.

**Fix:** Per-ray proportional `dt` scaling. Before expanding intervals into samples:
```
over_budget = (per_ray_total / max_steps_per_ray).clamp_min(1.0)
dt = dt * over_budget[ray_indices]
```
Relative frequency-based density ratios preserved. Hard-clip kept as safety net.

File: `nerfstudio/model_components/lookcloser_samplers.py`

**After fix:** ROI artifact scores dropped to 0.000. Metal stand holes: eliminated. Small wire
holes: greatly reduced (diagnostic micro artifact floor ~0.256, significant artifact 0.000).

---

### Phase 5 — ARM-Only Baseline (June 22–23, 2026)
Post-fix ARM-only recipe stabilization. Many seeds/continuation attempts to find clean LPIPS window.

**Best ARM-only clean 3-seed set** (seed42 Huber δ0.2 / seed43 MSE / seed44 Charbonnier):
- Mean: PSNR 29.243 / SSIM 0.670 / LPIPS 0.384 / artifact 0.000
- LPIPS still worse than fixed-640 (0.365). Gap traced to field quality, not occupancy.

**Late training occupancy fix:** `occupancy_occ_thre=1e-4` from seed42 clean step 39936, dense
128-step scan, selected step 40576:
- PSNR 29.535 / SSIM 0.693 / LPIPS 0.396 / significant artifact 0.000 / micro 0.256
- Metal stand: 0, cable: 0. Residual micro on off-ROI thin pipes in eval1.

Key insight: the remaining micro artifacts are `field_issue_likely=true`, not grid misses. Sample
counts are not saturated. Longer training, stronger regularization, fresh optimizer — all tried,
none fixed the micro floor at acceptable LPIPS.

**Maxfreq13 direction:** Raising `adaptive_max_frequency_level` from 12→13 improved ARM-only LPIPS
to ~0.397. Adding `adaptive_min_frequency_level=4` with cap 2048 brought mean LPIPS to 0.384.

---

### Phase 6 — Budget-ARM + Feature Reweighting = New Leader (June 23, 2026, commit `028d4d08`)

Loaded ARM-native checkpoint (not a dense checkpoint — dense-to-ARM transfer causes PSNR collapse),
enabled Feature Reweighting and budget-aware ARM:

**Results at step 106316 (confirmed surviving checkpoint, fromscratch_s42_A_fw03):**
- PSNR 29.618 / SSIM 0.6685 / LPIPS **0.2311** / ROI artifacts: 0
- Archived: `/fsx/oregon/tank_bkup/6A_4_EXR/artifacts/static_lookcloser_leader_007740/`

Original nofas_long run peak (pruned step 91128): PSNR 29.917 / LPIPS 0.280.
Surviving nofas_long step 106316: PSNR 29.858 / SSIM 0.695 / LPIPS 0.272.

**vs. old fixed-640 leader:** +0.05 dB PSNR, −37% LPIPS.
**vs. ARM-only baseline:** significantly better LPIPS (0.231 vs 0.396).

Feature Reweighting re-enabled cleanly because ARM front-loading bug was fixed. Earlier FR tests
failed because ARM holes dominated; with zero ROI artifacts, FR provides clean LPIPS gain.

FAS at same step (parallel run): PSNR 29.877 / LPIPS 0.258 — FAS enabled by default in `ns-train
lookcloser` since Jun 23 2026.

**Set as `ns-train lookcloser` defaults** (commit `9029fe04`).

---

### Phase 7 — FAS Experiments (June 22–24, 2026)

FAS oversamples high-frequency patches. Problem: early/mid unstable geometry → stand connector
artifact on eval0 (detached floating bar).

**Two-stage approach works:** train no-FAS to a visually stable checkpoint, then continue with FAS
(`fas_strength=0.35`, `fas_level_count_alpha=1.0`, `sampling_ramp_start=end=1.0`).
- Seed42/43/44 two-stage: all passed strict stride-1 stand-connector visual gate.
- Gate: `left_stand_connector_eval0` at `--stride 1` before any FAS promotion.

**FAS vs no-FAS (reproducible checkpoints):**
- PSNR difference within noise (~0.02–0.06 dB, flips by step).
- FAS consistently better LPIPS: 0.258 vs 0.272 at step 106316; 0.249 at step 121504.
- FAS PSNR drifts down slowly after ~100k steps → must use `eval_all_psnr` checkpoint selection.

FAS enabled by default. Disable only if long run needs late-PSNR stability without selection.

**Speed optimization (Jun 24):** FAS sampler vectorized, 2x+ speedup. Training speedup analysis:
42× faster vs original non-vectorized path.

---

### Phase 8 — Implementation Doubt Follow-Ups (Rejected)

Several paper-aligned implementation alternatives tested and reverted:

| Test | Result | Decision |
|------|--------|----------|
| Paper-aligned runtime grid updates (patch center rays) | Regressed 3-seed SSIM/LPIPS/PSNR | Reverted |
| FAS with only non-empty bucket probabilities | Improved LPIPS/PSNR, regressed SSIM | Reverted |
| Sparse SfM frequency-grid initialization | Regressed all 3 metrics | Reverted |
| Sparse depth supervision (COLMAP) | Regressed all 3 metrics | Reverted |
| `trunc_exp` density activation (Instant-NGP style) | Regressed crops | Rolled back |
| 32D appearance embedding | Regressed early eval loss | Rejected |

---

## Dead Ends Summary

| Approach | Why Failed |
|----------|-----------|
| Occupancy dilation / EMA decay / occ_thre tuning | Artifacts are `field_issue_likely`, not grid misses |
| Fallback samples for ARM (32 per ray) | artifact score 26.266 (catastrophic) |
| Dense fixed integration > 640 samples | LPIPS regression; LPIPS gate failure |
| uniformmax15 (ARM level forced 15) | PSNR 13.84, catastrophic collapse |
| Early aggressive FR (strength 0.5/1.0 on dirty ARM) | −4 to −10 dB PSNR regression |
| `occupancy_grid_levels=2` | Collapsed by first eval (~14 PSNR) |
| `color_num_layers=3` | Collapsed by first eval |
| `field_hidden_dim=128` | Collapsed, 3208× artifact score |
| `log2_hashmap_size=24` | Collapsed by second eval |
| `hash_features_per_level=4` + Charbonnier | Good metrics but artifact dirty; not variance-safe |
| Late higher frequency caps (maxfreq14) | Seed44 not confirmed; dirty without cap2048 |
| Frequency-grid validity filtering | Weakened frequency-map hypothesis; reverted |
| Sticky binary occupancy retention | Metric-improved regime dirty; field/trajectory cause |
| Simple loss switches from leader (Charb, Huber, MSE) | None beat micro 0.256 |
| Simple longer training from leader | Enters occupancy grid-miss regime past ~45k |
| Per-ray full-occupancy ARM batch mix | Worse micro; temporary code removed |
| High-frequency RGB loss weighting | Enters artifact-heavy regime |
| `las_sample` background compositing | PSNR 22.15, micro artifact 558 |
| Frequency-grid boundary index patch | No improvement; reverted |

---

## Next Steps (High Level)

1. **FAS integration with visual gate** — currently enabled in defaults; strictly gate future FAS
   variants with `left_stand_connector_eval0 --stride 1` before promoting.

2. **Scale-matched comparison vs Instant-NGP** — visual audit showed LookCloser smooths thin
   cables/stand label despite higher PSNR. Need apples-to-apples crop comparison.

3. **Residual micro artifact floor** — 0.256 diagnostic micro in off-ROI eval1 thin pipe.
   Debug: `field_issue_likely=true`, not grid miss, not sample-cap saturation. Next lever:
   frequency-map refinement around the failing eval1 cable segment, or ARM interval scheduling.

4. **Temporal 4D extension** — separate `time` branch; best result: SSIM 0.800 / LPIPS 0.308,
   0/51 artifacts. ARM ported to 4D with baked frequency grids. Out of scope for static scene.
