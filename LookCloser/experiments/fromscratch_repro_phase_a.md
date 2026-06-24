# Phase A: From-Scratch Reproduction — LookCloser Leader

## What was tested

Reproduce LookCloser leader (PSNR≥29.86, SSIM≥0.695, LPIPS≤0.272) from scratch
on scene 007740, seed 42, in a single 200k-step run.

**Target:** PSNR≥29.86 | SSIM≥0.695 | LPIPS≤0.272 | ROI artifact=0

---

## Results — All runs (Phase A)

### Single-stage from-scratch (Runs A, B, C) — seed 42

| Step | A (FW=1.0, warmup=4096) | B (warmup=8192) | C (FW=1.0, max_freq=13) |
|------|------------------------|-----------------|--------------------------|
| 15188 | 28.596 / 0.652 / 0.372 | 28.352 | 28.582 |
| 30376 | 29.210 / 0.676 / 0.306 | 29.110 | 29.197 |
| 45564 | 29.395 / 0.674 / 0.280 | 29.363 | 29.378 |
| 60752 | 29.528 / 0.677 / 0.262 ✓ | — | 29.487 |
| 75940 | **29.622** / 0.675 / 0.253 | — | 29.549 |
| 91128 | **29.683** / 0.670 / 0.240 | — | 29.604 |
| 106316 | 29.585 (dip) / 0.669 / 0.232 | — | 29.518 |
| 121504 | 29.587 / 0.669 / 0.226 | — | — |
| 136692 | 29.587 plateau | — | — |

**Best result ever: A step 91128 → PSNR=29.683** (0.177 dB below target)

### FW strength comparison (from A@75940, seed 42)

| FW | @91128 | @106316 | trend |
|----|--------|---------|-------|
| 0.0 | 29.680 | 29.590 | dip then plateau ~29.59 |
| 0.3 | 29.692 | 29.618 | dip then plateau ~29.62 |
| **1.0** | **29.683** | 29.585 | **dip then plateau ~29.59** |

Finding: dip is universal regardless of FW strength — FW is NOT the root cause.

### Staged path experiments

| Variant | Stage1 PSNR | @60752 | @75940 | @91128 | Notes |
|---------|------------|--------|--------|--------|-------|
| stage2 (C@91128 cont-LR) | 29.604 | — | — | — | plateau ~29.54 |
| stage2_from30k (stage1@30376, FW=1.0) | 29.153 | 29.555 | 29.579 | 29.558 | FW early plateau |
| stage2_from45k (stage1@45564, FW=1.0) | 29.342 | 29.495 | 29.574 | 29.521 | FW early plateau |

---

## Key findings

### Finding 1: Universal PSNR dip at step ~91–106k
Every run shows a PSNR dip/plateau at step ~91–106k regardless of FW strength,
starting checkpoint, or staged path design. LPIPS keeps improving through this zone
(FW adaptation signature). The dip is at the same **absolute step number**, not a
fixed number of continuation steps.

### Finding 2: PSNR ceiling at ~29.58–29.68 in single-run 200k training
No single 200k-step from-scratch run reached PSNR > 29.683. The best trajectory
(original A) peaked at 29.683 then plateaued at 29.587.

### Finding 3: Root cause — multi-stage LR resets
The historical leader was produced through **multiple short continuation runs**,
each with a **different `max_num_iterations`**. Key example: `maxfreq13_cont3` used
`max_num_iterations=40960`, so at step 38912 (95% of 40960) the LR ≈ 0.00016.
When stage 2 loaded that checkpoint with `max_num_iterations=200000`, the scheduler
placed step 38912 at 19.5% of the new 200k schedule, giving LR ≈ 0.009 — nearly
maximum. This "fresh high-LR start" allowed rapid improvement from 29.535 to 29.858
in 67k steps.

Our single 200k runs never get this effect. At step 91128 (45.6% of 200k), LR ≈
0.00575 — still in active decay, no LR reset.

### Finding 4: Historical stage 1 had FW=false
The `maxfreq13_cont3` config shows `enable_feature_reweighting: false`. Our from-
scratch runs all had FW=1.0 throughout. With FW=off, the hash grid develops without
high-frequency bias, producing a field more suitable for stage 2 transition.

### Finding 5: Staged path with stage1_proper (FW=off, max_iter=50000) did not help
stage1_proper reached PSNR ~29.15–29.34 at step 30376–45564 with low LR (≈0.001).
Stage 2 from those checkpoints got a LR boost (0.001→0.0075) but still hit the same
FW-driven plateau at step ~75–91k. The starting PSNR was too low to recover to 29.86.

---

## Recommended approach for successful Phase A

To replicate the leader's training history exactly:

1. **Stage 1a:** `max_iter=20000`, seed=42, `max_freq=13`, `FW=off`
   → at step ~19000 (95% done), LR ≈ 0.0002. Reach PSNR ~28.8.

2. **Stage 1b:** Load stage1a checkpoint, `max_iter=35000`, FW=off
   → at step ~33000, LR ≈ 0.0003. Reach PSNR ~29.1.
   LR at step 33000 in 200k schedule = 0.0085 (nearly max → fresh start effect).

3. **Stage 1c:** Load stage1b, `max_iter=50000`, FW=off
   → at step ~47500, LR ≈ 0.0002. Reach PSNR ~29.4.

4. **Stage 2:** Load stage1c's best checkpoint (PSNR ~29.4–29.5),
   `max_iter=200000`, `FW=1.0`, no freq cap, full leader recipe.
   → At step ~47.5k in 200k schedule: LR ≈ 0.0089 (nearly max → fresh start!).
   → Expected: rapid improvement similar to historical leader.

Each continuation gives a "fresh LR start" at higher PSNR, mimicking the historical
multi-stage path. Total GPU time: ~3–4 hours per stage.

---

## Best checkpoint to use for next attempt

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_fromscratch_s42_A/lookcloser/20260623_142529/nerfstudio_models/step-000075940.ckpt`
PSNR=29.622, SSIM=0.675, LPIPS=0.253 — best clean (non-corrupted) checkpoint from
Run A. If continuing the staged approach, this is the highest-quality starting point.

---

## Insights

- Phase A target NOT reached in this session. Best: PSNR=29.683 (step 91128, Run A).
- The bottleneck is structural: single 200k-step runs cannot replicate the historical
  leader's multi-stage LR reset dynamics.
- FW at strength 1.0 is appropriate for stage 2 continuation but causes premature
  plateau in single long runs from scratch.
- LPIPS consistently reaches target (~0.24) well before PSNR plateaus, confirming
  FAS is working correctly.
- The 29.86 target IS achievable — the leader proved it — but requires the proper
  multi-stage protocol described above.
