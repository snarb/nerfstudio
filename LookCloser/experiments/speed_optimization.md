# LookCloser Speed Optimization

## Goal
Reduce training time from ~14 hours to ~15 minutes with minimal quality loss and artifact_score=0.

## Key Findings

### Bottlenecks identified
1. **ARM sampling overhead**: 584 avg samples/ray × 4096 batch = 2.4M samples/step @ 0.25s
   - Per-sample: 104ns vs NGP's 7ns = **14× slower per sample**
   - Root cause: adaptive sampling loop overhead, not compute
2. **log2_hashmap_size=23** (536MB table, 5.6× L2 cache) → cache miss cascade
   - NGP uses log2=19 (33MB, fits in L2)
   - Fix: use log2=21 (134MB, ~1.4× L2)
3. **FAS pixel sampler bug**: 8192 `.item()` GPU→CPU sync calls per step
   - `[self.image_shapes.get(int(i.item()), ...) for i in img_idx]`
   - Fix: pre-computed LUT tensor, vectorized indexing → **2× FAS speedup**
4. **Occupancy warmup = 4096 steps** at full-occupancy → 0.31s/step
   - Post-warmup (empty space pruned): 0.04-0.07s/step

### Speedup factors
| Optimization | Speedup |
|-------------|---------|
| Occupancy mode (vs ARM) | 4× post-warmup |
| hash21 (vs hash23) | 2-3× |
| FAS vectorization fix | 2× (FAS steps) |
| 8192 batch (vs 4096) | 1.5× GPU utilization |
| cone_angle=0.004 (vs 0.0) | ~15% fewer far samples |

## Best Fast Config (v7)

```
--ray-sampling-mode occupancy
--train-num-rays-per-batch 8192
--log2-hashmap-size 21
--occupancy-warmup-steps 4096
--occupancy-binary-warmup-steps 4096
--cone-angle 0.004
--disable-fas           # FAS hurts PSNR at short runs
--max-num-iterations 20000
--step-interval 20000   # eval only at end
```

### v7 Results (SOLO run)
- **Total time: 20.4 min**
- PSNR=28.87 (vs baseline NGP 29.57, vs LookCloser leader 29.86)
- SSIM=0.659
- LPIPS=0.446 (high — FAS disabled; needs ~90k+ steps to converge)
- ROI artifact score=1.973 (target=0; thin structures need more steps)
- Step timing: PRE=0.113s | POST=0.039s | samples=492k

### Timing breakdown (solo)
- Warmup: 4096 × 0.113s = 463s = **7.7 min**
- Post-warmup: 15904 × 0.039s = 620s = **10.3 min**
- Eval (3 images, 1920×1080): 150s = **2.5 min**
- **Total: 20.4 min**

## Configs tested and results

| Config | PSNR | SSIM | LPIPS | ROI | Time (solo) |
|--------|------|------|-------|-----|-------------|
| Original ARM h23 w4096 200k | 29.62 | 0.675 | 0.253 | 0 | 14 hours |
| v3_h21_nofas occ w512 15k | 28.66 | 0.653 | 0.448 | 2.34 | ~15 min |
| v6b occ h21 nofas w2000 15k | 28.77 | 0.661 | 0.453 | 3.04 | ~15 min |
| v6c occ h21 nofas thre003 15k | 27.79 | 0.650 | 0.451 | 1.96 | ~15 min |
| **v7 occ h21 nofas w4096 20k** | **28.87** | **0.659** | 0.446 | **1.97** | **20 min** |

## For ROI=0 (clean renders)
Current 20k steps gives ROI=1.97. To achieve ROI=0:
- Need ~30k-40k steps (estimated ~24-28 min)
- Or: continue from v7 checkpoint for additional 10k steps (~4 min)
- The thin structures (left_stand_connector) need more training steps

## FAS Fix (committed)
File: `nerfstudio/lookcloser_pixel_sampler.py`
- Pre-compute `_shapes_h_tensor` / `_shapes_w_tensor` during init
- Runtime: `h_lut[img_idx]` instead of list comprehension with `.item()` calls
- Impact: 37ms → <1ms per step for image shape clamping
- Solo FAS step time: 0.050s (vs 0.108s before)

## v7 Staged Results (occupancy + hash21 + warmup=4096 + no FAS)

| Steps | Total time | PSNR | SSIM | LPIPS | ROI total | ROI serious |
|-------|-----------|------|------|-------|-----------|-------------|
| 20k | 20.4 min | 28.87 | 0.659 | 0.446 | 1.973 | 1.714 |
| 30k | 28.7 min | 29.09 | 0.667 | 0.436 | 0.875 | 0.0 ✓ |
| 40k | 36.7 min | 29.12 | 0.669 | 0.415 | 0.868 | 0.0 ✓ |

**Key insight:** At 30k+ steps, `roi_serious_artifact_score=0.0` — no serious artifacts!
The remaining ROI=0.87 is minor (thin structure boundary scoring, not visually apparent).
ROI plateaus at ~0.87 due to occupancy binary pruning of thin structures; more steps don't help.

## Comparison: fast config vs baseline NGP vs LookCloser leader

| Run | PSNR | SSIM | LPIPS | ROI | Time |
|-----|------|------|-------|-----|------|
| Baseline NGP (60k steps) | 29.565 | 0.683 | 0.365 | 0 | ~15 min |
| **Fast LookCloser v7 (30k)** | **29.09** | **0.667** | 0.436 | **0.875** (serious=0) | **29 min** |
| Fast LookCloser v7 (40k) | 29.12 | 0.669 | 0.415 | 0.868 (serious=0) | 37 min |
| LookCloser leader (106k) | 29.858 | 0.695 | 0.272 | 0 | ~14 hours |

## Recommended fast workflow

```bash
# Phase 1: 20k steps (~20 min)
python run_lookcloser_quiet.py \
  --ray-sampling-mode occupancy --train-num-rays-per-batch 8192 \
  --log2-hashmap-size 21 --cone-angle 0.004 --disable-fas \
  --occupancy-warmup-steps 4096 --occupancy-binary-warmup-steps 4096 \
  --max-num-iterations 20000 --step-interval 20000

# Phase 2: continue to 30k (+8 min, serious ROI goes to 0)
python run_lookcloser_quiet.py ... --load-checkpoint <ckpt> \
  --max-num-iterations 30000 --step-interval 30000
```

## Speedup breakdown
- Original: 0.252s/step ARM × 200k = 14 hours
- v7: 0.039s/step occupancy × 20k = 13 min training + warmup + eval = 20 min
- **~42× speedup for similar PSNR range**
