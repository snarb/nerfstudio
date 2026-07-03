# Motion-aware ray sampling — implementation plan (for temporal model; cameras static)

From a focused literature review (DyNeRF, K-Planes, MixVoxels). Cameras are STATIC across frames →
per-camera temporal frame differencing is a clean, ghost-free motion signal (no optical flow needed).

## Decisions
- **Motion signal = DyNeRF IST** (as implemented in K-Planes `video_datasets.py`): per camera, per pixel,
  `W = clamp_max( mean_RGB( max_{|i-j|<=25} |C_ti - C_tj| ), alpha_hi=0.1 )`. Optionally ISG
  (deviation from per-pixel temporal median, Geman-McClure γ=2e-2) for an early "global" phase.
- **Robustness adds:** deadband `relu(diff - 2..4/255)`; 7px max-pool dilation (fill moving-object
  interiors); compute on 4× downsampled frames (memory), upsample at sample time; small floor on p_motion.
- **Mix = HARD blend** per ray: `n_motion = round(B·α)`, rest uniform; concat + shuffle. (Cleaner than
  soft mixture; guarantees static coverage, no starvation.) **α target 0.3; sweep 0.2/0.3/0.4 (=80/20,70/30,60/40).**
- **Schedule:** α=0 for warmup (~0–2k steps) → linear ramp to target (~2k–10k) → hold. (DyNeRF "uniform→IS".)
- **Combine with frequency (FAS) = 3-way hard split**, NOT a product: e.g. uniform 50 / freq 25 / motion 25
  (also test 50/30/20 and 50/20/30). Motion and frequency are largely orthogonal (temporal change vs spatial
  detail) → union semantics via independent per-source categoricals avoids double-counting.
- **Eval:** IST mainly improves LOCAL dynamic regions (K-Planes finding) → also measure on motion-masked /
  dynamic-region crops, not just full-frame PSNR. (And the significant_artifacts_score, target 0.)

## Implementation (mirror existing FAS sampler)
- Precompute per-(camera,frame) IST maps offline → cache `motion_weights.pt` keyed by dataparser config.
  Need (camera_id, frame_order) per image_idx from the dataparser (filename/time). Group by camera, diff in time.
- New sampler subclass (pattern of `nerfstudio/lookcloser_pixel_sampler.py`, `sample_method` ≈L387):
  two-level sampling — pick image via `multinomial(image_weights ∝ Σ motion, n_motion)`, then
  `multinomial(per-image pixel_prob, count)` for (row,col). Keep the `[B,3]=(local_img_idx,row,col)` contract
  (collate remaps to absolute camera idx; ray_generators unchanged).
- Config: `motion_strength`, `motion_warmup_steps`, `motion_ramp_steps` (reuse FAS schedule fields);
  for 3-way: `uniform_frac / freq_strength / motion_strength`.

## Starting hyperparameters
IST window ±25, α_hi=0.1, deadband 2–4/255, dilation 7px, 4× weight res, p_motion floor ~few % of mean,
motion fraction 0.3, schedule warmup 2k → ramp to 10k → hold, ISG γ=2e-2 (optional early), combine 50/25/25.

## Key files
`nerfstudio/data/pixel_samplers.py` (base sample_method ≈L137, collate ≈L265),
`nerfstudio/lookcloser_pixel_sampler.py` (FAS to mirror), `nerfstudio/model_components/ray_generators.py` (no change).
NOTE: motion sampling needs `VanillaDataManager` (per-image pixel sampling) — same datamanager constraint as FAS;
the temporal model's `ParallelDataManager` streaming would need the sampler ported into RayBatchStream, or switch
to VanillaDataManager + bounded image cache (see temporal_lookcloser_integration_plan.md).

Sources: DyNeRF arXiv:2103.02597; K-Planes arXiv:2301.10241 (IST/ISG code); MixVoxels arXiv:2212.00190.

---

## UPDATE 2026-06-29 — motion sampling AS PART OF FAS (user directive)

Motion sampling must be integrated INTO frequency-aware sampling (FAS), not standalone. Target a single
combined pixel sampler with a **3-way hard split per batch**: `uniform / frequency(FAS) / motion(IST)`.

**Prerequisites:**
1. IST motion maps — precomputing now (agent acf2663c) → `temporal_ngp_ds_eval12/motion_maps/`.
2. FAS frequency maps for the temporal train set — NOT yet done. Cameras are STATIC → preprocess ONE
   frequency map per camera (52 maps) and reuse across that camera's frames (cheap vs per-image). Reuse
   `nerfstudio/scripts/lookcloser_preprocess.py`. (GPU-bound 2D-NGP per camera; run after the full run.)
3. A combined sampler (VanillaDataManager + custom pixel sampler; 2523 full-HD imgs ≈ 16GB RAM fits the
   30GB box once the streaming run frees RAM) OR port the 3-way split into the existing FAS sampler
   `nerfstudio/lookcloser_pixel_sampler.py`.

**Experiments (short screens, parallel 2–3 since light):**
- Split ratios (uniform/freq/motion): **50/25/25** (start), 50/30/20 (freq-heavy), 50/20/30 (motion-heavy).
- Motion fraction α schedule: warmup 0 → ramp to target by ~10k (reuse FAS warmup/ramp fields).
- Compare vs the artifact≈0 winner (eval-12th) on LPIPS + the moving-people ROI (motion-masked) + artifact
  score. Motion's value is LOCAL (dynamic regions) — measure on motion-masked crops, not just full-frame.
- Combine op = hard 3-way split (NOT product; product starves freq-static + low-freq-moving).

**Sequencing:** (a) finish + extend the current full run until eval plateaus; (b) preprocess freq maps
(per-camera) once GPU frees; (c) implement combined sampler; (d) run the split-ratio/α screens (parallel).

### IST maps DONE (2026-06-29) — with signal deviation
`motion_maps/motion_weights.pt` (2523 maps, float16 [270,480] = 4× down). **Signal = ISG (|C_t − median_τ C_τ|),
NOT plain IST** — plain temporal-diff saturated everywhere due to heavy film grain/lighting flicker + fast
motion + every-4th sampling (static-bg temporal std ~28/255). ISG (median residual) is the documented
alternative and localizes on the fighters + moving shadows. Robustness: deadband relu(dev−8/255), clamp 0.1,
7px dilation. Key = image filename stem.
**Caveat:** motion concentration modest (~1.6× over bg; 84% pixels nonzero — noisy footage bleeds onto static
wall). ⇒ use LOWER α and/or tighter deadband (15–20/255); expect a gentle bias, not a sharp mask. 3 cams have
single/few frames (M004_E014, O004_E014 all-zero maps; L004_B014 partial) → sampler floor keeps them uniform.

### DECISION (2026-06-29): motion signal = PERSON MASKS, drop ISG post-processing
User: since only people move, just use the human masks directly — no ISG temporal-diff needed. So the
combined sampler = uniform / frequency(FAS) / **motion(person-mask)**. ISG maps are deprecated (kept only
as a fallback where YOLO finds no person). Simpler + cleaner signal (sharp on people, no static-wall bleed).

### IMPLEMENTATION DONE + experiments started (2026-06-29)
- Person masks DONE: `person_masks/person_masks.pt` (2574 entries).
- Combined sampler `nerfstudio/motion_fas_pixel_sampler.py` (MotionFASPixelSampler: mode split/region/off)
  + method `instant-ngp-time-fasmotion` (DynamicBatch + VanillaDataManager + sampler + H2 winner model).
  Merged from worktree into main, registers + unit-tested OK.
- Runner extended: `--method` + `--ps key=val` passthrough (e.g. `--ps mode=region --ps person-frac=0.8`).
- FIRST experiment launched: Variant B region pf=0.8, 15k screen, eval-12th data, vs the 26.21/0.341/0.004
  winner. Then Variant A (50/25/25) + FAS-only (mode=off) for ablation. Parallelism decided by VRAM probe.

### BUG (2026-06-29): sampler vs bounded image cache → fix in progress
First fasmotion run OOM'd (VanillaDataManager full-cached 2523 full-HD imgs ≈16GB → 30GB box OOM). Bounded
the cache (train_num_images_to_sample_from=400) → then IndexError: sampler emits GLOBAL image indices (769)
but batch has only 400 cached imgs. FAS/MotionFAS sampler assumed full cache (local==global idx). Also note:
downscale can't be used as a RAM fix (freq maps are full-HD-aligned, patch grid would mismatch). FIX (agent
af6e0546, applied to MAIN checkout): sample LOCAL image positions within the batch, map local→global via
batch["image_idx"] for freq/mask lookup. This also makes the final 52-cam full-HD fasmotion run feasible
(bounded cache). GPU idle until fixed.

### BLOCKER (2026-06-29): FAS/motion sampler too SLOW to train with here
After fixing the subset-index bug, the sampler runs but at **178–846 rays/sec vs ~94k for the winner
(~100–500× slower)**. Cause: MotionFASPixelSampler/LookCloserPixelSampler do Python per-batch frequency
bucketing + per-image mask gating at full-HD; that dominates each step (the winner was fast via
ParallelDataManager's internal vectorized streaming sampler). RAM also pressured VanillaDataManager (cache=400
→27.8GB/swap; cache=64 fits but is the slow regime). Sampler is CORRECT (unit-tested) but impractical to
train with as-is. Needs a vectorized rewrite (precompute per-image categorical sampling distributions once;
torch.multinomial) before benchmarking — OR finalize on the core result. Escalated to user.

### VARIANT C (user idea 2026-06-30): mask-gated static/dynamic DECOMPOSITION
Specialize branches by person mask: 4D dynamic branch learns the people (mask pixels), 3D static branch
learns the background (non-mask pixels) → faster + fewer artifacts. DESIGN NOTE: don't hard-route whole rays
(a person-pixel ray still traverses static background behind the person). Implement as ADDITIVE
decomposition + mask-gated dynamic-sparsity loss (D²NeRF/NeRF-W style):
- field exposes SEPARATE static(3D, time-independent) and dynamic(4D) density/color contributions; combine
  (σ = σ_static + σ_dynamic; color blended by per-sample contribution).
- loss: on NON-mask pixels penalize dynamic contribution → ~0 (static explains background); on mask pixels
  dynamic is free. (Optionally also a small global dynamic-sparsity prior.)
- Requires arch tweak to TemporalNGPField (split the H2 concat-MLP into static + dynamic heads so dynamic is
  regularizable) + mask-weighted loss term in get_loss_dict (per-ray mask from person_masks).
EXPERIMENT after the A/B sampling screens: compare vs winner on artifact + LPIPS + speed-to-converge.
This is complementary to sampling (could combine: decomposition + region sampling).

### SAMPLER VECTORIZED (2026-06-30) → experiments running
User chose "optimize then run". Rewrote sampler: precompute per-image freq + motion pmfs once → two-level
torch.multinomial. **66k–1M rays/sec (was 178–846; ~130–1000× faster)**, all modes ≥30k, unit tests pass,
subset-safe. Live in main. Launched 2 screens IN PARALLEL (cache=64 ≈ winner's streaming regime; RAM ~22/28
avail, VRAM ~8GB each): Variant B region pf=0.8 + Variant A split 50/25/25, 15k/eval5k on eval-12th, vs
winner 26.21/0.341/0.004. Next: FAS-only (mode=off) + uniform control. Waiters byadgm6f1, b06zcqrj4.

### A/B RESULT (2026-06-30, 15k screens) — motion/freq oversampling HURTS here
| run | PSNR | SSIM | LPIPS | artifact mean | n |
|---|---|---|---|---|---|
| plain eval-12th baseline (no FAS/motion) | 25.70 | 0.750 | 0.400 | 0.136 | 17/51 |
| Variant A split 50/25/25 | 24.74 | 0.698 | 0.497 | 0.471 | 28/51 |
| Variant B region pf=0.8 | 23.92 | 0.670 | 0.569 | 0.485 | 34/51 |
Both WORSE than uniform; heavier people-oversampling (B, 80%) worst → likely BACKGROUND STARVATION (full-frame
artifact ↑). INSIGHT: once eval-coverage drove artifacts→~0, moving-people detail wasn't the bottleneck, so
biasing rays toward people hurt the under-sampled background. CONFOUND to rule out: cache=64+VanillaDataManager+
DynamicBatch vs baseline's ParallelDataManager streaming → running uniform + FAS-only CONTROLS on the SAME
setup (waiter byko3spks). If uniform-ctrl≈0.14 → motion hurts; if ≈0.47 → it's the setup, not the sampling.

### CONTROLS RESULT (2026-06-30) — it's the SETUP, motion itself helps slightly
Same setup (cache64/VanillaDM/DynamicBatch, 15k): uniform-ctrl artifact **0.544** (PSNR 24.32), FAS-only
**0.625** (worst), Variant A **0.471** (best), Variant B 0.485. ⇒ (1) MOTION sampling HELPS vs uniform
(0.471<0.544), FREQUENCY-only HURTS (0.625). (2) But the whole VanillaDM+DynamicBatch setup is ~3–4× worse
than the winner's streaming setup (0.47–0.54 vs 0.006) → the setup penalty dwarfs motion's gain.
CONCLUSION: pixel-sampler FAS+motion is NOT a net win here (requires VanillaDataManager, which underperforms
streaming). PIVOT to Variant C (mask-gated decomposition = architecture change, runs on the winning STREAMING
setup, no VanillaDM penalty). DynamicBatch-vs-Vanilla isolation is a possible last check for sampling, but
low expected payoff. Winner stays 26.21/0.341/0.006.

### DECISION: implement Variant C (mask-gated decomposition) — runs on STREAMING winner setup
User chose Variant C. Implementing as `instant-ngp-time-decomp`: separate static(3D)/dynamic(4D) density+color
heads, σ=σ_s+σ_d, blended color; mask-gated dynamic-sparsity loss (penalize dynamic accumulation on NON-person
rays). Per-ray person label attached in ParallelDataManager.next_train (masks loaded once in MAIN process,
looked up via batch['indices']=(img,row,col) → person_masks[stem]). Uses the WINNER streaming setup (VanillaPipeline
+ ParallelDataManager load_from_disk) so NO VanillaDataManager penalty. Delegated to impl agent.

### VARIANT C IMPLEMENTED + RUNNING (2026-06-30)
`instant-ngp-time-decomp` registered (VanillaPipeline + TemporalDecompDataManager(streaming) + TemporalDecompModel,
field H2D). Separate static(3D)/dynamic(4D) density+color heads, σ=σ_s+σ_d, blended color; per-ray
dynamic_accumulation; loss `dynamic_sparsity_mult·mean((1-is_person)·dyn_accum)` (default 0.05); per-ray
person label attached in next_train (masks in MAIN proc, lookup via batch['indices']). 4/4 CPU unit tests pass.
Running 30k screen (--hypothesis H2D, winner hash settings, dyn_sparsity 0.05) vs winner 30k point
(25.93/0.367/artifact 0.006). Waiters bsvm2wx4q (startup), b5w5y82y0 (completion). On streaming setup → full speed.

### VARIANT C RESULT (2026-06-30): COLLAPSED
First fix: H2D isn't a valid model-config hypothesis literal; decompose=True forces it (pass --hypothesis H2).
Then trained at 35k rays/s (full speed, streaming) but **COLLAPSED: PSNR ~14 / SSIM 0.52 / LPIPS 0.92 /
artifact 2.49 (51/51)**, flat from step 5k → broken-field signature, not slow convergence. Far worse than
winner (26.21/0.341/0.006). Likely cause: dynamic-sparsity loss over-suppresses (penalizes dynamic on ~91%
non-person rays) OR dual-branch (σ_s+σ_d two trunc_exp + color blend) instability. Diagnostic running:
dynamic_sparsity_mult=0 (loss off) → if PSNR recovers it's the loss; if still ~14 it's the architecture
(waiter bjyxf1ku3). Added runner --mm (model-flag passthrough).
NOTE: 3 enhancement directions (FAS sampling, motion sampling, decomposition) have not beaten the core
winner; artifact was already ~0.006 (little headroom). May finalize on the core result after this diagnostic.

### DIAGNOSTIC (2026-06-30): collapse = the LOSS, not the architecture
dyn_mult=0 (loss OFF): decomp trains fine → step5k PSNR 24.49 / SSIM 0.721 / LPIPS 0.456 / artifact 0.187.
⇒ H2D architecture is sound; dynamic_sparsity_mult=0.05 was WAY too strong (collapsed to PSNR 14).
But loss-off decomp (24.49@5k) is slightly BEHIND the concat-H2 winner (~25.1@5k), and winner artifact is
already 0.006 → little/no headroom for decomposition to beat it even with a tuned (tiny) loss weight.
DECISION POINT: finalize on core winner vs one tuned-loss retry (dyn_mult~0.001).
`/opt/dlami/nvme/temporal_ngp_ds_eval12/lookcloser_frequencies/` — all train stems covered (52 per-camera
maps + symlinks), params MATCH static reference (patch_size=8, stride=8, min_res=16, max_res=8192, n_levels=16,
log2=23, ssim_thr=0.95), shape (135,240), validated. FAS sampler path convention satisfied. ~2.45h GPU.

### EXPERIMENT VARIANT B (user 2026-06-29): region-gated FAS (high motion rate)
Instead of the 3-way uniform/freq/motion split, partition pixels by the PERSON MASK and allocate a high
fraction to the person region, with FAS applied WITHIN each region:
- **~80% of rays → person-mask pixels, sampled by FREQUENCY (FAS) within the mask** (so high-freq detail on
  the moving people gets the rays).
- **~20% of rays → background (outside mask), by normal FAS.**
Sweep the person fraction (e.g. 70/30, 80/20, 90/10). Compare vs Variant A (3-way split) and vs the
artifact≈0 winner. This concentrates capacity on the hard dynamic content while staying frequency-aware.
Implementation: per image, split pixel indices into mask vs ¬mask using the dilated person mask; run the FAS
bucket sampler separately on each subset with the allocated ray counts; concat+shuffle. Reuses the
person_masks cache + the FAS frequency maps.

### YOLO person masks (the motion signal; agent ab38c132)
Since the ONLY moving content is people, a YOLO person-seg mask (dilated 1–2px) is a CLEANER motion signal
than ISG. Generating per-image masks (Ultralytics yolo11x-seg/yolov8x-seg, person class) → cache
`person_masks/person_masks.pt` {stem:[270,480]} matching the motion_weights convention. Built in a
DEDICATED clean venv (`/opt/dlami/nvme/yolo_env`) to avoid clashing with the nerfstudio env; runs CPU/tiny-GPU
to not OOM the training. EXPERIMENT: A/B the motion signal in the FAS+motion sampler — ISG map vs person-mask
vs (mask ∪ ISG) — on the moving-people ROI + LPIPS.
